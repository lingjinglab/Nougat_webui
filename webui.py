import time

import gradio as gr
from datetime import datetime
import os
import sys
from functools import partial
from PIL import Image
from pathlib import Path
import hashlib
import fitz
import torch
from nougat import NougatModel
from nougat.postprocessing import markdown_compatible, close_envs
from nougat.utils.dataset import ImageDataset
from nougat.utils.checkpoint import get_checkpoint
from nougat.dataset.rasterize import rasterize_paper
from tqdm import tqdm



SAVE_DIR = Path("./pdfs")
BATCHSIZE = os.environ.get("NOUGAT_BATCHSIZE", 6)
NOUGAT_CHECKPOINT = get_checkpoint()
model = None
if NOUGAT_CHECKPOINT is None:
    print(
        "Set environment variable 'NOUGAT_CHECKPOINT' with a path to the model checkpoint!."
    )
    sys.exit(1)

def load_model(
    checkpoint: str = NOUGAT_CHECKPOINT,
):
    global model
    if model is None:
        model = NougatModel.from_pretrained(checkpoint).to(torch.bfloat16)
        if torch.cuda.is_available():
            model.to("cuda")
        model.eval()




def upload_file(file):
    # 在这里处理上传的文件
    pass

def read_HTML_template():
    # 打开文件
    file = open('./covert.html', 'r')
    # 读取文件内容
    content = file.read()
    # 关闭文件
    file.close()
    return content


def run_orc(file, start, stop):
    pdf = fitz.open(file.name)
    md5 = hashlib.md5(file.name.encode()).hexdigest()
    save_path = SAVE_DIR / md5

    if start is not None and stop is not None:
        pages = list(range(start - 1, stop))
    else:
        pages = list(range(len(pdf)))
    predictions = [""] * len(pages)
    dellist = []
    if save_path.exists():
        for computed in (save_path / "pages").glob("*.mmd"):
            try:
                idx = int(computed.stem) - 1
                if idx in pages:
                    i = pages.index(idx)
                    print("skip page", idx + 1)
                    predictions[i] = computed.read_text(encoding="utf-8")
                    dellist.append(idx)
            except Exception as e:
                print(e)
    compute_pages = pages.copy()
    for el in dellist:
        compute_pages.remove(el)
    images = rasterize_paper(pdf, pages=compute_pages)
    global model

    dataset = ImageDataset(
        images,
        partial(model.encoder.prepare_input, random_padding=False),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCHSIZE,
        pin_memory=True,
        shuffle=False,
    )

    for idx, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
        if sample is None:
            continue
        model_output = model.inference(image_tensors=sample)
        for j, output in enumerate(model_output["predictions"]):
            if model_output["repeats"][j] is not None:
                if model_output["repeats"][j] > 0:
                    disclaimer = "\n\n+++ ==WARNING: Truncated because of repetitions==\n%s\n+++\n\n"
                else:
                    disclaimer = (
                        "\n\n+++ ==ERROR: No output for this page==\n%s\n+++\n\n"
                    )
                rest = close_envs(model_output["repetitions"][j]).strip()
                if len(rest) > 0:
                    disclaimer = disclaimer % rest
                else:
                    disclaimer = ""
            else:
                disclaimer = ""

            predictions[pages.index(compute_pages[idx * BATCHSIZE + j])] = (
                    markdown_compatible(output) + disclaimer
            )

    (save_path / "pages").mkdir(parents=True, exist_ok=True)
    pdf.save(save_path / "doc.pdf")
    if len(images) > 0:
        thumb = Image.open(images[0])
        # thumb.thumbnail((400, 400))
        thumb.save(save_path / "thumb.jpg")
    for idx, page_num in enumerate(pages):
        (save_path / "pages" / ("%02d.mmd" % (page_num + 1))).write_text(
            predictions[idx], encoding="utf-8"
        )
    final = "".join(predictions).strip()
    (save_path / "doc.mmd").write_text(final, encoding="utf-8")
    return final


static_dir = Path('./static')
static_dir.mkdir(parents=True, exist_ok=True)

#创建iframe用于转换数据格式
def build_iframe(content):
    html = read_HTML_template()
    content = content.replace("\\","\\\\")
    content = content.replace("\n", "\\n")
    html = html.replace("{{__doc__}}", content)
    file_name = "temp.html"
    file_path = static_dir / file_name
    print(file_path)
    with open(file_path, "w") as f:
        f.write(html)
    iframe = f"""<iframe src="file={file_path}?t={time.time()}" width="100%" height="500px"></iframe>"""
    # link = f'<a href="file={file_path}" target="_blank">{file_name}</a>'
    return iframe
def submit_handler(file, dropdown1):
    # 在这里处理提交事件
    print("Submit 按钮被点击")
    print("上传的文件:", file.name)
    print("下拉框1选择的值:", dropdown1)
    # return markdown.markdown(run_orc(file, int(dropdown1), int(dropdown2)))
    content = run_orc(file, int(dropdown1), int(dropdown1))
    # html = read_HTML_template()
    # html = html.replace("{{__doc__}}",content)
    return build_iframe(content)


# 创建上传组件
upload_component = gr.inputs.File(label="pdf文件", type="file")

# 创建下拉框组件
dropdown_component1 = gr.inputs.Number(label="要转换的页码")

# 创建 Markdown 预览组件
# markdown_component = gr.Markdown(label="Markdown 预览")
markdown_component = gr.outputs.HTML()
# 创建 Gradio UI
iface = gr.Interface(
    fn=submit_handler,
    inputs=[
        upload_component,
        dropdown_component1,
    ],
    outputs=markdown_component,
    title="Nougat WebUI",
    description="上传文件PDF识别文件内容",
    capture_session=True,  # 启用会话捕获
)
print(torch.cuda.is_available())
load_model()
# 添加 submit 按钮的点击事件处理
iface.launch()
