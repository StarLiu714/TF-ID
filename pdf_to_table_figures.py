from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

import os
import json
import time

# 将 PDF 文件转换为图片
def pdf_to_image(pdf_path):
    # 使用 pdf2image 库将 PDF 文件的每一页转换为图片
    images = convert_from_path(pdf_path)
    return images

# 执行 TF-ID 模型的对象检测
def tf_id_detection(image, model, processor):
    prompt = "<OD>"  # 设置对象检测任务的提示符
    # 将图片和提示符转换为模型输入的张量格式
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    # 使用模型生成检测结果
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3
    )
    # 解码生成的文本
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    # 处理生成的文本并返回检测结果
    annotation = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))
    return annotation["<OD>"]

# 根据检测的边界框从图片中裁剪出图像并保存
def save_image_from_bbox(image, annotation, page, output_dir):
    # 文件名格式为页面编号 + 标签 + 索引
    for i in range(len(annotation['bboxes'])):
        bbox = annotation['bboxes'][i]  # 获取边界框坐标
        label = annotation['labels'][i]  # 获取检测到的标签
        x1, y1, x2, y2 = bbox  # 解包边界框坐标
        # 根据边界框裁剪图片
        cropped_image = image.crop((x1, y1, x2, y2))
        # 保存裁剪后的图片
        cropped_image.save(os.path.join(output_dir, f"page_{page}_{label}_{i}.png"))

# 将 PDF 文件中的图表或表格裁剪并保存为图片
def pdf_to_table_figures(pdf_path, model_id, output_dir):
    # 生成时间戳，用作输出文件夹名称
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(output_dir, timestr)

    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 将 PDF 转换为图片
    images = pdf_to_image(pdf_path)
    print(f"PDF loaded. Number of pages: {len(images)}")  # 打印 PDF 页数

    # 加载指定的预训练模型和处理器
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("Model loaded: ", model_id)
    
    print("=====================================")
    print("start saving cropped images")
    
    # 对每一页图片执行对象检测，并保存裁剪的图片
    for i, image in enumerate(images):
        annotation = tf_id_detection(image, model, processor)  # 进行对象检测
        save_image_from_bbox(image, annotation, i, output_dir)  # 保存裁剪的图片
        print(f"Page {i} saved. Number of objects: {len(annotation['bboxes'])}")  # 打印每页保存的对象数量
    
    print("=====================================")
    print("All images saved to: ", output_dir)

# 主函数调用：指定模型、PDF 路径和输出文件夹路径
model_id = "yifeihu/TF-ID-large"
pdf_path = "./pdfs/arxiv_2305_04160.pdf"
output_dir = "./sample_output"

# 执行 PDF 转换并保存结果
pdf_to_table_figures(pdf_path, model_id, output_dir)
