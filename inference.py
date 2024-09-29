import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 

# 指定使用的模型 ID，推荐使用 "large" 模型以获得更好的性能
model_id = "yifeihu/TF-ID-large"  # 推荐使用大模型以获得更好的性能
# 其他可选模型：
# model_id = "yifeihu/TF-ID-base"
# model_id = "yifeihu/TF-ID-large-no-caption"  # 无标题版本的大模型
# model_id = "yifeihu/TF-ID-base-no-caption"  # 无标题版本的基础模型

# 加载预训练的模型和处理器，设置 trust_remote_code=True 以允许远程代码
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# 设置用于对象检测 (Object Detection, OD) 的文本提示符
prompt = "<OD>"

# TF-ID 模型是在数字 PDF 论文上进行训练的
# 使用本地路径替换远程的图片下载
image_url = "./sample_images/arxiv_2305_10853_5.png"
# 打开图片
image = Image.open(requests.get(image_url, stream=True).raw)

# 使用处理器将文本提示和图片转换为模型可接受的张量格式
inputs = processor(text=prompt, images=image, return_tensors="pt")

# 使用模型生成预测结果
generated_ids = model.generate(
    input_ids=inputs["input_ids"],  # 输入的文本张量
    pixel_values=inputs["pixel_values"],  # 输入的图片张量
    max_new_tokens=1024,  # 最大生成的新标记数量
    do_sample=False,  # 不进行随机采样，使用束搜索
    num_beams=3  # 使用束搜索，束的数量为 3
)

# 解码生成的文本，将生成的标记转化为字符串，保留特殊标记
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

# 处理生成的文本，解析生成结果，指定任务类型为 "<OD>" 并根据图片的尺寸进行调整
parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))

# 打印解析后的答案
print(parsed_answer)

# 提示：如果需要可视化生成的答案，可以参考 Florence 2 仓库中的示例 Colab:
# https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb
