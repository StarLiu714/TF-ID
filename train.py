# 该脚本使用了 Roboflow 的教程代码片段：https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-florence-2-on-detection-dataset.ipynb

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoProcessor,
    get_scheduler
)
from tqdm import tqdm
import os
import json
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple, Generator
from accelerate import Accelerator

# 在运行本脚本前，先运行 coco_to_florence.py 将 COCO 注释文件转换为 Florence 格式
# 使用 "accelerate launch train.py" 运行本脚本

BATCH_SIZE = 4  # 根据 GPU 规格调整批次大小
gradient_accumulation_steps = 2  # 根据 GPU 规格调整梯度累积步数
NUM_WORKERS = 0  # DataLoader 中的并发数
epochs = 8  # 训练的轮次
learning_rate = 5e-6  # 学习率

# 设定图像文件夹、训练集和测试集的标签路径，以及模型检查点的输出路径
img_dir = "./images"
train_labels = "./annotations/train.jsonl"  # 由 coco_to_florence.py 生成的训练集标签
test_labels = "./annotations/test.jsonl"  # 由 coco_to_florence.py 生成的测试集标签
output_dir = "./model_checkpoints"  # 模型检查点保存路径

CHECKPOINT = "microsoft/Florence-2-base-ft"  # 模型检查点
# CHECKPOINT = "microsoft/Florence-2-large-ft"

# 使用 Accelerate 库进行分布式训练加速
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
DEVICE = accelerator.device

# 加载预训练的 Florence 模型和处理器
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True).to(DEVICE)
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)

# 自定义 JSONL 数据集类
class JSONLDataset:
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()

    # 加载 JSONL 文件中的所有条目
    def _load_entries(self) -> List[Dict[str, Any]]:
        entries = []
        with open(self.jsonl_file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                entries.append(data)
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    # 获取指定索引的图片及其注释数据
    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")
        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry['image'])
        try:
            image = Image.open(image_path)
            return (image, entry)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file {image_path} not found.")

# 自定义检测数据集类
class DetectionDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.dataset = JSONLDataset(jsonl_file_path, image_directory_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, data = self.dataset[idx]
        prefix = data['prefix']  # 提示符
        suffix = data['suffix']  # 注释结果
        return prefix, suffix, image

# 定义数据加载器的 collate_fn 函数，用于将批次的数据转换为模型输入格式
def collate_fn(batch):
	questions, answers, images = zip(*batch)
	inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(DEVICE)
	return inputs, answers

# 加载训练集和测试集
train_dataset = DetectionDataset(
    jsonl_file_path=train_labels,
    image_directory_path=img_dir
)
test_dataset = DetectionDataset(
    jsonl_file_path=test_labels,
    image_directory_path=img_dir
)

# 使用 DataLoader 加载数据集
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS, shuffle=False)

# 定义模型训练函数
def train_model(train_loader, val_loader, model, processor, epochs, lr):
	optimizer = AdamW(model.parameters(), lr=lr)  # 定义优化器
	num_training_steps = epochs * len(train_loader)  # 总训练步数
	lr_scheduler = get_scheduler(
		name="linear",
		optimizer=optimizer,
		num_warmup_steps=12,
		num_training_steps=num_training_steps,
	)

	# 使用 Accelerator 加速模型和数据加载器
	model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
		model, optimizer, train_loader, lr_scheduler
	)

	for epoch in range(epochs):
		model.train()
		train_loss = 0
		# 遍历每个批次的训练数据
		for inputs, answers in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
			with accelerator.accumulate(model):
				input_ids = inputs["input_ids"]
				pixel_values = inputs["pixel_values"]
				# 将答案转换为模型输入的标签格式
				labels = processor.tokenizer(
					text=answers,
					return_tensors="pt",
					padding=True,
					return_token_type_ids=False
				).input_ids.to(DEVICE)

				# 前向传播并计算损失
				outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
				loss = outputs.loss

				# 反向传播更新参数
				accelerator.backward(loss)
				optimizer.step()
				lr_scheduler.step()
				optimizer.zero_grad()
				train_loss += loss.item()

		avg_train_loss = train_loss / len(train_loader)  # 计算平均训练损失
		print(f"Average Training Loss: {avg_train_loss}")

		# 评估模型在验证集上的表现
		model.eval()
		val_loss = 0
		with torch.no_grad():
			for inputs, answers in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
				input_ids = inputs["input_ids"]
				pixel_values = inputs["pixel_values"]
				labels = processor.tokenizer(
					text=answers,
					return_tensors="pt",
					padding=True,
					return_token_type_ids=False
				).input_ids.to(DEVICE)

				# 计算验证集损失
				outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
				loss = outputs.loss
				val_loss += loss.item()

			avg_val_loss = val_loss / len(val_loader)  # 计算平均验证损失
			print(f"Average Validation Loss: {avg_val_loss}")
			# 注意：验证损失不是非常具有信息量，应该使用更好的评估指标

		# 保存每轮训练后的模型权重
		weights_output_dir = output_dir + f"/epoch_{epoch+1}"
		os.makedirs(weights_output_dir, exist_ok=True)
		accelerator.save_model(model, weights_output_dir)

# 调用模型训练函数
train_model(train_loader, test_loader, model, processor, epochs=epochs, lr=learning_rate)
