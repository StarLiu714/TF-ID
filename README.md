# TF-ID
本仓库包含了复现所有 TF-ID 模型的完整训练代码。我们也开源了模型权重和人工标注的数据集，全部遵循 MIT 许可证。

## 模型摘要
![TF-ID](https://github.com/ai8hyf/TF-ID/blob/main/assets/cover.png)

TF-ID（表格/图像识别器）是一系列对象检测模型，用于从学术论文中提取表格和图像，由 [Yifei Hu](https://x.com/hu_yifei) 创建。它们有四个版本：
| 模型        | 模型大小  | 模型描述  | 
| ----------- | --------- | --------- |  
| TF-ID-base[[HF]](https://huggingface.co/yifeihu/TF-ID-base) | 0.23B | 提取表格/图像及其标题文本  
| TF-ID-large[[HF]](https://huggingface.co/yifeihu/TF-ID-large) (推荐) | 0.77B | 提取表格/图像及其标题文本  
| TF-ID-base-no-caption[[HF]](https://huggingface.co/yifeihu/TF-ID-base-no-caption) | 0.23B | 提取不带标题文本的表格/图像  
| TF-ID-large-no-caption[[HF]](https://huggingface.co/yifeihu/TF-ID-large-no-caption) (推荐) | 0.77B | 提取不带标题文本的表格/图像  

所有 TF-ID 模型均在 [microsoft/Florence-2](https://huggingface.co/microsoft/Florence-2-large-ft) 的检查点上进行微调。

## 使用示例
- 使用 `python inference.py` 从给定的图片中提取边界框
- 使用 `python pdf_to_table_figures.py` 从 PDF 论文中提取所有表格和图像，并将裁剪的图像保存到 `./sample_output` 文件夹中
- 默认脚本使用 **TF-ID-large**。您可以通过更改脚本中的 model_id 来切换到不同的版本，但始终推荐使用 large 模型。

## 从头训练 TF-ID 模型
1. 克隆仓库：`git clone https://github.com/ai8hyf/TF-ID`
2. 进入目录：`cd TF-ID`
3. 从 Hugging Face 下载 [huggingface.co/datasets/yifeihu/TF-ID-arxiv-papers](https://huggingface.co/datasets/yifeihu/TF-ID-arxiv-papers)
4. 将 **annotations_with_caption.json** 移动到 `./annotations` 文件夹中（如果不需要标题文本，可以使用 **annotations_no_caption.json**）
5. 解压 **arxiv_paper_images.zip** 并将 .png 图像移动到 `./images`
6. 将 COCO 格式数据集转换为 Florence 2 格式：`python coco_to_florence.py`
7. 你应该能看到 `./annotations` 文件夹下的 **train.jsonl** 和 **test.jsonl**
8. 使用 Accelerate 进行模型训练：`accelerate launch train.py`
9. 模型检查点将保存在 `./model_checkpoints` 文件夹中

## 硬件要求
使用 [microsoft/Florence-2-large-ft](https://huggingface.co/microsoft/Florence-2-large-ft)，`BATCH_SIZE=4` 需要至少 40GB 的 VRAM 来进行单 GPU 训练。[microsoft/Florence-2-base-ft](https://huggingface.co/microsoft/Florence-2-base-ft) 模型占用的 VRAM 要少得多。在开始训练之前，请修改 `train.py` 中的 `BATCH_SIZE` 和 `CHECKPOINT` 参数。

## 基准测试
我们在训练数据集之外的论文页面上测试了模型。这些论文是 huggingface 每日论文的一个子集。
正确输出 - 模型为给定页面中的每个表格/图像绘制正确的边界框。

| 模型                                                            | 总图像数 | 正确输出数 | 成功率     |
|-----------------------------------------------------------------|----------|------------|------------|
| TF-ID-base[[HF]](https://huggingface.co/yifeihu/TF-ID-base)     | 258      | 251        | 97.29%     |
| TF-ID-large[[HF]](https://huggingface.co/yifeihu/TF-ID-large)   | 258      | 253        | 98.06%     |
| TF-ID-base-no-caption[[HF]](https://huggingface.co/yifeihu/TF-ID-base-no-caption)   | 261      | 253        | 96.93%     |
| TF-ID-large-no-caption[[HF]](https://huggingface.co/yifeihu/TF-ID-large-no-caption) | 261      | 254        | 97.32%     |

根据具体的使用场景，一些 "不正确" 的输出可能完全可用。例如，模型为包含两个子组件的图像绘制了两个边界框。

## 致谢
- 我通过这篇 [Roboflow 的优秀教程](https://blog.roboflow.com/fine-tune-florence-2-object-detection/) 学会了如何使用 Florence 2 模型。
- 我的朋友 Yi Zhang 帮助标注了一些数据，用于训练我们的概念验证模型，包括基于 YOLO 的 TF-ID 模型。

## 引用
如果你发现 TD-ID 项目有用，请引用此项目：
```
@misc{TF-ID,
  author = {Yifei Hu},
  title = {TF-ID: Table/Figure IDentifier for academic papers},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ai8hyf/TF-ID}},
}
```