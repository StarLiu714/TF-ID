import os
import json
import random

# 1. 从 https://huggingface.co/datasets/yifeihu/TF-ID-arxiv-papers 下载数据集
# 2. 将 annotations_with_caption.json 移动到 ./annotations 文件夹
# 3. 解压 arxiv_paper_images.zip 并将图片移动到 ./images 文件夹
# 4. 运行脚本：python coco_to_florence.py

train_percentage = 0.85  # 训练集的比例设置为 85%
coco_json_dir = "./annotations/annotations_with_caption.json"  # COCO 格式数据集路径
# coco_json_dir = "./annotations/annotations_no_caption.json"  # 如果没有 caption 的数据集，可以使用这个路径
output_dir = "./annotations"  # 输出目录，用于保存转换后的数据

# 定义一个函数，用于将 COCO 格式的注释文件转换为 Florence 格式
def convert_to_florence_format(coco_json_dir, output_dir):
    
    # 开始转换 COCO 格式的注释到 Florence 格式
    print("start converting coco annotations to florence format...")

    # 打开并读取 COCO 格式的 JSON 文件
    with open(coco_json_dir, 'r') as file:
        data = json.load(file)

    # 创建一个字典，将类别 ID 映射到类别名称
    category_dict = {category['id']: category['name'] for category in data['categories']}
    print("labels :", category_dict)
    
    # 初始化一个字典，用于存储图片信息
    img_dict = {}
    for img in data['images']:
        img_dict[img['id']] = {
            'width': img['width'],
            'height': img['height'],
            'file_name': img['file_name'],
            'annotations': [],  # 用于存储该图片的注释
            'annotations_str': ""  # 注释的字符串形式
        }

    # 初始化一个字典，存储每个图片的注释（边界框信息）
    annotation_dict = {annotation['image_id']: annotation['bbox'] for annotation in data['annotations']}

    # 定义一个函数，用于格式化注释信息
    def format_annotation(annotation):
        category_id = annotation['category_id']  # 获取类别 ID
        bbox = annotation['bbox']  # 获取 COCO 格式的边界框 [x, y, width, height]
        this_image_width = img_dict[int(annotation['image_id'])]['width']  # 获取图片宽度
        this_image_height = img_dict[int(annotation['image_id'])]['height']  # 获取图片高度
        # 将边界框坐标归一化到 0 到 1，然后乘以 1000 转换为 Florence 格式
        x1 = int(bbox[0] / this_image_width * 1000)
        y1 = int(bbox[1] / this_image_height * 1000)
        x2 = int((bbox[0] + bbox[2]) / this_image_width * 1000)
        y2 = int((bbox[1] + bbox[3]) / this_image_height * 1000)

        # 返回符合 Florence 格式的注释字符串
        return f"{category_dict[category_id]}<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>"

    # 遍历所有注释，并将其转换为 Florence 格式
    for annotation in data['annotations']:
        try:
            annotation_str = format_annotation(annotation)
            if annotation['image_id'] in img_dict:
                img_dict[annotation['image_id']]['annotations'].append(annotation_str)
        except:
            continue

    # 创建一个列表，存储 Florence 格式的数据
    florence_data = []
    for img_id, img_data in img_dict.items():
        annotations_str = "".join(img_data['annotations'])  # 将所有注释连接成一个字符串

        if len(annotations_str) > 0:
            # 如果有注释，将其存储到 Florence 格式数据中
            florence_data.append({
                "image": img_data['file_name'],
                "prefix": "<OD>",  # 对象检测任务的前缀
                "suffix": annotations_str  # 注释的字符串形式
            })
        else:
            # 可选：如果图片没有注释，可以选择忽略或者保留 5% 的图片无注释
            if random.random() < 0.05:
                florence_data.append({
                    "image": img_data['file_name'],
                    "prefix": "<OD>",  # 对象检测任务的前缀
                    "suffix": ""  # 无注释
                })

    print("total number of images:", len(florence_data))

    # 将数据集分割为训练集和测试集，并保存为 jsonl 文件
    train_split = int(len(florence_data) * train_percentage)  # 根据比例计算训练集的大小
    train_data = florence_data[:train_split]
    test_data = florence_data[train_split:]

    print("train size:", len(train_data))
    print("test size:", len(test_data))

    # 训练集和测试集的输出路径
    train_output_dir = os.path.join(output_dir, "train.jsonl")
    test_output_dir = os.path.join(output_dir, "test.jsonl")

    # 如果文件存在，先删除旧文件
    if os.path.exists(train_output_dir):
        os.remove(train_output_dir)

    # 将训练数据保存到 jsonl 文件
    with open(train_output_dir, 'w') as file:
        for entry in train_data:
            json.dump(entry, file)
            file.write("\n")
    
    # 如果文件存在，先删除旧文件
    if os.path.exists(test_output_dir):
        os.remove(test_output_dir)

    # 将测试数据保存到 jsonl 文件
    with open(test_output_dir, 'w') as file:
        for entry in test_data:
            json.dump(entry, file)
            file.write("\n")
    
    print("train and test data saved to ", output_dir)
    print("Now you can run \"accelerate launch train.py\" to train the model.")

# 调用函数，将 COCO 格式注释转换为 Florence 格式
convert_to_florence_format(coco_json_dir, output_dir)
