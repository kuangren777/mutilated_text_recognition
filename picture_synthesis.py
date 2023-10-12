# -*- coding: utf-8 -*-
# @Time    : 2023/10/12 15:35
# @Author  : KuangRen777
# @File    : picture_synthesis.py
# @Tags    :
import os
import cv2
import numpy as np
import random
from PIL import Image
import json

# 从JSON文件中加载字典
with open('my_dict.json', 'r') as file:
    loaded_dict = json.load(file)


def stitch_images_from_folder(src_folder, dst_folder):
    # 获取指定文件夹中的所有png图片
    all_images = [img for img in os.listdir(src_folder) if img.endswith('.png')]

    # 如果图片数量少于100张,则返回
    if len(all_images) < 100:
        print("Not enough images in the folder. Need at least 100.")
        return

    # 从所有图片中随机选择100张
    selected_images = random.sample(all_images, 100)

    # 读取选中的图片
    images = [load_image_with_pillow(os.path.join(src_folder, img)) for img in selected_images]

    # 确保所有图片大小相同，以第一张图片为参考
    target_shape = images[0].shape
    for i in range(1, len(images)):
        if images[i].shape != target_shape:
            images[i] = cv2.resize(images[i], (target_shape[1], target_shape[0]))

    # 将图片拼接成10x10的大图
    rows = [np.hstack(images[i * 10:(i + 1) * 10]) for i in range(10)]
    stitched_image = np.vstack(rows)

    folder_name = get_folder_name(src_folder)
    # 输出大图到指定的文件夹
    output_filename = f'{loaded_dict[folder_name]}.png'
    output_path = os.path.join(dst_folder, output_filename)
    cv2.imwrite(output_path, stitched_image)

    print(f"Stitched image saved at: {output_path}")


def list_subfolders(directory):
    """
    获取指定文件夹中的所有子文件夹。
    :param directory: 要遍历的文件夹路径
    :return: 子文件夹列表
    """
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]


def load_image_with_pillow(path):
    try:
        # 使用Pillow读取图像
        pil_image = Image.open(path)
        # 将Pillow图像转换为NumPy数组
        image_np = np.array(pil_image)
        # 将RGB转换为BGR
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:  # 检查图像是否为彩色
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        return image_np
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def get_folder_name(path):
    folder_name = os.path.basename(os.path.normpath(path))
    return folder_name


# 使用方法
src_folder = "./data/test"
subfolders = list_subfolders(src_folder)
print(subfolders)

for subfolders_name in subfolders:
    # 使用方法
    src_folder = f"./data/test/{subfolders_name}/"
    dst_folder = "./data/big_picture/"
    stitch_images_from_folder(src_folder, dst_folder)
