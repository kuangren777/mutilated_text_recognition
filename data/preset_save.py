import os
import cv2
import numpy as np
import urllib.parse

# 设置文件夹路径和数据集保存路径
data_path = "./data_pre_set/enhance/"
save_path = "./test/"


# def cv_imread(file_path = ""):
#     file_path_gbk = file_path.encode('gbk')        # unicode转gbk，字符串变为字节数组
#     img_mat = cv2.imread(file_path_gbk.decode())  # 字节数组直接转字符串，不解码
#     return img_mat


def cv_imread(filePath):
    # 核心就是下面这句，一般直接用这句就行，直接把图片转为mat数据
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    # cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img


# 获取标签列表（即文件夹名称列表）
labels = os.listdir(data_path)

# 遍历标签列表，读取文件夹中的图片并转换为数据集
for label in labels:
    label_path = os.path.join(data_path, label)
    if os.path.isdir(label_path):
        images = os.listdir(label_path)
        for image in images:
            image_path = os.path.join(label_path, image)
            if os.path.isfile(image_path):
                # 读取图片

                # 转换路径编码为UTF-8
                # path_encoded = urllib.parse.quote(image_path.encode('utf-8'))

                # 规范化路径
                # path_norm = os.path.normpath(path_encoded)

                img = cv_imread(image_path)
                # 将图片转换为灰度图像
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 调整图像大小为指定尺寸
                img = cv2.resize(img, (28, 28))
                # 将图像数据保存到numpy数组中
                img_data = np.array(img)
                # 将数据保存到指定路径
                label_save_path = os.path.join(save_path, label)
                if not os.path.exists(label_save_path):
                    os.makedirs(label_save_path)
                np.save(os.path.join(label_save_path, image), img_data)
