# -*- coding: utf-8 -*-
# @Time    : 2023/10/12 14:44
# @Author  : KuangRen777
# @File    : line_graph_data.py
# @Tags    :
import csv
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data import CustomDataset, GetLabelMap
from utils.train import train
from utils.evaluate import evaluate
from models.cnn import CNN
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 指定CSV文件的名称
csv_file_name = "line_graph_data.csv"

# 设置超参数
num_epochs = 100
batch_size = 256
learning_rate = 0.001

train_dataset = CustomDataset('data/', 'train')
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = CustomDataset('data/', 'val')
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_dataset = CustomDataset('data/', 'test')
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
label_map = GetLabelMap()
num_classes = len(label_map)
print(f'共有{num_classes}个汉字')

# 定义模型和优化器
model = CNN(num_classes=num_classes).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss().to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss().to(device)

# 加载模型
model = CNN(num_classes=num_classes).to(device)
id = input('输入加载的模型id:')
for ep in range(100):
    model.load_state_dict(torch.load(f'models/cnn{id}_{ep}.pth'))
    model.eval().to(device)

    # 评估指标
    total = 0
    correct = 0

    # 评估模型
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels).to(device)

        print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
        print('Test Loss: {:.4f}'.format(loss.item()))

        data = [ep, 100 * correct / total, loss.item()]
        with open(csv_file_name, mode="a", newline="", encoding="gbk") as file:
            writer = csv.writer(file)
            writer.writerow(data)
