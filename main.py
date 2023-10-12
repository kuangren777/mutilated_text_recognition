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

# 设置超参数
num_epochs = 100
batch_size = 256
learning_rate = 0.001

# 加载数据集
# data_dir = './data/train/'  # 数据集所在的目录
# image_size = (128, 128)  # 图像大小
# train_ratio = 0.8  # 训练集所占的比例

# # 遍历所有的文件夹，获取文件夹的名称作为图片的标签，同时获取每张图片的路径
# images = []
# labels = []
# for root, dirs, files in os.walk(data_dir):
#     for file in files:
#         if file.endswith('.jpg') or file.endswith('.png'):
#             image_path = os.path.join(root, file)
#             label = os.path.basename(root)
#             images.append(image_path)
#             labels.append(label)

# # 使用PIL库读取每张图片，并将其转换为指定的大小
# processed_images = []
# for image_path in images:
#     image = Image.open(image_path).convert('RGB')
#     image = image.resize(image_size)
#     processed_images.append(image)
#
# # 将转换后的图片和对应的标签存储到一个列表中
# dataset = list(zip(processed_images, labels))

# 将存储图片和标签的列表打乱，以便后续训练时可以更好地训练模型

# random.shuffle(dataset)

# # 将数据集分为训练集和测试集，并使用ImageFolder类将数据集转换为PyTorch中的数据集格式
# num_train = int(len(dataset) * train_ratio)
# train_dataset = ImageFolder(dataset[:num_train])
# test_dataset = ImageFolder(dataset[num_train:])

# train_dataset = ImageFolder('./data/train/')
# test_dataset = ImageFolder('./data/test/')

# train_dataset = CustomDataset('data/', 'train')
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# val_dataset = CustomDataset('data/', 'val')
# val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
# test_dataset = CustomDataset('data/', 'test')
# test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
# num_classes = len(GetLabelMap())
# print(num_classes)

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
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss().to(device)

# # 加载模型
# model = CNN(num_classes=num_classes)
# model.load_state_dict(torch.load('models/cnn.pth'))
# model.eval()

# 定义损失函数
criterion = nn.CrossEntropyLoss().to(device)

# # 评估模型
# evaluate(model=model, criterion=criterion, test_loader=test_loader)

# 用户交互界面
while True:
    print('1. 训练模型')
    print('2. 评估模型')
    print('3. 使用单张图片测试')
    print('4. 退出')
    choice = input('输入你的选项: ')

    if choice == '1':
        # 训练模型
        train(model=model, optimizer=optimizer, criterion=criterion, train_loader=train_loader, val_loader=val_loader,
              num_epochs=num_epochs, num_classes=num_classes)

        # 保存模型
        if not os.path.exists('models'):
            os.mkdir('models')
        torch.save(model.state_dict(), 'models/cnn.pth')

    elif choice == '2':
        # 加载模型
        model = CNN(num_classes=num_classes).to(device)
        id = input('输入加载的模型id:')
        ep = input('输入加载的模型epoch:')
        # model.load_state_dict(torch.load('models/cnn.pth'))
        model.load_state_dict(torch.load(f'models/cnn{id}_{ep}.pth'))
        model.eval().to(device)

        # 评估模型
        evaluate(model=model, criterion=criterion, test_loader=test_loader)

    elif choice == '3':
        # 加载模型
        model = CNN(num_classes=num_classes).to(device)
        id = input('输入加载的模型id:')
        ep = input('输入加载的模型epoch:')
        # model.load_state_dict(torch.load('models/cnn.pth'))
        model.load_state_dict(torch.load(f'models/cnn{id}_{ep}.pth')).to(device)
        model.eval().to(device)

        image_path = input('输入图片路径: ')
        try:
            image = Image.open(image_path).convert('L')
        except:
            print('不合法图片路径')
            continue
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        image = transform(image).to(device)
        image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image).to(device)
            _, predicted = torch.max(output.data, 1).to(device)
            print('预测值是', label_map[predicted.item()])
    elif choice == '4':
        break
    else:
        print('不合法输入')
