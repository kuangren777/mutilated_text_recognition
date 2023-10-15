import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from models.cnn import CNN


# 设置超参数
num_epochs = 10
batch_size = 32
learning_rate = 0.001
num_classes = 11

# 加载数据集
data_dir = './data/train/'  # 数据集所在的目录
image_size = (128, 128)  # 图像大小
train_ratio = 0.8  # 训练集所占的比例

# 遍历所有的文件夹，获取文件夹的名称作为图片的标签，同时获取每张图片的路径
images = []
labels = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            image_path = os.path.join(root, file)
            label = os.path.basename(root)
            images.append(image_path)
            labels.append(label)

# 将数据集分为训练集和测试集
num_train = int(len(images) * train_ratio)

train_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = ImageFolder(root=data_dir, transform=train_transform, is_valid_file=lambda x: x in images[:num_train])
test_dataset = ImageFolder(root=data_dir, transform=test_transform, is_valid_file=lambda x: x in images[num_train:])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# 定义模型和优化器
model = CNN(num_classes=num_classes)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 训练模型
train(model=model, optimizer=optimizer, criterion=criterion, train_loader=train_loader, val_loader=val_loader, num_epochs=num_epochs)

# 评估模型
evaluate(model=model, criterion=criterion, test_loader=test_loader)

# 保存模型
if not os.path.exists('models'):
    os.mkdir('models')
torch.save(model.state_dict(), 'models/cnn.pth')

# 用户交互界面
while True:
    print('1. Test on a single image')
    print('2. Exit')
    choice = input('Enter your choice: ')

    if choice == '1':
        image_path = input('Enter the path of the image: ')
        try:
            image = Image.open(image_path).convert('L')
        except:
            print('Invalid image path')
            continue
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        image = transform(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            print('The predicted digit is:', predicted.item())
    elif choice == '2':
        break
    else:
        print('Invalid choice')
