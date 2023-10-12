import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data import CustomDataset
from models.cnn import CNN
import os
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 设置超参数
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# # 加载数据集
# train_dataset = CustomDataset('data/', 'train')
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
# val_dataset = CustomDataset('data/', 'val')
# val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

id = random.randint(10000, 99999)

def train(model, optimizer, criterion, train_loader, val_loader, num_epochs, num_classes):
    idd = input('请选择要加载的模型的id，没有就输入0')
    if idd != '0':
        epp = input('请输入要加载的epoch值')
        # 定义模型和优化器
        model.load_state_dict(torch.load(f'models/cnn{idd}_{epp}.pth'))
        model.eval().to(device)
    else:
        # 定义模型和优化器
        model = CNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)

    # 训练模型
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # print(f"Batch {i + 1}: Images size: {images.size()}, Labels size: {labels.size()}")
            # print(i)

            images = images.to(device)
            outputs = model(images).to(device)
            # print(f"Batch {i + 1}: output size: {outputs.size()}, Labels size: {labels.size()}")
            labels = labels.to(device)
            loss = criterion(outputs, labels).to(device, non_blocking=True)
            # print(f"Batch {i + 1}: Images size: {images.size()}, Labels size: {labels.size()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
            #       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            if (i+1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

            if (i+1) % 200 == 0:
                if not os.path.exists('models'):
                    os.mkdir('models')
                torch.save(model.state_dict(), f'models/cnn{id}_{epoch}_{i + 1}.pth')

        # 验证模型
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                outputs = model(images).to(device)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0).to(device)
                correct += (predicted == labels).sum().item().to(device)

            print('Epoch [{}/{}], Val Accuracy: {:.2f}%'
                  .format(epoch+1, num_epochs, 100 * correct / total))

    # 保存模型
    # torch.save(model.state_dict(), f'models/cnn{id}.pth')
