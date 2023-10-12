import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data import CustomDataset
from models.cnn import CNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 加载数据集
# test_dataset = CustomDataset('data/', 'test')
# test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


def evaluate(model, criterion, test_loader):

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
