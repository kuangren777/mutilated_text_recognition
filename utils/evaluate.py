import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from utils.data import CustomDataset
from models.cnn import CNN
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_test_subset_loader(test_loader, num_samples=1000):
    # 获取测试集的索引列表
    indices = list(range(len(test_loader.dataset)))
    # 随机选择1000个索引
    random.shuffle(indices)
    indices = indices[:num_samples]
    sampler = SubsetRandomSampler(indices)

    return DataLoader(test_loader.dataset, batch_size=test_loader.batch_size, sampler=sampler)


def evaluate(
        model: nn.Module,
        criterion: nn.Module,  # 更改为nn.Module，因为torch.optim并不是损失函数的正确类型
        test_loader: DataLoader):
    total = 0
    correct = 0

    # 使用新的测试子集加载器
    test_subset_loader = get_test_subset_loader(test_loader)

    # evaluate model
    with torch.no_grad():
        for images, labels in test_subset_loader:  # 使用 test_subset_loader 替代 test_loader
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels).to(device)

        print(f'Test Accuracy: {100 * correct / total:.2f}%')
        print(f'Test Loss: {loss.item():.4f}')
