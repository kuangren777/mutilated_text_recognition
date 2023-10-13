import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data import CustomDataset
from models.cnn import CNN
import os
import random
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

id = random.randint(10000, 99999)


def train(
        model: nn.Module,
        optimizer: torch.optim,
        criterion: nn.CrossEntropyLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        log: bool,
        attention: bool) -> int:
    if log:
        if attention:
            writer = SummaryWriter(log_dir=f'runs/cnn_with_attention{id}')
        else:
            writer = SummaryWriter(log_dir=f'runs/cnn{id}')

    # train model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # print(f"Batch {i + 1}: Images size: {images.size()}, Labels size: {labels.size()}")
            # print(i)

            images = images.to(device)
            outputs = model(images).to(device)
            labels = labels.to(device)
            loss = criterion(outputs, labels).to(device, non_blocking=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 使用 writer 添加想要在 TensorBoard 中查看的数据
            if log:
                writer.add_scalar("Step Loss", loss.item(), epoch * 1700 + i + 1)

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

            if (i + 1) % 200 == 0:
                if not os.path.exists('models'):
                    os.mkdir('models')
                torch.save(model.state_dict(), f'models/cnn{id}_{epoch}_{i + 1}.pth')

        # verification model each episode
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Val Accuracy: {100 * correct / total:.2f}%')

            if log:
                writer.add_scalar('Epoch Accuracy', 100 * correct / total, epoch + 1)

    if log:
        writer.close()

    return id
