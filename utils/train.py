# -*- coding: utf-8 -*-
# @Time    : 2023/10/16 12:24
# @Author  : KuangRen777
# @File    : train.py
# @Tags    :
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data import CustomDataset
from models.cnn import CNN
import os
import random
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, SubsetRandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_val_subset_loader(val_loader, num_samples=5000):
    indices = list(range(len(val_loader.dataset)))
    random.shuffle(indices)
    indices = indices[:num_samples]
    sampler = SubsetRandomSampler(indices)
    return DataLoader(val_loader.dataset, batch_size=val_loader.batch_size, sampler=sampler)


def parameters_changed(model, last_parameters):
    for p, last_p in zip(model.parameters(), last_parameters):
        if not torch.equal(p, last_p):
            return True
    return False


def train(
        id: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        log: bool = False,
        attention: bool = False,
        batch_size: int = 32,
        scheduler: CosineAnnealingLR = None) -> None:
    model = model.to(device)

    # Initialize the model's weights using He initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    if log:
        log_path = f'runs/cnn_with_attention{id}' if attention else f'runs/cnn{id}'
        writer = SummaryWriter(log_dir=log_path)

    # last_parameters = [p.clone().detach() for p in model.parameters()]
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):

            images, labels = images.to(device), labels.to(device)

            outputs = model(images).to(device)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if parameters_changed(model, last_parameters):
            #     print("Parameters changed!")
            # else:
            #     print("Parameters did NOT change!")

            # last_parameters = [p.clone().detach() for p in model.parameters()]

            if log and i % 50 == 0:
                # print(loss)
                writer.add_scalar("Step Loss", loss.item(), epoch * total_step + i)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

            if i % (1699 * 256 // batch_size // 5) == 0:
                os.makedirs(f'models/{id}', exist_ok=True)
                torch.save(model.state_dict(), f'models/{id}/cnn{id}_{epoch}_{i + 1}.pth')

        correct, total = 0, 0
        val_subset_loader = get_val_subset_loader(val_loader)
        with torch.no_grad():
            for images, labels in val_subset_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Val Accuracy: {val_accuracy:.2f}%')

        if log:
            writer.add_scalar('Epoch Accuracy', val_accuracy, epoch + 1)

        if scheduler:
            scheduler.step()

    if log:
        writer.close()
