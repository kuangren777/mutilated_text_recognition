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


# device = 'cpu'

def get_val_subset_loader(val_loader, num_samples=1000):
    # 获取验证集的索引列表
    indices = list(range(len(val_loader.dataset)))
    # 随机选择1000个索引
    random.shuffle(indices)
    indices = indices[:num_samples]
    sampler = SubsetRandomSampler(indices)

    return DataLoader(val_loader.dataset, batch_size=val_loader.batch_size, sampler=sampler)


def has_parameters_changed(model, last_parameters):
    for p, last_p in zip(model.parameters(), last_parameters):
        if not torch.equal(p, last_p):
            return True
    return False


# def train(
#         id: int,
#         model: nn.Module,
#         optimizer: torch.optim,
#         criterion: nn.CrossEntropyLoss,
#         train_loader: DataLoader,
#         val_loader: DataLoader,
#         num_epochs: int,
#         log: bool,
#         attention: bool,
#         batch_size: int,
#         scheduler: CosineAnnealingLR) -> None:
#     model = model.to(device)
#
#     # 在训练循环开始之前
#     last_parameters = [p.clone().detach() for p in model.parameters()]
#
#     if log:
#         if attention:
#             writer = SummaryWriter(log_dir=f'runs/cnn_with_attention{id}')
#         else:
#             writer = SummaryWriter(log_dir=f'runs/cnn{id}')
#
#     # train model
#     total_step = len(train_loader)
#     for epoch in range(num_epochs):
#         for i, (images, labels) in enumerate(train_loader):
#             # print(f"Batch {i + 1}: Images size: {images.size()}, Labels size: {labels.size()}")
#             # print(i)
#
#             images = images.to(device)
#             outputs = model(images)
#             labels = labels.to(device)
#             loss = criterion(outputs, labels).to(device)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             # 在训练循环内部，每次迭代后
#             if has_parameters_changed(model, last_parameters):
#                 print("Parameters changed!")
#             else:
#                 print("Parameters did NOT change!")
#
#             # 更新 last_parameters 为当前参数，以便于下次迭代进行比较
#             last_parameters = [p.clone().detach() for p in model.parameters()]
#
#             # 使用 writer 添加想要在 TensorBoard 中查看的数据
#             if log:
#                 writer.add_scalar("Step Loss", loss.item(), epoch * 1699 * 256 // batch_size + i + 1)
#
#             if (i + 1) % 50 == 0:
#                 print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')
#
#             if (i + 1) % (1699 * 256 // batch_size // 5) == 0:
#                 if not os.path.exists(f'models/{id}'):
#                     os.mkdir(f'models/{id}')
#                 torch.save(model.state_dict(), f'models/{id}/cnn{id}_{epoch}_{i + 1}.pth')
#
#         # 使用新的验证子集加载器
#         val_subset_loader = get_val_subset_loader(val_loader)
#         # verification model each episode
#         with torch.no_grad():
#             correct = 0
#             total = 0
#             for images, labels in val_subset_loader:  # 使用 val_subset_loader 替代 val_loader
#                 images = images.to(device)
#                 labels = labels.to(device)
#                 outputs = model(images)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#
#             print(f'Epoch [{epoch + 1}/{num_epochs}], Val Accuracy: {100 * correct / total:.2f}%')
#
#             if log:
#                 writer.add_scalar('Epoch Accuracy', 100 * correct / total, epoch + 1)
#
#         # 在epoch结束时更新学习率
#         scheduler.step()
#
#     if log:
#         writer.close()


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

    if log:
        log_path = f'runs/cnn_with_attention{id}' if attention else f'runs/cnn{id}'
        writer = SummaryWriter(log_dir=log_path)

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Check if parameters have changed
            changed = any([torch.any(p.grad != 0) for p in model.parameters()])
            if changed:
                print("Parameters changed!")
            else:
                print("Parameters did NOT change!")

            # Logging
            if log and i % 50 == 0:
                writer.add_scalar("Step Loss", loss.item(), epoch * total_step + i)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

            # Save model
            if i % (1699 * 256 // batch_size // 5) == 0:
                os.makedirs(f'models/{id}', exist_ok=True)
                torch.save(model.state_dict(), f'models/{id}/cnn{id}_{epoch}_{i + 1}.pth')

        # Validation
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

        # Scheduler step
        if scheduler:
            scheduler.step()

    if log:
        writer.close()
