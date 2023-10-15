import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from utils.data import CustomDataset, GetLabelMap
from utils.train import train
from utils.evaluate import evaluate
from models.cnn import CNN, CNNWithAttention
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
import random
from csv_writer import csv_writer
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR


def set_random_seeds(seed):
    # 设置PyTorch的随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 设置Python内置random模块的随机种子
    random.seed(seed)

    # 设置NumPy的随机种子
    np.random.seed(seed)

    # 设置os模块的随机种子（这不是标准用法，os模块通常不用于生成随机数）
    random_bytes = os.urandom(4)
    random_seed = int.from_bytes(random_bytes, byteorder="big")
    random.seed(random_seed)


id = random.randint(10000, 99999)
seed_value = id  # 你可以选择任何整数作为种子值
set_random_seeds(seed_value)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set hyperparameters
num_epochs = 100
batch_size = 512
# batch_size = 128
learning_rate = 0.005
# 在定义优化器后添加余弦退火学习率调度器

LOG = True
ATTENTION = True

train_dataset = CustomDataset('data/', 'train')
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = CustomDataset('data/', 'val')
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_dataset = CustomDataset('data/', 'test')
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
label_map = GetLabelMap()
num_classes = len(label_map)
print(f'共有{num_classes}个汉字')

if ATTENTION:
    # define model, optimizer, loss function
    model = CNNWithAttention(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0001)
    criterion = nn.CrossEntropyLoss().to(device)

else:
    # define model, optimizer, loss function
    model = CNN(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0001)
    criterion = nn.CrossEntropyLoss().to(device)

# UI
while True:
    print('1. Train model')
    print('2. Evaluate model')
    print('3. Test by single img')
    print('4. Exit')
    choice = input('Input your choice: ')

    if choice == '1':
        load_id = input('Please input the ID of the model to be loaded, enter 0 if you dont have it:')
        if load_id != '0':
            epp = input('Please input the epoch:')
            set_random_seeds(load_id)
            # define model
            if ATTENTION:
                model.load_state_dict(torch.load(f'models/{load_id}/cnn_with_attention{load_id}_{epp}.pth'))
            else:
                model.load_state_dict(torch.load(f'models/{load_id}/cnn{load_id}_{epp}.pth'))
            model.eval().to(device)
        else:
            if ATTENTION:
                # define model
                model = CNNWithAttention(num_classes=num_classes).to(device)
            else:
                # define model
                model = CNN(num_classes=num_classes).to(device)

        note = input('NOTE:')
        csv_writer('train_id_note.csv', [id, ATTENTION, batch_size, learning_rate, note], True)

        # train model
        train(
            id=id,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            log=LOG,
            attention=ATTENTION,
            batch_size=batch_size,
            scheduler=scheduler
        )

        # save model
        if not os.path.exists(f'models/{id}'):
            os.mkdir(f'models/{id}')
        if ATTENTION:
            torch.save(model.state_dict(), f'models/{id}/cnn_with_attention{id}.pth')
        else:
            torch.save(model.state_dict(), f'models/{id}/cnn{id}.pth')

    elif choice == '2':
        if ATTENTION:
            # load model
            model = CNNWithAttention(num_classes=num_classes).to(device)
            id = input('Please input the ID of the model to be loaded:')
            ep = input('Please input the epoch:')
            model.load_state_dict(torch.load(f'models/{id}/cnn_with_attention{id}_{ep}.pth'))
            model.eval().to(device)

            # evaluate model
            evaluate(model=model, criterion=criterion, test_loader=test_loader)
        else:
            # load model
            model = CNN(num_classes=num_classes).to(device)
            id = input('Please input the ID of the model to be loaded:')
            ep = input('Please input the epoch:')
            model.load_state_dict(torch.load(f'models/{id}/cnn{id}_{ep}.pth'))
            model.eval().to(device)

            # evaluate model
            evaluate(model=model, criterion=criterion, test_loader=test_loader)

    elif choice == '3':
        if ATTENTION:
            # load model
            model = CNNWithAttention(num_classes=num_classes).to(device)
            id = input('Please input the ID of the model to be loaded:')
            ep = input('Please input the epoch:')
            model.load_state_dict(torch.load(f'models/{id}/cnn_with_attention{id}_{ep}.pth'))
            model.eval().to(device)
        else:
            # load model
            model = CNN(num_classes=num_classes).to(device)
            id = input('Please input the ID of the model to be loaded:')
            ep = input('Please input the epoch:')
            model.load_state_dict(torch.load(f'models/{id}/cnn{id}_{ep}.pth'))
            model.eval().to(device)

        image_path = input('Please input the path of img: ')
        try:
            image = Image.open(image_path).convert('L')
        except FileNotFoundError:
            print('Illegal image paths.')
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
            print('The prediction is:', label_map[predicted.item()])
    elif choice == '4':
        break
    else:
        print('Illegal input.')
