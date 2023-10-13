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
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set hyperparameters
num_epochs = 100
batch_size = 256
learning_rate = 0.01
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)
else:
    # define model, optimizer, loss function
    model = CNN(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)

# UI
while True:
    print('1. Train model')
    print('2. Evaluate model')
    print('3. Test by single img')
    print('4. Exit')
    choice = input('Input your choice: ')

    if choice == '1':
        idd = input('Please input the ID of the model to be loaded, enter 0 if you dont have it:')
        if idd != '0':
            epp = input('Please input the epoch:')
            # define model
            if ATTENTION:
                model.load_state_dict(torch.load(f'models/cnn_with_attention{idd}_{epp}.pth'))
            else:
                model.load_state_dict(torch.load(f'models/cnn{idd}_{epp}.pth'))
            model.eval().to(device)
        else:
            if ATTENTION:
                # define model
                model = CNNWithAttention(num_classes=num_classes).to(device)
            else:
                # define model
                model = CNN(num_classes=num_classes).to(device)

        # train model
        id = train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            log=LOG,
            attention=ATTENTION
        )

        # save model
        if not os.path.exists('models'):
            os.mkdir('models')
        if ATTENTION:
            torch.save(model.state_dict(), f'models/cnn_with_attention{id}.pth')
        else:
            torch.save(model.state_dict(), f'models/cnn{id}.pth')

    elif choice == '2':
        if ATTENTION:
            # load model
            model = CNNWithAttention(num_classes=num_classes).to(device)
            id = input('Please input the ID of the model to be loaded:')
            ep = input('Please input the epoch:')
            model.load_state_dict(torch.load(f'models/cnn_with_attention{id}_{ep}.pth'))
            model.eval().to(device)

            # evaluate model
            evaluate(model=model, criterion=criterion, test_loader=test_loader)
        else:
            # load model
            model = CNN(num_classes=num_classes).to(device)
            id = input('Please input the ID of the model to be loaded:')
            ep = input('Please input the epoch:')
            model.load_state_dict(torch.load(f'models/cnn{id}_{ep}.pth'))
            model.eval().to(device)

            # evaluate model
            evaluate(model=model, criterion=criterion, test_loader=test_loader)

    elif choice == '3':
        if ATTENTION:
            # load model
            model = CNNWithAttention(num_classes=num_classes).to(device)
            id = input('Please input the ID of the model to be loaded:')
            ep = input('Please input the epoch:')
            model.load_state_dict(torch.load(f'models/cnn_with_attention{id}_{ep}.pth'))
            model.eval().to(device)
        else:
            # load model
            model = CNN(num_classes=num_classes).to(device)
            id = input('Please input the ID of the model to be loaded:')
            ep = input('Please input the epoch:')
            model.load_state_dict(torch.load(f'models/cnn{id}_{ep}.pth'))
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
