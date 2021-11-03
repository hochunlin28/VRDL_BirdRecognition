'''
hyperparameter setting:
    model : efficientnet
    learning rate: 0.0002
    optimizer: Adam
    loss function: cross entropy loss
    batch size: 22
    epoch: 50
'''
import numpy as np
import pandas as pd
import os
import matplotlib.image as mpimg

import torch
import torch.nn as nn
import torch.optim as optim 

import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as utils
from torchvision import transforms

import matplotlib.pyplot as plt
import pandas as pd
from efficientnet_pytorch import EfficientNet
from PIL import Image

data_dir = './'
train_dir = data_dir + 'train/'
test_dir = data_dir + 'test/'

#use pandas to read the training labels
labels = pd.read_csv('training_labels.txt', sep=" ",header=None)
labels.columns = ["id", "bird_class"]

#class ImgaeData to get labels and transformed images
class ImageData(Dataset):
    def __init__(self, df, data_dir, transform):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):       
        img_name = self.df.id[index]
        number = self.df.bird_class[index]
        number = int(number[:number.index(".")]) - 1
        label = number
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path)
        image = self.transform(image)

        return image, label

#use dataloader to generate training dataset
'''
data_augmentation = transforms.RandomChoice([
transforms.RandomHorizontalFlip(p=0.5),
transforms.RandomRotation([-15,15], expand=True),
transforms.CenterCrop([350, 310])])

data_transf = transforms.Compose([
data_augmentation,
transforms.Resize([224,224]),
transforms.ToTensor()])
'''

data_transf = transforms.Compose([
transforms.RandomHorizontalFlip(p=0.5),
transforms.RandomRotation([-15,15], expand=True), 
transforms.CenterCrop([350, 310]),
transforms.Resize([224,224]),
transforms.ToTensor()])


train_data = ImageData(df = labels, data_dir = train_dir, transform = data_transf)
train_loader = DataLoader(dataset = train_data, batch_size = 22, shuffle = True)

#load model from pretrained
model = EfficientNet.from_pretrained('efficientnet-b1')
print("load_success")

#unfreeze model weights
for param in model.parameters():
    param.requires_grad = True

#change the last fully connected layer to fit the 200 labels
feature = model._fc.in_features
model._fc = nn.Linear(in_features=feature,out_features=200,bias=True)

#use GPU to train
model = model.to('cuda')

#set optimize function and loss function
optimizer = optim.Adam(model.parameters(),lr = 0.0001)
loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)

# Train model
loss_log = []


for epoch in range(50):    
    model.train()
    train_correct = 0    
    for ii, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        train_correct += (output.max(1)[1] == target).sum()
        optimizer.step()  
        if ii % 50 == 0:
            print("batch: %s" %ii)
        if ii % 1000 == 0:
            loss_log.append(loss.item())
       
    print('Epoch: {} - Loss: {:.6f} - Correct: {}'.format(epoch + 1, loss.item(), train_correct))

torch.save(model,"save.pt")

