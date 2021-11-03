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
from PIL import Image

#use pandas to read the all classes
table = pd.read_csv("./classes.txt",header = None)
print(table)

#load model and evaluate
model = torch.load("./savedModel.pt")
model.eval()

with open('testing_img_order.txt') as f:
     test_images = [x.strip() for x in f.readlines()]  # all the testing images

# transformation to testing images   
data_transf = transforms.Compose([transforms.CenterCrop([350, 300]),
transforms.Resize([224,224]),
transforms.ToTensor()])

submission = []
for img_dir in test_images:  # image order is important to your result
    img = Image.open("./test/"+img_dir)
    img = data_transf(img)
    img = img.unsqueeze(0) #from 3 dimension (3*224*224) to four dimension (1*3*224*224)
    if torch.cuda.is_available():
        img = img.cuda()
    predicted_class = model(img)  # the predicted category
    n = nn.Softmax(dim=1) # use softmax to easily recognize answer
    predicted_class = n(predicted_class)
    submission.append([img_dir, table[0][predicted_class.max(1)[1].cpu().item()]])

#save answer
np.savetxt('answer.txt', submission, fmt='%s')
