'''
Modified MobileNet to be used with optical flow vector 128x128x2 and binary classification
Original Source Code: https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py
'''

import cv2, dlib, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small
import torch.optim as optim

from models.vectorLoader import Vector_Loader, Test_Vector_Loader

PATH = 'mobilenet_model.pth'
THRESHOLD = 0.3

class MobileNet(nn.Module):
    def __init__(self, num_classes: int = 1, in_channels: int = 2):
        super().__init__()
        self.mobilenet = mobilenet_v3_small(weights=None)

        # Modify input of the first layer of MobileNet to use with 128x128x2 vector
        # Change input channels to 2 (only thing that changes is the input channels)
        self.mobilenet.features[0][0] = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=True)

        # Modify the last MobileNet layer 'classifier'
        # Change last activation to sigmoid for binary classification
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(self.mobilenet.classifier[0].in_features, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mobilenet(x)



def train_mobile(data):
    dataLoader = Vector_Loader(data)
    train_loader = DataLoader(dataLoader, batch_size=64, num_workers=0, shuffle=False)

    model = MobileNet()
    criterion = nn.BCELoss()

    # training using stochastic gradient descent and back propagation
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    epochs = 50
    print("\nTraining")
    for e in range(epochs):
        model.train()
        running_loss = 0
        for vector, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(vector).squeeze()
            loss = criterion(outputs.double(), labels.double())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * vector.size(0)

        epoch_loss = running_loss / dataLoader.__len__()
        print(f'Epoch {e}: {epoch_loss} loss')

    try:
        torch.save(model.state_dict(), 'mobilenet_model.pth')
        print('Model Saved')
    except:
        print('Error occured when saving model')



def test_mobile(data):
    dataLoader = Test_Vector_Loader(data)
    test_loader = DataLoader(dataLoader, batch_size=64, num_workers=0, shuffle=False)

    model = MobileNet()
    model.load_state_dict(torch.load(PATH))
    model.eval()

    preds = np.array([])
    with torch.no_grad():
        for vector in test_loader:
            y_pred = model(vector).squeeze()
            y_pred = y_pred.numpy()
            preds = np.concatenate((preds, y_pred), axis=None)
        print(preds)

    for i in range(len(preds)):
        if preds[i] > THRESHOLD:
            preds[i] = 1
        else:
            preds[i] = 0

    return preds
            
