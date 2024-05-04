'''
Modified ShuffleNet to be used with optical flow vector 128x128x2 and binary classification
Original Source Code: https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py
Other Documentation: https://pytorch.org/hub/pytorch_vision_shufflenet_v2/
'''

import torch
import numpy as np
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import shufflenet_v2_x0_5
import torch.optim as optim

from models.vectorLoader import Vector_Loader, Test_Vector_Loader

PATH = 'shufflenet_model.pth'

class ShuffleNet(nn.Module):
    def __init__(self, num_classes: int = 1, in_channels: int = 2):
        super().__init__()
        self.shufflenet = shufflenet_v2_x0_5(weights=None)

        self.shufflenet.conv1[0] = nn.Conv2d(in_channels, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)
        self.shufflenet.classifier = nn.Sequential(
            nn.Linear(self.shufflenet.fc.in_features, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.shufflenet(x)


def train_shuffle(data):
    dataLoader = Vector_Loader(data)
    train_loader = DataLoader(dataLoader, batch_size=64, num_workers=0, shuffle=False)

    model = ShuffleNet()
    criterion = nn.BCELoss()

    # training using gradient descent and back propagation
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    epochs = 50
    print("\nTraining")
    for e in range(epochs):
        model.train()
        running_loss = 0
        for vector, labels in train_loader:
            optimizer.zero_grad()
            vector = preprocess(vector)
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



def test_shuffle(data):
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
        if preds[i] > 0.075:
            preds[i] = 1
        else:
            preds[i] = 0

    return preds
            
