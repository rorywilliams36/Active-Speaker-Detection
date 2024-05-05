'''
Modified MobileNet to be used with optical flow vector 128x128x2 and binary classification
Original Source Code: https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py
From paper: Searching for MobileNetV3, Howard et al
'''

import torch
import numpy as np
from torch import nn
from torchvision.models import mobilenet_v3_small
import torch.optim as optim

PATH = 'mobilenet_model.pth'

class MobileNet(nn.Module):
    def __init__(self, num_classes: int = 1, in_channels: int = 2):
        super().__init__()
        self.mobilenet = mobilenet_v3_small(weights=None)
        self.sigmoid = nn.Sigmoid()

        # Modify input of the first layer of MobileNet to use with 128x128x2 vector
        # Change input channels to 2 (only thing that changes is the input channels)
        self.mobilenet.features[0][0] = nn.Conv2d(in_channels, 16, kernel_size=3, stride=(2,2), padding=(1,1), bias=True)

        # Modify the last MobileNet layer 'classifier'
        # Change last activation to sigmoid for binary classification
        self.mobilenet.classifier[3] = nn.Linear(self.mobilenet.classifier[3].in_features, num_classes, bias=True)

    def forward(self, x):
        return self.sigmoid(self.mobilenet(x))


