'''
Modified ShuffleNet to be used with optical flow vector 128x128x2 and binary classification
Original Source Code: https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py
Other Documentation: https://pytorch.org/hub/pytorch_vision_shufflenet_v2/
'''

import torch
import numpy as np
from torch import nn
from torchvision import transforms
from torchvision.models import shufflenet_v2_x0_5
import torch.optim as optim


class ShuffleNet(nn.Module):
    def __init__(self, num_classes: int = 1, in_channels: int = 2):
        super().__init__()
        self.shufflenet = shufflenet_v2_x0_5(weights=None)

        # Modify input layer to account for 2 channels
        self.shufflenet.conv1[0] = nn.Conv2d(in_channels, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)

        # Modify output layer for binary classification and change output activation function
        self.shufflenet.fc = nn.Sequential(
            nn.Linear(self.shufflenet.fc.in_features, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.shufflenet(x)
