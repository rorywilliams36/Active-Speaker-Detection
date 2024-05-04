
import os, cv2, torch
import pandas as pd
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt

from utils import tools

class Vector_Loader(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['Label'])

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        vector = torch.from_numpy(self.data['Flow'][index])
        label = self.data['Label'][index]
        vector = vector.permute(2, 0, 1)
        return vector, label

class Test_Vector_Loader(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        vector = torch.from_numpy(self.data[index])
        vector = vector.permute(2, 0, 1)
        return vector