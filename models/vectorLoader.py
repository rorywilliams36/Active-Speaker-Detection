
import torch
import pandas as pd
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Functions to load data from the dictionary containing the feature vectors into the models for classification
# Follows the general plan for PyTorch's custom dataloaders
class Vector_Loader(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['Flow'])

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        vector = torch.from_numpy(self.data['Flow'][index])
        label = self.data['Label'][index]
        vector = vector.permute(2, 0, 1)
        return vector, label

class Test_Vector_Loader(Dataset):
    def __init__(self, data):
        Vector_Loader.__init__(self, data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        vector = torch.from_numpy(self.data[index])
        vector = vector.permute(2, 0, 1)
        return vector