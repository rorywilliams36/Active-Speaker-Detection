import os
import argparse
import cv2
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from dataLoader import Train_Loader, Val_Loader

def main():
    # parser = argparse.ArgumentParser(description = "Training Stage")
    # parser.add_argument('--epoch', type=int, default=15, help="Number of epochs")
    # parser.add_argument('--evaluate', type=bool, default=False, help="Perform Evaluation (True/False)")
    # parser.add_argument('--train_dir', type=str, default='train', help="Data path for the training data")
    # parser.add_argument('--test_dir', type=str, default='test', help="Data path for the testing data")
    # args = parser.parse_args()




    trainLoader = Train_Loader(video_id='_mAfwH6i90E', root_dir='train')
    trainLoader = DataLoader(trainLoader, batch_size=64, num_workers=0, shuffle=False)
    train_features, train_labels = next(iter(trainLoader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")

    # img = train_features[0]
    # label = train_labels[0]
    # plt.imshow(img, cmap="gray")
    # plt.show()
    # print(f"Label: {label}")

    # val_loader = Val_Loader()


if __name__ == "__main__":
    main()