import os
import argparse
import cv2
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from dataLoader import Train_Loader, Val_Loader
from model import ActiveSpeaker
from utils import tools

def main():
    # parser = argparse.ArgumentParser(description = "Training Stage")
    # parser.add_argument('--epoch', type=int, default=15, help="Number of epochs")
    # parser.add_argument('--face_detect_threshold', type=float, default='0.5', help="Confidence score for face detection")
    # parser.add_argument('--face_detect_model', type=str, default='res10_300x300_ssd_iter_140000.caffemodel', help="OpenCV model used for face detection")
    
    # parser.add_argument('--evaluate', type=bool, default=False, help="Perform Evaluation (True/False)")
    # parser.add_argument('--train_dir', type=str, default='train', help="Data path for the training data")
    # parser.add_argument('--test_dir', type=str, default='test', help="Data path for the testing data")
    # args = parser.parse_args()

    trainLoader = Train_Loader(video_id='_mAfwH6i90E', root_dir='train')
    trainLoader = DataLoader(trainLoader, batch_size=64, num_workers=0, shuffle=False)
    train_features, train_labels = next(iter(trainLoader))
    asd = ActiveSpeaker(frame = train_features[60])
    asd.model()

    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")

    # val_loader = Val_Loader()


if __name__ == "__main__":
    main()