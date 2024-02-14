import os, argparse, cv2, glob
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from dataLoader import Train_Loader, Val_Loader
from model import ActiveSpeaker
from evaulation import *
from utils import tools

path = os.getcwd()
frames = glob.glob(f"{path}/dataset/train/*.jpg")

def main():
    # parser = argparse.ArgumentParser(description = "Training Stage")
    # parser.add_argument('--epoch', type=int, default=15, help="Number of epochs")
    # parser.add_argument('--face_detect_threshold', type=float, default='0.5', help="Confidence score for face detection")
    # parser.add_argument('--face_detect_model', type=str, default='res10_300x300_ssd_iter_140000.caffemodel', help="OpenCV model used for face detection")
    
    # parser.add_argument('--evaluate', type=bool, default=False, help="Perform Evaluation (True/False)")
    # parser.add_argument('--dataset', type=str, default='AVA Active Speaker Dataset', help="Type of dataset to test and train")

    # parser.add_argument('--train_dir', type=str, default='train', help="Data path for the training data")
    # parser.add_argument('--test_dir', type=str, default='test', help="Data path for the testing data")
    # parser.add_argument('--CUDA', type=bool, default=False, help="Whether to run code on GPU is available")

    # args = parser.parse_args()

    trainLoader = Train_Loader(video_id='_mAfwH6i90E', root_dir='train')
    # trainLoader = Train_Loader(video_id='B1MAUxpKaV8', root_dir='B1MAUxpKaV8')
    trainLoader = DataLoader(trainLoader, batch_size=64, num_workers=0, shuffle=True)

    num = 0
    for images, labels in trainLoader:
        for i in range(len(images)):
            asd = ActiveSpeaker(images[i])
            faces = asd.model()
            # print(i)
            # if num % 100 == 0:
            #     tools.plot_faces_detected(images[i].numpy(), faces)

            num += 1

if __name__ == "__main__":
    main()