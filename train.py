import os, argparse, cv2, glob, timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from dataLoader import Train_Loader, Val_Loader
from model import ActiveSpeaker
from evaulation import *
from utils import tools

ids = [ '_mAfwH6i90E', 'B1MAUxpKaV8', '7nHkh4sP5Ks', '2PpxiG0WU18', '-5KQ66BBWC4', '5YPjcdLbs5g', '20TAGRElvfE', '2fwni_Kjf2M']


def main():
    # parser = argparse.ArgumentParser(description = "Training Stage")
    # parser.add_argument('--epoch', type=int, default=15, help="Number of epochs")
    # parser.add_argument('--face_detect_threshold', type=float, default='0.5', help="Confidence score for face detection")
    # parser.add_argument('--face_detect_model', type=str, default='res10_300x300_ssd_iter_140000.caffemodel', help="OpenCV model used for face detection")
    
    # parser.add_argument('--evaluate', type=bool, default=False, help="Perform Evaluation (True/False)")
    # parser.add_argument('--dataset', type=str, default='AVA Active Speaker Dataset', help="Type of dataset to test and train")

    # parser.add_argument('--train_dir', type=str, default='train', help="Data path for the training data")
    # parser.add_argument('--test_dir', type=str, default='test', help="Data path for the testing data")

    # args = parser.parse_args()

    #trainLoader = Train_Loader(video_id='_mAfwH6i90E', root_dir='_mAfwH6i90E')
    #trainLoader = Train_Loader(video_id='B1MAUxpKaV8', root_dir='B1MAUxpKaV8')
    #trainLoader = Train_Loader(video_id='7nHkh4sP5Ks', root_dir='7nHkh4sP5Ks')
    # trainLoader = Train_Loader(video_id='2fwni_Kjf2M', root_dir='2fwni_Kjf2M')
    # trainLoaded = DataLoader(trainLoader, batch_size=64, num_workers=0, shuffle=True)

    counts = [0,0,0,0] # tp, fp, tn, fn
    a_total = 0
    
    for video_id in ids:
        vid_counts = [0,0,0,0]
        angles = []
        prev_labels = []
        trainLoader = Train_Loader(video_id=video_id, root_dir=video_id)
        trainLoaded = DataLoader(trainLoader, batch_size=64, num_workers=0, shuffle=False)

        for images, labels in trainLoaded:
            for i in range(len(images)):
                actual_label = trainLoader.extract_labels(trainLoader.labels, labels, i)
                # print(i)
                # print(labels['label'][i])
                # tools.plot_frame(images[i].numpy())
                asd = ActiveSpeaker(images[i], prev_angles=angles, prev_labels=prev_labels)
                prediction, angles, prev_labels = asd.model()
                # print(angles)
                # print(prev_labels)
                # print('------------')
                tp, fp, tn, fn = evaluate(prediction, actual_label)
                counts[0] += tp
                counts[1] += fp
                counts[2] += tn
                counts[3] += fn

                vid_counts[0] += tp
                vid_counts[1] += fp
                vid_counts[2] += tn
                vid_counts[3] += fn

        a_total += trainLoader.__len__()
        p, r, fm = metrics(vid_counts)
        non_p, non_r, non_fm = metrics([vid_counts[2], vid_counts[3], vid_counts[0], vid_counts[1]])


        print(f'\nSpeakers: {video_id}')
        print('TP,FP,TN,FN: ',vid_counts)
        print('Correct: ', vid_counts[0]+vid_counts[2])
        print('Total: ', np.sum(vid_counts))
        print('Actual Total: ', trainLoader.__len__())
        print('Precision: ', p)
        print('Recall: ', r)
        print('F-Measure: ', fm)

        print(f'\nNon-Speakers')
        print('Precision: ', non_p)
        print('Recall: ', non_r)
        print('F-Measure: ', non_fm)
        print('--------------------')

        a_total += trainLoader.__len__()

    p, r, fm = metrics(counts)

    print('\nGENERAL')
    print('TP,FP,TN,FN: ', counts)
    print('Correct: ', counts[0]+counts[2])
    print('Total: ', np.sum(counts))
    print('Actual Total ', a_total)
    print('\n----------- Speakers -----------')
    print('Precision: ', p)
    print('Recall: ', r)
    print('F-Measure: ', fm)

    non_p, non_r, non_fm = metrics([counts[2], counts[3], counts[0], counts[1]])

    print('\n----------- Non-Speakers -----------')
    print('Precision: ', non_p)
    print('Recall: ', non_r)
    print('F-Measure: ', non_fm)

    print('\nAverage Precision: ', (p + non_p) /2)
    print('Macro F1: ', (non_fm+fm) /2)

    # conf_matrix(counts[0], counts[1], counts[2], counts[3])


if __name__ == "__main__":
    main()