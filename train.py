import os, argparse, cv2, glob, timeit
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
ids = [ '_mAfwH6i90E', 'B1MAUxpKaV8', 'AYebXQ8eUkM', '7nHkh4sP5Ks']


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

    #trainLoader = Train_Loader(video_id='_mAfwH6i90E', root_dir='train')
    #trainLoader = Train_Loader(video_id='B1MAUxpKaV8', root_dir='B1MAUxpKaV8')
    #trainLoader = Train_Loader(video_id='AYebXQ8eUkM', root_dir='AYebXQ8eUkM')
    # trainLoader = Train_Loader(video_id='7nHkh4sP5Ks', root_dir='7nHkh4sP5Ks')
    # trainLoaded = DataLoader(trainLoader, batch_size=64, num_workers=0, shuffle=True)

    counts = [0,0,0,0] # tp, fp, tn, fn
    a_total = 0
    for video_id in ids:
        trainLoader = Train_Loader(video_id=video_id, root_dir=video_id)
        trainLoaded = DataLoader(trainLoader, batch_size=64, num_workers=0, shuffle=True)

        for images, labels in trainLoaded:
            for i in range(len(images)):
                actual_label = trainLoader.extract_labels(trainLoader.labels, labels, i)
                #print(labels['label'][i])
                asd = ActiveSpeaker(images[i])
                prediction = asd.model()
                #print('------------')
                tp, fp, tn, fn = evaluate(prediction, actual_label)
                counts[0] += tp
                counts[1] += fp
                counts[2] += tn
                counts[3] += fn


        a_total += trainLoader.len_labels()
    
    p, r, fm = metrics(counts)

    print('TP, FP, TN, FN: ', counts[:])
    print('Actual Total: ', a_total)
    print('Total: ', np.sum(counts))
    print('Correct: ', counts[0] + counts[2])
    print('Incorrect: ', counts[1] + counts[3])
    print('----------- Speakers -----------')
    print('Precision: ', p)
    print('Recall: ', r)
    print('F-Measure: ', fm)

    non_p, non_r, non_fm = metrics([counts[2:-1], counts[0:1]])
    print('----------- Non-Speakers -----------')
    print('Precision: ', non_p)
    print('Recall: ', non_r)
    print('F-Measure: ', non_fm)

    print('Average Precision: ', (p + non_p) /2)


            
    # print(faces)
    # img = images[i].numpy()
    # for face in faces:
    #     # tools.plot_frame(face)
    #     x1, y1, x2, y2 = face[3:7] * 300

    #     # Grabs extra pixels around box to account for errors and also check ranges
    #     x1 = max(round(float(x1))-5, 0)
    #     y1 = max(round(float(y1))-5, 0)
    #     x2 = min(round(float(x2))+5, 300)
    #     y2 = min(round(float(y2))+5, 300)

    #     # Extracts and resizes the face detected from the frame
    #     face_region = cv2.resize(img[y1:y2, x1:x2], (64,64))
    #     save_facetracks(face_region, actual_label, trainLoader, face, i)

    
    # tools.plot_frame(images[i].numpy())
    # if num % 100 == 0:
    # tools.plot_faces_detected(images[i].numpy(), faces)
    # tools.plot_actual(images[i].numpy(), actual_label[1])





if __name__ == "__main__":
    main()