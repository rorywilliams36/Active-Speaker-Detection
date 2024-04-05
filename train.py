import os, argparse, cv2, glob, timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from dataLoader import Train_Loader, Val_Loader
from model import ActiveSpeaker
from evaluation import *
from utils import tools

# ids = ['_mAfwH6i90E', 'B1MAUxpKaV8', '7nHkh4sP5Ks', '2PpxiG0WU18', '-5KQ66BBWC4', '5YPjcdLbs5g', '20TAGRElvfE', '2fwni_Kjf2M']
ids = ['_mAfwH6i90E']

def main():
    # parser = argparse.ArgumentParser(description = "Training Stage")
    # parser.add_argument('--face_detect_threshold', type=float, default='0.5', help="Confidence score for face detection")
    # parser.add_argument('--face_detect_model', type=str, default='res10_300x300_ssd_iter_140000.caffemodel', help="OpenCV model used for face detection")
    
    # parser.add_argument('--evaluate', type=bool, default=False, help="Perform Evaluation (True/False)")
    # parser.add_argument('--dataset', type=str, default='AVA Active Speaker Dataset', help="Type of dataset to test and train")

    # parser.add_argument('--train_dir', type=str, default='train', help="Data path for the training data")
    # parser.add_argument('--test_dir', type=str, default='test', help="Data path for the testing data")

    # args = parser.parse_args()

    counts = [0,0,0,0] # tp, fp, tn, fn
    a_total = 0
    speaker_diff = []
    non_speaker_diff = []
    train_Data = {'Flow' : [], 'Label' : []}
    for video_id in ids:
        vid_counts = [0,0,0,0]
        prev_frames = []
        trainLoader = Train_Loader(video_id=video_id, root_dir=video_id)
        trainLoaded = DataLoader(trainLoader, batch_size=64, num_workers=0, shuffle=False)

        for images, labels in trainLoaded:
            for i in range(len(images)):
                actual_label = trainLoader.extract_labels(trainLoader.labels, labels, i)
                print()
                print(labels['label'][i])
                tools.plot_frame(images[i].numpy())
                asd = ActiveSpeaker(images[i], prev_frames=prev_frames)
                prediction, prev_frames = asd.model()
                
                filtered = organise_data(prediction, actual_label)
                train_Data['Flow'].append(filtered['Flow'])
                train_Data['Label'].append(filtered['Label'])
                        










                # ------------------------

                # print(angles)
                # print(prev_labels)
                if labels['label'][i] == "Speaking":
                    speaker_diff.append(img_diff)
                else:
                    non_speaker_diff.append(img_diff)

                # tools.plot_frame(images[i].numpy())
                print('------------')
                tp, fp, tn, fn = evaluate(prediction, actual_label)
                counts[0] += tp
                counts[1] += fp
                counts[2] += tn
                counts[3] += fn

                vid_counts[0] += tp
                vid_counts[1] += fp
                vid_counts[2] += tn
                vid_counts[3] += fn

        p, r, fm = metrics(vid_counts)
        non_p, non_r, non_fm = metrics([vid_counts[2], vid_counts[3], vid_counts[0], vid_counts[1]])

        display_results('SPEAKING', vid_counts, p, r, fm)
        display_results('NON-SPEAKING', [vid_counts[2], vid_counts[3], vid_counts[0], vid_counts[1]], non_p, non_r, non_fm)

        a_total += trainLoader.__len__()

    p, r, fm = metrics(counts)
    non_p, non_r, non_fm = metrics([counts[2], counts[3], counts[0], counts[1]])


    display_results('GENERAL-SPEAKING', counts, p, r, fm)
    display_results('GENERAL-NON-SPEAKING', [counts[2], counts[3], counts[0], counts[1]],  non_p, non_r, non_fm)

    print('Total: ', np.sum(counts))
    print('Number of Frames ', a_total)
    print('\nAverage Precision: ', (p + non_p) /2)
    print('Macro F1: ', (non_fm+fm) /2)

    # conf_matrix(counts[0], counts[1], counts[2], counts[3])

# Function to print results
def display_results(title, counts, p, r, f):
    print(f'\n----------- {title} -----------')
    print('TP,FP,TN,FN: ',counts)
    print('Correct: ', counts[0]+counts[2])
    print('Precision: ', p)
    print('Recall: ', r)
    print('F-Measure: ', f)

def filter_faces(predicted_face, actual):
    if len(predicted_face) == 0:
        return False
    
    if torch.is_tensor(actual[1]):
        a_faces = actual[1].numpy()
    else:
        a_faces = actual[1]

     # Evaluates if there is more than one label for the frame
    if len(a_faces.shape) > 1:
        for i in range(len(a_faces)):
            for j in range(len(prediction)):
                # Checks if bounding box for face detected is correct
                # Then compares the predicted label with the actual label and returns the counts
                return face_evaluate(predicted_face, a_faces[i])
                
    return face_evaluate(predicted_face, a_faces)
                
def organise_data(prediction, actual):
    # FIlter Faces
    # Rearrange order to match labels
    vector = {'Flow' : [], 'Label' : []}
    
    if torch.is_tensor(actual[-1]):
        label = actual[-1].numpy()
    else:
        label = actual[-1]

    p_faces = prediction['faces']
    for i in range(len(p_faces)):
        if filter_faces(p_faces[i], actual):
            vector['Flow'].append(prediction['Flow'][i])
            if len(label.shape) > 1:
                vector['Label'].append(label[i])
            else:
                vector['Label'].append(label) 

    return vector

if __name__ == "__main__":
    main()