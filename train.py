import os, argparse, cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from dataLoader import Train_Loader, Val_Loader
from asd import ActiveSpeaker
from model import SVM
from evaluation import *
from utils import tools

# ids = ['_mAfwH6i90E', 'B1MAUxpKaV8', '7nHkh4sP5Ks', '2PpxiG0WU18', '-5KQ66BBWC4', '5YPjcdLbs5g', '20TAGRElvfE', '2fwni_Kjf2M']
ids = ['_mAfwH6i90E']

def main():
    # parser = argparse.ArgumentParser(description = "Training Stage")
    # parser.add_argument('face_detect_threshold', type=float, default='0.5', help="Confidence score for face detection")
    # parser.add_argument('face_detect_model', type=str, default='res10_300x300_ssd_iter_140000.caffemodel', help="OpenCV model used for face detection")
    
    # parser.add_argument('test', type=bool, default=False, help="Perform testing (True/False)")

    # parser.add_argument('evaluate', type=bool, default=False, help="Perform Evaluation (True/False)")

    # parser.add_argument('train_dir', type=str, default='train', help="Data path for the training data")
    # parser.add_argument('test_dir', type=str, default='test', help="Data path for the testing data")

    # parser.add_arguement('train_flow_vector', type=str, default=None, help='Data path to csv file containing flow values and labels for training')
    # parser.add_arguement('test_flow_vector', type=str, default=None, help='Data path to csv file containing flow values for testing')

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
                # print()
                # print(labels['label'][i])
                # tools.plot_frame(images[i].numpy())
                asd = ActiveSpeaker(images[i], prev_frames=prev_frames)
                prediction, prev_frames = asd.model()

                filtered = organise_data(prediction, actual_label)
                if len(filtered['Flow']) > 0 or len(filtered['Label']) > 0:
                    for i in range(len(filtered['Flow'])):
                        train_Data['Flow'].append(filtered['Flow'][i])
                        train_Data['Label'].append(filtered['Label'][i])

                # ------------------------
                tp, fp, tn, fn = evaluate(prediction, actual_label)

                vid_counts[0] += tp
                vid_counts[1] += fp
                vid_counts[2] += tn
                vid_counts[3] += fn
            
        counts[0] += vid_counts[0]
        counts[1] += vid_counts[1] 
        counts[2] += vid_counts[2]
        counts[3] += vid_counts[3]

        a_total += trainLoader.__len__()

    # display_evaluate(counts, a_total)


    # conf_matrix(counts[0], counts[1], counts[2], counts[3])

    # -------------------

    train_Data['Label'] = np.array(train_Data['Label']).flatten()

    X_train = np.array(train_Data['Flow'])
    print(X_train)
    Y_train = train_Data['Label']
    classify = SVM(False)
    model = classify.train(X_train, Y_train)


# Function to print results
def display_results(title, counts, p, r, f):
    print(f'\n----------- {title} -----------')
    print('TP,FP,TN,FN: ',counts)
    print('Correct: ', counts[0]+counts[2])
    print('Precision: ', p)
    print('Recall: ', r)
    print('F-Measure: ', f)

# Function to present results
def display_evaluate(counts, total):
    p, r, fm = metrics(counts)
    non_p, non_r, non_fm = metrics([counts[2], counts[3], counts[0], counts[1]])

    display_results('SPEAKING', counts, p, r, fm)
    display_results('NON-SPEAKING', [counts[2], counts[3], counts[0], counts[1]], non_p, non_r, non_fm)

    print('Total: ', np.sum(counts))
    print('Number of Frames ', total)
    print('\nAverage Precision: ', (p + non_p) /2)
    print('Macro F1: ', (non_fm+fm) /2)


def filter_faces(predicted_face, actual):
    '''
    Function to remove faces which have been detected but aren't in the actual labels for the frame
    
    params:
        predicted_face: array containing coordinates for bounding box
        actual: array/tensor of labels for the frame
    
    returns: boolean whether face is in label or not
    '''
    if len(predicted_face) == 0:
        return False
    
    if torch.is_tensor(actual[1]):
        a_faces = actual[1].numpy()
    else:
        a_faces = actual[1]

     # Evaluates if there is more than one label for the frame
    if len(a_faces.shape) > 1:
        for i in range(len(a_faces)):
            check = False
            # Checks if bounding box for face detected is correct
            # Then compares the predicted label with the actual label and returns the counts
            if face_evaluate(predicted_face, a_faces[i]):
                return True
        return False
            
                
    return face_evaluate(predicted_face, a_faces)
                
def organise_data(prediction, actual, train=True):
    '''
    Function to organise the flow vectors with corresponding labels

    params:
        prediction: dict containing the predicted face and label
        actual: dict containing the actual label for the frame
        train: boolean indicating training or testing
    
    returns: 
        vector: dict containing flow values with corresponding label
    '''
    vector = {'Flow' : [], 'Label' : []}
    flow = []
    labels = []
    if torch.is_tensor(actual[-1]):
        label = actual[-1].numpy()
    else:
        label = actual[-1]

    p_faces = prediction['Faces']
    for i in range(len(p_faces)):
        if filter_faces(p_faces[i], actual):

            if prediction['Flow'][i] is not None:
                flow.append(prediction['Flow'][i])
                if train:
                    if len(actual[1].shape) > 1:
                        labels.append(label[i])
                    else:
                        labels.append(label)

    return {'Flow' : flow, 'Label' : labels}

if __name__ == "__main__":
    main()