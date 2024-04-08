import os, argparse, cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from dataLoader import Train_Loader, Test_Loader
from asd import ActiveSpeaker
from model import SVM
from evaluation import *
from utils import tools

# ids = ['_mAfwH6i90E', 'B1MAUxpKaV8', '7nHkh4sP5Ks', '2PpxiG0WU18', '-5KQ66BBWC4', '5YPjcdLbs5g', '20TAGRElvfE', '2fwni_Kjf2M']
# ids = ['_mAfwH6i90E']
ids = ['_mAfwH6i90E', 'B1MAUxpKaV8', '20TAGRElvfE', '-5KQ66BBWC4', '7nHkh4sP5Ks']

def main():
    parser = argparse.ArgumentParser(description = "Active Speaker Detection Program")
    parser.add_argument('--face_detect_threshold', type=float, default='0.5', required=False, help="Confidence score for face detection")
    parser.add_argument('--face_detect_model', type=str, default='res10_300x300_ssd_iter_140000.caffemodel', help="OpenCV model used for face detection")
    parser.add_argument('--train', type=bool, required=True, default=True, help="Perform training (True/False)")
    parser.add_argument('--loss', type=bool, required=False, default=False, help="Show loss function for model (Training must be selected)")
    parser.add_argument('--test', type=bool, required=False, default=False, help="Perform testing (True/False)")
    parser.add_argument('--evaluate', type=bool, default=False, required=False, help="Perform Evaluation (True/False)")
    parser.add_argument('--trainDataPath', type=str, default='train', required=False, help="Data path for the training dataset")
    parser.add_argument('--testDataPath', type=str, default='test', required=False, help="Data path for the testing dataset")
    parser.add_argument('--trainFlowVector', type=str, default=None, required=False, help='Data path to csv file containing flow values and labels for training')
    parser.add_argument('--testFlowVector', type=str, default=None, required=False, help='Data path to csv file containing flow values for testing')
    parser.add_argument('--saveResults', type=bool, default=False, required=False, help='Save results from testing')
    parser.add_argument('--loadCustModel', type=str, default=None, required=False, help='Data path to presaved model used for classification')
    parser.add_argument('--loadPreviousModel', type=bool, default=True, required=False, help='Boolean value to use the previously trained model')

    args = parser.parse_args()

    if args.train:
        data = feature_extract(ids=ids, root_dir='')
        data['Label'] = np.array(data['Label']).flatten()
        X_train = np.array(data['Flow'])
        Y_train = data['Label'].astype(np.int64)
        classify = SVM(False, None)
        model = classify.train(X_train, Y_train)
        if args.loss:
            y = classify.test(X_train)
            print(classify.loss(X_train, y))

    if args.test:
        model = classify.load_parameters()
        data = feature_extract(ids='', root_dir='')
        classify = SVM(args.loadPreviousModel, args.loadCustModel)
        X = np.array(data['Flow'])
        y = classify.test(X)

def feature_extract(ids, root_dir):
    data = {'Flow' : [], 'Label' : []}

    for video_id in ids:
        prev_frames = {'Frame' : [], 'Faces' : []}
        trainLoader = Train_Loader(video_id=video_id, root_dir=video_id)
        trainLoaded = DataLoader(trainLoader, batch_size=64, num_workers=0, shuffle=False)

        for images, labels in trainLoaded:
            for i in range(len(images)):
                actual_label = trainLoader.extract_labels(trainLoader.labels, labels, i)
                asd = ActiveSpeaker(images[i], prev_frames=prev_frames)
                prediction = asd.model()

                prev_frames['Frame'] = images[i].numpy()
                prev_frames['Faces'] = prediction['Faces']
                filtered = organise_data(prediction, actual_label)

                if len(filtered['Flow']) > 0 or len(filtered['Label']) > 0:
                    for i in range(len(filtered['Flow'])):
                        data['Flow'].append(filtered['Flow'][i])
                        data['Label'].append(filtered['Label'][i])

    return data


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
    
    returns: index of the corresponding face detected compared to the label
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
                return i
        return None
            
                
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
        pos = filter_faces(p_faces[i], actual)
        if (prediction['Flow'][i] is not None) and (pos is not None):
            flow.append(prediction['Flow'][i])
            if train:
                if len(label) > 1:
                    print(p_faces)
                    print(actual)
                    print(label, pos)
                    labels.append(label[pos])
                else:
                    labels.append(label)

    return {'Flow' : flow, 'Label' : labels}

if __name__ == "__main__":
    main()