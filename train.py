import os, argparse, cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from dataLoader import Train_Loader, Test_Loader, extract_labels
from asd import ActiveSpeaker
from model import SVM
from evaluation import *
from utils import tools

train_ids = ['_mAfwH6i90E', 'B1MAUxpKaV8', '7nHkh4sP5Ks', '2PpxiG0WU18', '-5KQ66BBWC4', '5YPjcdLbs5g', '20TAGRElvfE', 'Db19rWN5BGo', 'rFgb2ECMcrY', 'N0Dt9i9IUNg']
test_ids = ['4ZpjKfu6Cl8', '2qQs3Y9OJX0', 'HV0H6oc4Kvs', 'KHHgQ_Pe4cI']

def main():
    parser = argparse.ArgumentParser(description = "Active Speaker Detection Program")
    parser.add_argument('--train', action='store_true', help="Perform training (True/False)")
    parser.add_argument('--n_iter', type=int, required=False, default=100, help="Number of training iterations performed (Int)")
    parser.add_argument('--loss', action='store_true', help="Show loss function for model (Training must be selected)")
    parser.add_argument('--test', action='store_true', help="Perform testing")
    parser.add_argument('--evaluate',  action='store_true', required=False, help="Perform Evaluation")
    parser.add_argument('--trainDataPath', type=str, default='train', required=False, help="Data path for the training dataset")
    parser.add_argument('--testDataPath', type=str, default='test', required=False, help="Data path for the testing dataset")
    parser.add_argument('--trainFlowVector', type=str, default=None, required=False, help='Data path to csv file containing flow values and labels for training')
    parser.add_argument('--testFlowVector', type=str, default=None, required=False, help='Data path to csv file containing flow values for testing')
    parser.add_argument('--saveResults',  action='store_true', required=False, help='Save results from testing')
    parser.add_argument('--loadCustModel', type=str, default=None, required=False, help='Data path to presaved model used for classification')
    parser.add_argument('--loadPreviousModel', action='store_true', required=False, help='Boolean value to use the previously trained model')

    args = parser.parse_args()

    if args.train:
        data = feature_extract(ids=train_ids, root_dir=args.trainDataPath, train=True)
        data['Label'] = np.array(data['Label']).flatten()
        X_train = np.array(data['Flow'])
        Y_train = data['Label'].astype(np.int64)
        svm = SVM(False, None, args.n_iter)
        model = svm.train(X_train, Y_train)
        svm.save_parameters(model)
        print(model.best_estimator_)
        print(model.best_params_)

        if args.loss:
            y = svm.test(X_train)
            print(svm.loss(X_train, y))
            print(svm.model.predict_proba)

    if args.test:
        data = feature_extract(ids=test_ids, root_dir=args.testDataPath, train=False)
        svm = SVM(args.loadPreviousModel, args.loadCustModel, args.n_iter)
        print(data['Flow'])
        X = np.array(data['Flow'])
        y = svm.test(X)

        if args.evaluate:
            data['Label'] = np.array(data['Label']).flatten()
            test_y = data['Label'].astype(np.int64)
            print(svm.evaluate(y, test_y))

def feature_extract(ids, root_dir, train):
    data = {'Flow' : [], 'Label' : []}

    print('Extracting features\n')
    for video_id in ids:
        prev_frames = {'Frame' : [], 'Faces' : []}
        if train:
            dataLoader = Train_Loader(video_id, root_dir)
        else:
            dataLoader = Test_Loader(video_id, root_dir)

        dataLoaded = DataLoader(dataLoader, batch_size=64, num_workers=0, shuffle=False)

        for images, labels in dataLoaded:
            for i in range(len(images)):
                actual_label = extract_labels(dataLoader.labels, labels, i)
                asd = ActiveSpeaker(images[i], prev_frames=prev_frames)
                prediction = asd.model()

                prev_frames['Frame'] = images[i].numpy()
                prev_frames['Faces'] = prediction['Faces']
                filtered = organise_data(prediction, actual_label)

                if len(filtered['Flow']) > 0 or len(filtered['Label']) > 0:
                    for i in range(len(filtered['Flow'])):
                        data['Flow'].append(filtered['Flow'][i])
                        data['Label'].append(filtered['Label'][i])

        print(f'{video_id} done')
 
    return data

def filter_faces(predicted_face, actual):
    '''
    Function to remove faces which have been detected but aren't in the actual labels for the frame
    
    params:
        predicted_face: array containing coordinates for bounding box
        actual: array/tensor of labels for the frame
    
    returns: index of the corresponding face detected compared to the label
    '''
    if len(predicted_face) == 0:
        return None
    
    if torch.is_tensor(actual[1]):
        a_faces = actual[1].numpy()
    else:
        a_faces = actual[1]

     # Evaluates if there is more than one label for the frame
    if len(a_faces.shape) > 1:
        for i in range(len(a_faces)):
            # Checks if bounding box for face detected is correct
            # Then compares the predicted label with the actual label and returns the counts
            if face_evaluate(predicted_face, a_faces[i]) and check_centres(predicted_face, a_faces[i]):
                return i
        return None
            
    if face_evaluate(predicted_face, a_faces) and check_centres(predicted_face, a_faces):
        return True
    return None
                
def organise_data(prediction, actual):
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
        c = filter_faces(p_faces[i], actual)
        if (prediction['Flow'][i] is not None) and (c != None):
            flow.append(prediction['Flow'][i])
            if len(actual[1].shape) > 1:
                labels.append(label[c])

            else:
                labels.append(label)

    return {'Flow' : flow, 'Label' : labels}

if __name__ == "__main__":
    main()