import os, argparse, cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from dataLoader import Train_Loader, Test_Loader, extract_labels
from asd import ActiveSpeaker
from models.model import SVM
from models.mobilenet import MobileNet
from models.shuffle import ShuffleNet
from models.train_vectors import train_model, test_model

from evaluation import *
from utils import tools
from sklearn.metrics import classification_report

train_ids = ['_mAfwH6i90E', 'B1MAUxpKaV8', '7nHkh4sP5Ks', '2PpxiG0WU18', '-5KQ66BBWC4', '5YPjcdLbs5g',
 '20TAGRElvfE', 'Db19rWN5BGo', 'rFgb2ECMcrY', 'N0Dt9i9IUNg', '8aMv-ZGD4ic', 'Ekwy7wzLfjc', 
 '0f39OWEqJ24']

# train_ids = ['_mAfwH6i90E']

test_ids = ['4ZpjKfu6Cl8', '2qQs3Y9OJX0', 'HV0H6oc4Kvs', 'rJKeqfTlAeY', '1j20qq1JyX4', 'C25wkwAMB-w']
# test_ids = ['rJKeqfTlAeY']

def main():
    parser = argparse.ArgumentParser(description = "Active Speaker Detection Program")
    parser.add_argument('--train', action='store_true', help="Perform training (True/False)")
    parser.add_argument('--test', action='store_true', help="Perform testing")
    parser.add_argument('--confMatrix', action='store_true',  required=False, help="Plot Confusion Matrix from testing")
    parser.add_argument('--roc', action='store_true',  required=False, help="Plot ROC curve from testing")
    parser.add_argument('--trainDataPath', type=str, default='train', required=False, help="Data path for the training dataset")
    parser.add_argument('--testDataPath', type=str, default='test', required=False, help="Data path for the testing dataset")
    parser.add_argument('--saveResults',  action='store_true', required=False, help='Save results from testing')

    parser.add_argument('--SVM', action='store_true', required=False, help='Selects Support Vector Machine to be used as classifer')
    parser.add_argument('--MobileNet', action='store_true', required=False, help='Selects MobileNetV3 Small to be used as classifer')
    parser.add_argument('--ShuffleNet', action='store_true', required=False, help='Selects ShuffleNetV2 to be used as classifer')
    parser.add_argument('--epochs', type=int, default=50, required=False, help='Select the number of epochs to train for (int)')
    parser.add_argument('--lr', type=float, default=0.001, required=False, help='Select the learing rate for training (float)')

    args = parser.parse_args()


    if args.train:
        data = feature_extract(ids=train_ids, root_dir=args.trainDataPath, train=True)
        data['Label'] = np.array(data['Label']).flatten().astype(np.int64)
        X_train = np.array(data['Flow'])
        Y_train = data['Label']
        
        if args.SVM:
            svm = SVM(False)
            model = svm.train(X_train, Y_train)
            svm.save_parameters(model)        

        if args.MobileNet:
            model = MobileNet()
            train_model(data, model, 'mobilenet_model.pth', args.epoch, args.lr)
            
        if args.ShuffleNet:
            model = ShuffleNet()
            train_model(data, model, 'shufflenet_model.pth', args.epoch, args.lr)


    if args.test:
        data = feature_extract(ids=test_ids, root_dir=args.testDataPath, train=False)
        X = np.array(data['Flow'])
        data['Label'] = np.array(data['Label']).flatten()
        test_y = data['Label'].astype(np.int64)

        if args.SVM:
            svm = SVM(True)
            predictions = svm.test(X)

        if args.MobileNet:
            model = MobileNet()
            predictions = test_model(data, load_path='mobilenet_model.pth')

        if args.ShuffleNet:
            model = ShuffleNet()
            predictions = test_model(data, load_path='shufflenet_model.pth')
        
        data['Pred'] = predictions
        print(classification_report(predictions, test_y))

        if args.saveResults:
            save_results(data)

        if args.confMatrix:
            conf_matrix(predictions, test_y)

        if args.roc:
            roc(X, test_y, y, svm.model)

def feature_extract(ids, root_dir, train):
    data = {'Id' : [], 'Timestamp': [], 'Flow' : [], 'Faces' : [], 'Label' : []}

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
                prev_frames = update_prev_frames(prev_frames, images[i].numpy(), prediction['Faces'])
                filtered = organise_data(prediction, actual_label)

                if len(filtered['Flow']) > 0 or len(filtered['Label']) > 0:
                    for i in range(len(filtered['Flow'])):
                        data['Flow'].append(filtered['Flow'][i])
                        data['Label'].append(filtered['Label'][i])
                        data['Timestamp'].append(filtered['Timestamp'])
                        data['Id'].append(video_id)
                        data['Faces'].append(filtered['Face'][i])

        print(f'{video_id} done')
 
    return data

def update_prev_frames(prev_frames, frame, faces):
    if len(prev_frames['Frame']) >= 5:
        _ = prev_frames['Frame'].pop(0)
        _ = prev_frames['Faces'].pop(0)
    prev_frames['Frame'].append(frame)
    prev_frames['Faces'].append(faces)
    return prev_frames


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
        return 0
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
    flow = []
    labels = []
    faces = []
    if torch.is_tensor(actual[-1]):
        label = actual[-1].numpy()
    else:
        label = actual[-1]

    p_faces = prediction['Faces']
    for i in range(len(p_faces)):
        c = filter_faces(p_faces[i], actual)
        if (prediction['Flow'][i] is not None) and (c != None):
            flow.append(prediction['Flow'][i])
            faces.append(p_faces[i])
            if len(actual[1].shape) > 1:
                labels.append(label[c])
            else:
                labels.append(label)

    return {'Timestamp' : actual[0], 'Flow' : flow, 'Face' : faces, 'Label' : labels}


def save_results(data):
    df = pd.DataFrame.from_dict(data)
    df.to_pickle('results.pkl')

if __name__ == "__main__":
    main()