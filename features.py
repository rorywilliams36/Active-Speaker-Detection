'''
features.py

Module to load frames as features by finding the locations of each face
of each speaker in frame and organising it in a dict so that
the face, optical flow and labels are all stored together

'''

import cv2
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader

from dataLoader import Train_Loader, Test_Loader, extract_labels
from asd import ActiveSpeaker
from evaluation import face_evaluate
from utils.misc import check_centres

train_ids = ['_mAfwH6i90E', 'B1MAUxpKaV8', '7nHkh4sP5Ks', '2PpxiG0WU18', '-5KQ66BBWC4', '5YPjcdLbs5g',
'20TAGRElvfE', 'Db19rWN5BGo', 'rFgb2ECMcrY', 'N0Dt9i9IUNg', '8aMv-ZGD4ic', 'Ekwy7wzLfjc', 
'0f39OWEqJ24']

test_ids = ['4ZpjKfu6Cl8', '2qQs3Y9OJX0', 'HV0H6oc4Kvs', 'rJKeqfTlAeY', '1j20qq1JyX4', 'C25wkwAMB-w']

obst_ids = ['4ZpjKfu6Cl8', 'HV0H6oc4Kvs', '1j20qq1JyX4', 'KHHgQ_Pe4cI', 'BCiuXAuCKAU']

MODEL_PATH = '/parameter_files'

def feature_extract(ids, root_dir, train, svm_check):
    '''
    Feature Extraction
    Loads all frames an acquires the relevant features and stores in dictionary

    Args:
        ids: Array of video ids to be loaded
        root_dir: Path of dataset (training or testing)
        train: boolean indicating training/testing
        svm_check: boolean to indicate svm being used

    Return:
        data: Dictionary storing features for relevant frame {ID, Timestamp, Flow, Faces, Label}
    '''

    data = {'Id' : [], 'Timestamp': [], 'Flow' : [], 'Faces' : [], 'Label' : []}

    print('Extracting features\n')
    for video_id in ids:
        prev_frames = {'Frame' : [], 'Faces' : []}
        # Loads training or testing data
        if train:
            data_loader = Train_Loader(video_id, root_dir)
        else:
            data_loader = Test_Loader(video_id, root_dir)

        data_loaded = DataLoader(data_loader, batch_size=64, num_workers=0, shuffle=False)

        for images, labels in data_loaded:
            for i, img in enumerate(images):

                # Checks if there is multiple labels associated with frame
                actual_label = extract_labels(data_loader.labels, labels, i)
                # Feature Extraction
                # Stores features in dict
                asd = ActiveSpeaker(img, prev_frames=prev_frames, svm=svm_check)
                prediction = asd.model()
                prev_frames = update_prev_frames(prev_frames, img.numpy(), prediction['Faces'])

                # Filters out any features with a label associated
                filtered = organise_data(prediction, actual_label)

                # Creates new dictionary for filtered features
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
    '''
    Function to update the previous frame dictionary (acts as a stack data structure)
    Once at certain size the oldest item is removed and new item is added
    '''
    if len(prev_frames['Frame']) >= 5:
        _ = prev_frames['Frame'].pop(0)
        _ = prev_frames['Faces'].pop(0)
    prev_frames['Frame'].append(frame)
    prev_frames['Faces'].append(faces)
    return prev_frames


def filter_faces(predicted_face, actual):
    '''
    Function to remove faces which have been detected but aren't in the actual labels for the frame
    
    Args:
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

    Args:
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
        if (prediction['Flow'][i] is not None) and (c is not None):
            flow.append(prediction['Flow'][i])
            faces.append(p_faces[i])
            if len(actual[1].shape) > 1:
                labels.append(label[c])
            else:
                labels.append(label)

    return {'Timestamp' : actual[0], 'Flow' : flow, 'Face' : faces, 'Label' : labels}

def save_results(data):
    ''' Saves results from testing '''
    df = pd.DataFrame.from_dict(data)
    df.to_pickle('results.pkl')