import cv2, torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay

'''
Calculate Precision, Recall, F-Measure
Calculate mAP

Compare predicted labels with actual labels

'''

def general_face_evaluation(prediction, actual):
    '''
    Function to return whether face detected is correct

    Params:
        prediction: array of faces detected
        actual: Annotations/labels for the frame

    Return: Correct count, total faces count
    '''
    correct = 0
    # Get face bound boxes for the labels
    if torch.is_tensor(actual[1]):
        a_faces = actual[1].numpy()
    else:
        a_faces = actual[1]

    # Gets number of faces in frame
    if len(a_faces.shape) > 1:
        total = a_faces.shape[0]
    else:
        total = 1

    # If no label is returned for the frame 
    if len(prediction) == 0:
        return 0, total

    # Evaluates if there is more than one label for the frame
    if len(a_faces.shape) > 1:
        for i in range(len(a_faces)):
            for j in range(len(prediction)):
                # Checks if bounding box for face detected is correct
                # Then compares the predicted label with the actual label and returns the counts
                if face_evaluate(prediction[j][3:7], a_faces[i]):
                    correct += 1

    # Evaluation for frames with single labels   
    else:
        for j in range(len(prediction)):
            if face_evaluate(prediction[j][3:7], a_faces):
                correct += 1
    
    return correct, total


# Evaluation function for the prediction and actual label
def evaluate(prediction, actual):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # Get face bound boxes for the labels
    if torch.is_tensor(actual[1]):
        a_faces = actual[1].numpy()
    else:
        a_faces = actual[1]

    a_label = actual[-1]
    p_faces = prediction['Faces']
    p_labels = prediction['Label']

    # If no label is returned for the frame 
    if len(p_labels) == 0 or len(p_faces) == 0:
        return 0,0,0,0 

    # Evaluates if there is more than one label for the frame
    if len(a_faces.shape) > 1:
        for i in range(len(a_faces)):
            for j in range(len(p_faces)):
                # Checks if bounding box for face detected is correct
                # Then compares the predicted label with the actual label and returns the counts
                if face_evaluate(p_faces[j], a_faces[i]):
                    tp, fp, tn, fn = label_eval(p_labels[j], a_label[i], [tp,fp,tn,fn])
       
    # Evaluation for frames with single labels   
    else:
        for j in range(len(p_faces)):
            if face_evaluate(p_faces[j], a_faces):
                tp, fp, tn, fn = label_eval(p_labels[j], a_label, [tp,fp,tn,fn])

    return tp, fp, tn, fn


# Evaluation for face detection
# Since the coords for bound boxes are normalised we check if they contain the same centre coordinate
# This is also because each bound box detected aren't exactly the same size/area
def face_evaluate(prediction, actual):
    x1, y1, x2, y2 = prediction * 300
    a_x1, a_y1, a_x2, a_y2 = actual * 300
    return (x1 <= a_x2) and (x2 >= a_x1) and (y1 <= a_y2) and (y2 >= a_y1)

def check_centres(prediction, actual):
    x1, y1, x2, y2 = prediction * 300
    a_x1, a_y1, a_x2, a_y2 = actual * 300
    c_x, c_y = ((a_x1+a_x2)/2, (a_y1+a_y2)/2)
    return (c_x >= x1 and c_x <= x2) and (c_y >= y1 and c_y <= y2)

# Function to compare the predicted label with the actual and update the metrics
def label_eval(prediction, actual, counts):
    tp, fp, tn, fn = counts
    if prediction == actual:
        if prediction == 1:
            tp += 1
        else:
            tn += 1
    else:
        if prediction == 1 and actual != 1:
            fp += 1
        else:
            fn += 1

    return tp, fp, tn, fn

# Calculates evaluation metrics from the give counts
def metrics(counts):
    tp,fp,tn,fn = counts

    # Catch dividing by 0
    if tp+fp == 0:
        precision = 0
    else:
        precision = tp / (tp+fp)

    # Catch dividing by 0
    if tp+fn == 0:
        recall = 0
    else:
        recall =  tp / (tp+fn)
    
    if precision == 0 and recall == 0:
        return 0,0,0
         
    f_measure = (2 * precision * recall) / (precision + recall)

    return precision, recall, f_measure

def mean_avg_precision():
    pass

# Calculates confusion matrix using seaborn
def conf_matrix(tp,fp,tn,fn):
    sns.heatmap([[tn, fp],[fn, tp]], cmap='crest', annot=True, fmt='.0f', 
                xticklabels=['NOT_SPEAKING', 'SPEAKING'], yticklabels=['NOT_SEAKING', 'SPEAKING'])

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


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