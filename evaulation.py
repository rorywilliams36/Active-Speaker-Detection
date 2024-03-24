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

# General evaluation for face detection
def eval_face_detection(predicteds, actuals):
    correct = 0
    if len(predicteds) == 0 or len(actuals) == 0:
        if not torch.is_tensor(actuals[1]):
            return 0, len(actuals[1])
        return 0, 1

    if not torch.is_tensor(actuals[1]):
        for prediction in predicteds:
            x1, y1, x2, y2 = prediction[3:7] * 300
            for actual in actuals[1]:
                if face_evaluate(prediction, actual):
                    correct += 1

        total = len(actuals[1])

    else:
        for prediction in predicteds:
            if face_evaluate(prediction, actuals[1]):
                correct += 1

        total = 1
            
    return count, total


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
    p_faces = prediction['faces']
    p_labels = prediction['label']

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
    c_x = (a_x1 + a_x2) / 2
    c_y = (a_y1 + a_y2) / 2
    if (c_x >= x1 and c_x <= x2) and (c_y >= y1 and c_y <= y2):
        return True
    return False

# Function to compare the predicted label with the actual and update the metrics
def label_eval(prediction, actual, counts):
    tp, fp, tn, fn = counts
    if prediction == actual:
        if prediction == 'SPEAKING':
            tp += 1
        else:
            tn += 1
    else:
        if prediction == 'SPEAKING' and actual != 'SPEAKING':
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