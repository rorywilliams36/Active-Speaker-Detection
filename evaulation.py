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
    count = 0
    if len(predicteds) == 0 or len(actuals) == 0:
        if not torch.is_tensor(actuals[1]):
            return 0, len(actuals[1])
        return 0, 1

    if not torch.is_tensor(actuals[1]):
        for prediction in predicteds:
            x1, y1, x2, y2 = prediction[3:7] * 300
            for actual in actuals[1]:
                a_x1, a_y1, a_x2, a_y2 = actual * 300
                c_x = (a_x1 + a_x2) / 2
                c_y = (a_y1 + a_y2) / 2
                if (c_x >= x1 and c_x <= x2) and (c_y >= y1 and c_y <= y2):
                    count += 1
                    break

        total = len(actuals[1])

    else:
        for prediction in predicteds:
            x1, y1, x2, y2 = prediction[3:7] * 300
            a_x1, a_y1, a_x2, a_y2 = actuals[1] * 300
            c_x = (a_x1 + a_x2) / 2
            c_y = (a_y1 + a_y2) / 2
            if (c_x >= x1 and c_x <= x2) and (c_y >= y1 and c_y <= y2):
                count += 1

        total = 1
            
    return count, total


def evaluate(prediction, actual):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    if torch.is_tensor(actual[1]):
        a_faces = actual[1].numpy()
    else:
        a_faces = actual[1]

    a_label = actual[-1]
    p_faces = prediction['faces']
    p_labels = prediction['label']

    if len(p_labels) == 0 or len(p_faces) == 0:
        if len(a_faces.shape) > 1:
            return 0,0,0,0
        return 0,0,0,0 

    if len(a_faces.shape) > 1:
        for i in range(len(a_faces)):
            for j in range(len(p_faces)):
                if face_evaluate(p_faces[j], a_faces[i]):
                    tp, fp, tn, fn = label_eval(p_labels[j], a_label[i], [tp,fp,tn,fn])

                        
    else:
        for j in range(len(p_faces)):
            if face_evaluate(p_faces[j], a_faces):
                tp, fp, tn, fn = label_eval(p_labels[j], a_label, [tp,fp,tn,fn])

    return tp, fp, tn, fn


def face_evaluate(prediction, actual):
    x1, y1, x2, y2 = prediction * 300
    a_x1, a_y1, a_x2, a_y2 = actual * 300
    c_x = (a_x1 + a_x2) / 2
    c_y = (a_y1 + a_y2) / 2
    if (c_x >= x1 and c_x <= x2) and (c_y >= y1 and c_y <= y2):
        return True
    return False

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

def metrics(counts):
    tp,fp,tn,fn = counts
    if tp+fp == 0:
        precision = 0
    else:
        precision = tp / (tp+fp)

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

def conf_matrix(tp,fp,tn,fn):
    sns.heatmap([[tn, fp],[fn, tp]], cmap='crest', annot=True, fmt='.0f', 
                xticklabels=['NOT_SPEAKING', 'SPEAKING'], yticklabels=['NOT_SEAKING', 'SPEAKING'])

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()