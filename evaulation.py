import cv2, torch
import numpy as np

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
    if torch.is_tensor(actual[1]):
        a_faces = actual[1].numpy()
    else:
        a_faces = actual[1]

    a_label = actual[-1]
    p_faces = prediction['faces']
    p_labels = prediction['label']
    t_correct = 0
    total = 0

    if len(p_labels) == 0 or len(p_faces) == 0:
        if len(a_faces.shape) > 1:
            return len(a_faces), 0 
        return 1, 0

    if len(a_faces.shape) > 1:
        for i in range(len(a_faces)):
            for j in range(len(p_faces)):
                if face_evaluate(p_faces[j], a_faces[i]):
                    if p_labels[j] == a_label[i]:
                        t_correct += 1
                        break

            total += 1
    else:
        for j in range(len(p_faces)):
            if face_evaluate(p_faces[j], a_faces):
                if p_labels[j] == a_label:

                    t_correct += 1
                    break
        total += 1


    return total, t_correct




def face_evaluate(prediction, actual):
    x1, y1, x2, y2 = prediction * 300
    a_x1, a_y1, a_x2, a_y2 = actual * 300
    c_x = (a_x1 + a_x2) / 2
    c_y = (a_y1 + a_y2) / 2
    if (c_x >= x1 and c_x <= x2) and (c_y >= y1 and c_y <= y2):
        return True
    return False