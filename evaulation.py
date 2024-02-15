import cv2, torch
import numpy as np

'''
Calculate Precision, Recall, F-Measure
Calculate mAP

Compare predicted labels with actual labels

'''

# For single labels
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
                    # print(True)
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
                # print(True)
                count += 1
        total = 1
            
    return count, total

# if __name__ == "__main__":

