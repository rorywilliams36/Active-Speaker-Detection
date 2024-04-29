import cv2
import pandas as pd
import numpy as np

def get_face_coords(face, h, w):
    # Gets coordinates of bounding box
    if len(face) > 4:
        x1, y1, x2, y2 = face[3:7] * h
    else:
        x1, y1, x2, y2 = face * h

    # Grabs extra pixels around bounding box to account for errors and also check ranges
    x1 = max(round(float(x1))-5, 0)
    y1 = max(round(float(y1))-5, 0)
    x2 = min(round(float(x2))+5, w)
    y2 = min(round(float(y2))+5, h)
    return x1, y1, x2, y2

def check_face(current, previous):
    x1,y1,x2,y2 = current
    p_x1, p_y1, p_x2, p_y2 = previous
    return (x1 <= p_x2) and (x2 >= p_x1) and (y1 <= p_y2) and (y2 >= p_y1)

def check_centres(prediction, actual):
    x1, y1, x2, y2 = prediction * 300
    a_x1, a_y1, a_x2, a_y2 = actual * 300
    c_x, c_y = ((a_x1+a_x2)/2, (a_y1+a_y2)/2)
    return (c_x >= x1 and c_x <= x2) and (c_y >= y1 and c_y <= y2)


def percent_overlap(prediction, actual):
    x1, y1, x2, y2 = prediction * 300
    a_x1, a_y1, a_x2, a_y2 = actual * 300
    # Get area of each bounding box
    # +1 used to avoid 0 values
    predicted_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    actual_area = (a_x2 - a_x1 + 1) * (a_y2 - a_y1 + 1)

    # Get coordinates for the intersection box
    m_x1 = max(x1, a_x1)
    m_x2 = min(x2, a_x2)

    m_y1 = max(y1, a_y1)
    m_y2 = min(y2, a_y2)

    # Calculate intersection and union areas to get IoU
    inter = max(0, (m_x2 - m_x1 + 1)) * max(0, (m_y2 - m_y1 + 1))

    return (inter / min(predicted_area, actual_area)) * 100
