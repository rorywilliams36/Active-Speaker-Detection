import cv2
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Displays the frame and given label on the image
# Frame must be converted to a numpy array 
def show_labels(frame, labels):
    y_dims, x_dims = frame.shape[:2]
    for i in labels:

        x1, y1 = (labels[i][1], labels[i][1])
        x2, y2 = (labels[i][1], labels[i][1])
        speak = labels[i][1]

        x1 = round(float(x1)*x_dims)
        y1 = round(float(y1)*y_dims)
        x2 = round(float(x2)*x_dims)
        y2 = round(float(y2)*y_dims)

        if speak == 0:
            c = (0,0,255)
        else:
            c = (0,255,0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color=c)

    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Shows frame without label
def plot_frame(frame):

    # Scales frame up if frame is too small
    if frame.shape[0] < 100 or frame.shape[1] < 100:
        frame = cv2.resize(frame, (150, 150))

    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Plots bounding boxes on faces
def plot_faces_detected(frame, faces):
    h, w = frame.shape[:2]

    for face in faces:
        x1, y1, x2, y2 = face[3:7] 
        x1 = round(float(x1)*w)
        y1 = round(float(y1)*h)
        x2 = round(float(x2)*w)
        y2 = round(float(y2)*h) 
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Plots single bounding box from coords
def plot_box(frame, coords):
    x1, y1, x2, y2 = coords[:]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


