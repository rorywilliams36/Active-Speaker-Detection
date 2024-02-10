import cv2
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Displays the frame and label on the image
# Frame must be converted to a numpy array 
def show_labels(frame, label):
    y_dims, x_dims = frame.shape[:2]
    x1, y1 = (label[1], label[2])
    x2, y2 = (label[3], label[4])
    speak = label[-1]

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
def show_frame(frame):
    plt.imshow(frame)
    plt.show()

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



# Converts and saves the video as a series of frames/images
def split_into_frames(video_id):
    current_path = os.getcwd()
    output = f'/dataset/{video_id}'
    cap = cv2.VideoCapture(f'{current_path}/dataset/{video_id}.mkv')

    if not os.path.exists(f'{current_path}/{output}'):
        os.makedirs(f'{current_path}/{output}')

    while True:
        ret, frame = cap.read()
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)	

        if ret:
            if timestamp >= 900000 and timestamp < 912000:
                norm_time = round((timestamp / 1000), 2) # Converts milliseconds to seconds ot match the timestamps in the labels
                cv2.imwrite(f'{current_path}/{output}/{video_id}_{norm_time}.jpg', frame)
                    
            if timestamp > 912000:
                break

        else:
            print('Error occured whilst prepping video')
            break

    cap.release()