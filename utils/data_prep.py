
import cv2
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

current_path = os.getcwd()
current_path = '\\'.join(current_path.split('\\')[:-1])
ALL_TRAIN_LABELS = f'{current_path}/dataset/ava_activespeaker_train_v1.0/'

# Converts and saves the video as a series of frames/images
def split_into_frames(video_id):
    labels = pd.read_csv(f'{ALL_TRAIN_LABELS}{video_id}-activespeaker.csv')
    labels.columns = ['Video_ID', 'Timestamp', 'x1', 'y1', 'x2', 'y2', 'label', 'face_track_id']

    output = f'/dataset/{video_id}'
    cap = cv2.VideoCapture(f'{current_path}/dataset/{video_id}.mkv')

    if not os.path.exists(f'{current_path}/{output}'):
        os.makedirs(f'{current_path}/{output}')

    count = 0
    while True:
        ret, frame = cap.read()
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)	

        if ret:
            # Labels are only assigned between 900s and 1500s (timestamp is in msec)
            if timestamp >= 900000 and count <= 500:
                # Converts milliseconds to seconds ot match the timestamps in the labels
                norm_time = round((timestamp / 1000), 2) 

                # Only gets frames with a corresponding label
                for i in range(len(labels)):
                    label_timestamp = labels.at[i, "Timestamp"]
                    if (label_timestamp == norm_time) or (label_timestamp == (norm_time - 0.01)) or (label_timestamp == (norm_time + 0.01)):
                        norm_time = label_timestamp
                        cv2.imwrite(f'{current_path}/{output}/{video_id}_{norm_time}.jpg', frame)
                        count += 1
                    
            if count > 500:
                break

        else:
            print('Error occured whilst prepping video')
            break

    cap.release()

if __name__ == "__main__":
    split_into_frames('_mAfwH6i90E')