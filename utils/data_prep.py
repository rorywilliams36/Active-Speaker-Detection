import os, cv2, subprocess
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

current_path = os.getcwd()
current_path = '\\'.join(current_path.split('\\')[:-1])
ALL_TRAIN_LABELS = f'{current_path}/dataset/ava_activespeaker_train_v1.0/'
ALL_TEST_LABELS = f'{current_path}/dataset/ava_activespeaker_test_v1.0/'

URL = 'https://s3.amazonaws.com/ava-dataset/trainval/'
train_files = ['B1MAUxpKaV8.mkv']
test_files = ['2qQs3Y9OJX0.mkv', '4ZpjKfu6Cl8.mkv']

# Converts and saves the video as a series of frames/images
def split_into_frames(video_id, test):
    if test:
        print(ALL_TEST_LABELS)
        labels = pd.read_csv(f'{ALL_TEST_LABELS}{video_id}-activespeaker.csv')

    else:
        labels = pd.read_csv(f'{ALL_TRAIN_LABELS}{video_id}-activespeaker.csv')
    labels.columns = ['Video_ID', 'Timestamp', 'x1', 'y1', 'x2', 'y2', 'label', 'face_track_id']

    output = f'/dataset/{video_id}'
    cap = cv2.VideoCapture(f'{current_path}/dataset/{video_id}.mkv')

    start = labels.at[0, 'Timestamp'] - 0.04
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip = round(fps/10)
        
    # if not os.path.exists(f'{current_path}/{output}'):
    #     os.makedirs(f'{current_path}/{output}')

    frame_count = 0
    count = 0
    stop = 60
    while True and count < stop:
        ret, frame = cap.read()
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)	

        if ret:
        # Labels are only assigned between 900s and 1500s (timestamp is in msec)
        # Converts milliseconds to seconds ot match the timestamps in the labels
            norm_time = round((timestamp / 1000), 2)
            # Only gets frames with a corresponding label
            if norm_time >= 1242 and count <= stop:
                for i in range(len(labels)):
                    label_timestamp = labels.at[i, "Timestamp"]
                    if frame_count % skip == 0:                    
                        if (label_timestamp == norm_time) or ((label_timestamp >= (norm_time - 0.02)) and (label_timestamp <= (norm_time + 0.02))):
                            cv2.imwrite(f'{current_path}/dataset/trial/{video_id}_{label_timestamp}.jpg', frame)
                            print(count)
                            print(norm_time)
                            print(label_timestamp)
                            count += 1
                            break

            frame_count += 1
  
        else:
            print('Error occured whilst prepping video')
            break

    cap.release()

def save_facetracks(face, label, trainLoader, face_coords, index):
    output = 'facetracks'
    if not os.path.exists(f'{path}/dataset/{output}'):
        os.makedirs(f'{path}/dataset/{output}')

    pos_labels = trainLoader.extract_all(trainLoader.labels, label, index)
    if len(pos_labels) > 1:
        for l in pos_labels:
            facetrack_id = l[-1].split(':')[-1]
            if get_face_track_from_coords(face_coords, l[2:-2]):
                cv2.imwrite(f'{path}/dataset/{output}/{l[0]}_{facetrack_id}_{l[1]}.jpg', face)

    else:
        l = pos_labels[0]
        facetrack_id = l[-1].split(':')[-1]
        cv2.imwrite(f'{path}/dataset/{output}/{l[0]}_{facetrack_id}_{l[1]}.jpg', face)
    
def get_face_track_from_coords(face, coords):
    x1, y1, x2, y2 = face[3:7] * 300
    a_x1, a_y1, a_x2, a_y2 = coords * 300
    c_x = (a_x1 + a_x2) / 2
    c_y = (a_y1 + a_y2) / 2
    if (c_x >= x1 and c_x <= x2) and (c_y >= y1 and c_y <= y2):
        return True
    return False

def get_frame_rate(video_id):
    cap = cv2.VideoCapture(f'{current_path}/dataset/{video_id}.mkv')
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

if __name__ == "__main__":
    split_into_frames('HV0H6oc4Kvs', True)