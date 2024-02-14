
import os, cv2, subprocess
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

current_path = os.getcwd()
current_path = '\\'.join(current_path.split('\\')[:-1])
ALL_TRAIN_LABELS = f'{current_path}/dataset/ava_activespeaker_train_v1.0/'

URL = 'https://s3.amazonaws.com/ava-dataset/trainval/'
train_files = ['B1MAUxpKaV8.mkv', 'CZ2NP8UsPuE.mkv', '55Ihr6uVIDA.mkv', '4gVsDd8PV9U.mp4']

# Converts and saves the video as a series of frames/images
def split_into_frames(video_id):
    labels = pd.read_csv(f'{ALL_TRAIN_LABELS}{video_id}-activespeaker.csv')
    labels.columns = ['Video_ID', 'Timestamp', 'x1', 'y1', 'x2', 'y2', 'label', 'face_track_id']

    output = f'/dataset/{video_id}'
    cap = cv2.VideoCapture(f'{current_path}/dataset/{video_id}.mkv')
    cap.set(cv2.CAP_PROP_FPS, 20)

    if not os.path.exists(f'{current_path}/{output}'):
        os.makedirs(f'{current_path}/{output}')

    count = 0
    while True:
        ret, frame = cap.read()
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)	

        if ret:
        # Labels are only assigned between 900s and 1500s (timestamp is in msec)
        # Converts milliseconds to seconds ot match the timestamps in the labels
            norm_time = round((timestamp / 1000), 2) 
            # Only gets frames with a corresponding label
            if norm_time >= (labels.at[0, "Timestamp"] - 0.04) and count <= 500:
                print(norm_time)
                for i in range(len(labels)):
                    label_timestamp = labels.at[i, "Timestamp"]
                    if label_timestamp > (norm_time + 0.2):
                        break

                    if (label_timestamp == norm_time) or ((label_timestamp >= (norm_time - 0.02)) and (label_timestamp <= (norm_time + 0.02))):
                        cv2.imwrite(f'{current_path}/{output}/{video_id}_{label_timestamp}.jpg', frame)
                        count += 1
                        break
                
            if count > 500:
                break

        else:
            print('Error occured whilst prepping video')
            break

    cap.release()

def download_files(file_urls, output_directory):
    # Checks if output directory exists if not creates it
    path = f'{current_path}/dataset/{output_directory}'
    if not os.path.exists(path):
        os.makedirs(path)

    # Downloads files
    for url in file_urls:
        try:
            command = f"curl -o {path} {url}"
            subprocess.run(command, shell=True, check=True)
            print(f"Downloaded")
        # if error occured
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {url}: {e}")


if __name__ == "__main__":
    # split_into_frames('_mAfwH6i90E')
    split_into_frames('B1MAUxpKaV8')
    # download_files(f'{URL}/{train_files[0]}', 'B1MAUxpKaV8')