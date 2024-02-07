import os
import cv2
import pandas as pd
import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

current_path = os.getcwd()
ID_LIST = open(f'{current_path}/dataset/ava_speech_file_names_v1.txt', 'r')
ALL_TRAIN_LABELS = f'{current_path}/dataset/ava_activespeaker_train_v1.0/'
TEST_LABELS = f'{current_path}/dataset/ava_activespeaker_train_v1.0/'

class Dataset_Loader(Dataset):
    def __init__(self, root_dir, video_id):
        self.root_dir = root_dir # directory holding the data/frames
        self.data_path = os.path.join(f'{current_path}/dataset', self.root_dir) # path to respective dataset folder
        self.video_id = video_id
        self.labels = self.prep_labels()

    # Returns number of items in dataset
    def __len__(self):
        return len(self.labels)

    # Returns a certain point from dataset
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        '''Each frame is named after the video_id followed by the timestamp in seconds
        Due to OpenCV's method of tracking the time compared to the dataset's labelling
        We ammend the labels to correspond to the correct file (this is mainly due to different methods of rounding)
        Gets corresponding frame file from the label timestamp '''

        timestamp = self.labels.iloc[index]["Timestamp"]
        print(self.labels.iloc[index])
        if not os.path.exists(f'{self.data_path}/{self.video_id}_{timestamp}.jpg'):
            if os.path.exists(f'{self.data_path}/{self.video_id}_{timestamp - 0.01}.jpg'):
                self.labels.at[index, "Timestamp"] -= 0.01
            elif os.path.exists(f'{self.data_path}/{self.video_id}_{timestamp + 0.01}.jpg'):
                self.labels.at[index, "Timestamp"] += 0.01
            else:
                return None
            
        frame_name = f'{self.video_id}_{self.labels.iloc[index]["Timestamp"]}.jpg'
        frame = cv2.imread(f'dataset/{self.root_dir}/{frame_name}')
        
        # we can ignore first and last items as they are ids
        label = np.array(self.labels.iloc[index, 2:-1])

        return {'frame' : frame, 'label' : label}

    # Since we are not using every frame from the data and only the first 300 frames
    def prep_labels(self):
        labels = pd.read_csv(f'{ALL_TRAIN_LABELS}{self.video_id}-activespeaker.csv').iloc[:400]
        labels.columns = ['Video_ID', 'Timestamp', 'x1', 'y1', 'x2', 'y2', 'label', 'face_track_id']
        # labels.drop(['face_track_id'], axis=1)
        return labels
    
    def show_labels(self, frame, label):
        y_dims, x_dims = frame.shape[:2]
        print(label)
        x1, y1 = (label[0], label[1])
        x2, y2 = (label[2], label[3])
        speak = label[4]

        x1 = round(float(x1)*x_dims)
        y1 = round(float(y1)*y_dims)
        x2 = round(float(x2)*x_dims)
        y2 = round(float(y2)*y_dims)
        print(x1,y1)
        print(x2,y2)

        if speak == 'NOT_SPEAKING':
            c = (0,0,255)
        else:
            c = (0,255,0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color=c)
        cv2.imshow('frame', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ds = Dataset_Loader('train', '_mAfwH6i90E')
    sample = ds.__getitem__(198)
    # print(sample)
    ds.show_labels(sample['frame'], sample['label'])
    # print(ds.labels)
