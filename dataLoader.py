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

class Train_Loader(Dataset):
    def __init__(self, video_id, root_dir: str = 'train'):
        self.root_dir = root_dir # directory holding the data/frames
        self.data_path = os.path.join(f'{current_path}/dataset', self.root_dir) # path to respective dataset folder
        self.video_id = video_id
        self.labels = self.prep_labels()

    # Returns number of items in dataset
    def __len__(self):
        return len(self.labels)

    # Returns a certain point from dataset
    # Gets a example image from dataset given a index
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        '''Each frame is named after the video_id followed by the timestamp in seconds
        Due to OpenCV's method of tracking the time compared to the dataset's labelling
        We ammend the labels to correspond to the correct file (this is mainly due to different methods of rounding)
        Gets corresponding frame file from the label timestamp '''

        timestamp = self.labels.iloc[index]["Timestamp"]
        if not os.path.exists(f'{self.data_path}/{self.video_id}_{timestamp}.jpg'):
            if os.path.exists(f'{self.data_path}/{self.video_id}_{timestamp - 0.01}.jpg'):
                self.labels.at[index, "Timestamp"] -= 0.01
            elif os.path.exists(f'{self.data_path}/{self.video_id}_{timestamp + 0.01}.jpg'):
                self.labels.at[index, "Timestamp"] += 0.01
            else:
                return None
            
        frame_name = f'{self.video_id}_{self.labels.iloc[index]["Timestamp"]}.jpg'
        frame = cv2.imread(f'dataset/{self.root_dir}/{frame_name}')

        # Transform frame (recaling and Grayscale)
        frame = transform_frame(frame)

        # we can ignore first and last items as they are ids
        label = np.array(self.labels.iloc[index, 2:-1])

        # Return frames and labels as tensors
        return torch.from_numpy(frame), convert_label_to_tensor(label)

    # Since we are not using every frame from the data and only the first x frames
    # Also intrduce columns for easier indexing
    def prep_labels(self):
        labels = pd.read_csv(f'{ALL_TRAIN_LABELS}{self.video_id}-activespeaker.csv').iloc[:400]
        labels.columns = ['Video_ID', 'Timestamp', 'x1', 'y1', 'x2', 'y2', 'label', 'face_track_id']
        return labels


class Val_Loader(Dataset):
    def __init__(self, video_id, root_dir: str = 'test'):
        Train_Loader.__init__()

    # Gets a example image from dataset given a index
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        timestamp = self.labels.iloc[index]["Timestamp"]
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

        return torch.from_numpy(frame)

    def __len__(self):
        return len(self.labels)



# Transforms the image by resizing and turning to grayscale
def transform_frame(frame):
    H = 200
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (H, H))
    return frame

# Covnerts label to tensor
# Since labels contain the coordinates of the face speaking and the actual label
# all values need to be of the same type
def convert_label_to_tensor(label):
    if label[-1] == 'NOT_SPEAKING':
        label[-1]  = 0
    else:
        label[-1]  = 1  

    label = np.array(label, dtype=float)

    return torch.from_numpy(label)

# Displays the frame and label on the image
# Frame must be converted to a numpy array 
def show_labels(frame, label):
    frame = frame.numpy()
    label = label.numpy()
    y_dims, x_dims = frame.shape[:2]
    x1, y1 = (label[0], label[1])
    x2, y2 = (label[2], label[3])
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


if __name__ == "__main__":
    ds = Train_Loader(video_id='_mAfwH6i90E')
    frame, label = ds.__getitem__(198)
    # print(sample)
    # show_labels(frame, label)
    # print(ds.labels)
