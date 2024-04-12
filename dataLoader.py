import os, cv2, torch, glob, re
import pandas as pd
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt

from utils import tools

current_path = os.getcwd()
ID_LIST = open(f'{current_path}/dataset/ava_speech_file_names_v1.txt', 'r')
TRAIN_LABELS = f'{current_path}/dataset/ava_activespeaker_train_v1.0/'
TEST_LABELS = f'{current_path}/dataset/ava_activespeaker_test_v1.0/'

class Train_Loader(Dataset):
    def __init__(self, video_id, root_dir: str = 'train'):
        '''
        Args:
            root_dir: directory holding the dataset
            data_path: path to data
            video_id: id of the video being loaded
            frames: list of all frame files as jpgs
            labels: Preprocessed csv file containing all labels

        '''

        self.root_dir = root_dir # directory holding the data/frames
        self.video_id = video_id
        self.data_path = os.path.join(f'{current_path}/dataset/{self.root_dir}', self.video_id) # path to respective dataset folder
        self.frames = sort_frames(self.data_path)
        self.labels = self.prep_labels()

    # Returns number of items in dataset
    def __len__(self):
        return len(self.frames)

    # Returns a certain point from dataset
    # Gets a example image from dataset given a index
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        # Gets indexed frame
        frame = self.frames[index]

        # Gets frame name and timestamp from the file naem
        frame_name = frame.split('/')[-1].split('\\')[-1]
        timestamp = float(frame_name.split('_')[-1].split('.jpg')[0])
        frame = cv2.imread(frame)

        # Gets corresponding label from the timestamp
        # Also remove irrelevant ids/values from label
        p_labels = np.array(self.labels.loc[self.labels['Timestamp'] == timestamp])[:, 1:-1]

        # Get dict of labels
        label = create_labels_dict(p_labels)

        # Transform frame (rescaling and Grayscale)
        frame = transform_frame(frame)

        # Return frames and labels as tensors
        return torch.from_numpy(frame), convert_label_to_tensor(label)

    # Since we are not using every frame from the data and only the first x frames
    # Also intrduce columns for easier indexing
    def prep_labels(self):
        # Gets the last frame recorded
        frame_name = (self.frames[-1]).split('/')[-1].split('\\')[-1]

        # Read csv file and create columns
        labels_df = pd.read_csv(f'{TRAIN_LABELS}{self.video_id}-activespeaker.csv')
        labels_df.columns = ['Video_ID', 'Timestamp', 'x1', 'y1', 'x2', 'y2', 'label', 'face_track_id']
        labels_df = labels_df.replace('SPEAKING_AUDIBLE', 'SPEAKING')
        labels_df = labels_df.replace('SPEAKING_NOT_AUDIBLE', 'SPEAKING')
        labels_df = labels_df.replace('NOT_SPEAKING', '0')
        labels_df = labels_df.replace('SPEAKING', '1')

        return labels_df

    def extract_all(self, all_labels, current_labels, index):
        timestamp = float(current_labels[0])
        pos_labels = np.array(all_labels.loc[all_labels['Timestamp'] == timestamp])
        return pos_labels

class Test_Loader(Dataset):
    def __init__(self, video_id, root_dir: str = 'test'):
        Train_Loader.__init__(self, video_id, root_dir)

    # Returns number of items in dataset
    def __len__(self):
        return len(self.frames)

    # Returns a certain point from dataset
    # Gets a example image from dataset given a index
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        # Gets indexed frame
        frame = self.frames[index]

        # Gets frame name and timestamp from the file naem
        frame_name = frame.split('/')[-1].split('\\')[-1]
        timestamp = float(frame_name.split('_')[-1].split('.jpg')[0])
        frame = cv2.imread(frame)

        # Gets corresponding label from the timestamp
        # Also remove irrelevant ids/values from label
        p_labels = np.array(self.labels.loc[self.labels['Timestamp'] == timestamp])[:, 1:-1]

        # Get dict of labels
        label = create_labels_dict(p_labels)

        # Transform frame (rescaling and Grayscale)
        frame = transform_frame(frame)

        # Return frames and labels as tensors
        return torch.from_numpy(frame), convert_label_to_tensor(label)

        # Since we are not using every frame from the data and only the first x frames
    # Also intrduce columns for easier indexing
    def prep_labels(self):
        # Gets the last frame recorded
        frame_name = (self.frames[-1]).split('/')[-1].split('\\')[-1]

        # Read csv file and create columns
        labels_df = pd.read_csv(f'{TEST_LABELS}{self.video_id}-activespeaker.csv')
        labels_df.columns = ['Video_ID', 'Timestamp', 'x1', 'y1', 'x2', 'y2', 'label', 'face_track_id']
        labels_df = labels_df.replace('SPEAKING_AUDIBLE', 'SPEAKING')
        labels_df = labels_df.replace('SPEAKING_NOT_AUDIBLE', 'SPEAKING')
        labels_df = labels_df.replace('NOT_SPEAKING', '0')
        labels_df = labels_df.replace('SPEAKING', '1')

        return labels_df

# Transforms frame by applying gaussian blur, histogram eq and resizing
def transform_frame(frame):
    H = 300
    frame = cv2.GaussianBlur(frame, (5,5), 2.5)
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    img_hsv[:,:,2] = cv2.equalizeHist(img_hsv[:,:,2])
    frame = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    frame = cv2.resize(frame, (H, H))
    return frame

# Since frames can have multiple labels we convert the labels into a dict for pytorch to handle
def create_labels_dict(labels):
    label_dict = {}
    for label in labels:
        if label[0] not in label_dict:
            label_dict[label[0]] = [label]
        else:
            label_dict[label[0]].append(label)
    
    return label_dict

# Converts each label for the timestampS to tensor
# Since labels contain the coordinates of the face speaking and the actual label
# all values need to be of the same type
def convert_label_to_tensor(label):
    label_tensors = {}
    label_list = list(label.items())
    labels = label_list[0][1]

    for arr in labels:
        label_tensors['timestamp'] = arr[0]
        label_tensors['bnd_box'] = arr[1:-1].astype(dtype=float)
        label_tensors['label'] = arr[-1]

    return label_tensors

# Since file are returned in arbitary order we sort them by timestamp
# Credit: https://stackoverflow.com/questions/4813061/non-alphanumeric-list-order-from-os-listdir/48030307#48030307
def sort_frames(path):
    frames = glob.glob(f"{path}/*.jpg")
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(frames, key=alphanum_key)



def extract_labels(all_labels, current_labels, index):
    timestamp = float(current_labels['timestamp'][index])
    pos_labels = np.array(all_labels.loc[all_labels['Timestamp'] == timestamp])[:, 1:-1]

    if len(pos_labels) > 1:
        return timestamp, pos_labels[:, 1:5], pos_labels[:, -1]

    return current_labels['timestamp'][index], current_labels['bnd_box'][index], current_labels['label'][index]
    
if __name__ == "__main__":
    ds = Train_Loader(video_id='_mAfwH6i90E', root_dir='_mAfwH6i90E')
    # ds = Train_Loader(video_id='AYebXQ8eUkM', root_dir='AYebXQ8eUkM')
    print(ds.frames)
    # for i in range(len(ds.frames)):
    #     _, labels = ds.__getitem__(i)
    #     print(labels)

    # show_labels(frame, label)
    # print(ds.labels)
