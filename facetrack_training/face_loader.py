import os, cv2, torch, glob
import pandas as pd
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

current_path = '\\'.join(os.getcwd().split('\\')[:-1])
ALL_TRAIN_LABELS = f'{current_path}/dataset/ava_activespeaker_train_v1.0/'


class Train_Loader(Dataset):
    def __init__(self):
        '''
        Args:
            root_dir: directory holding the dataset
            data_path: path to data
            video_id: id of the video being loaded
            frames: list of all frame files as jpgs
            labels: Preprocessed csv file containing all labels

        '''
        self.data_path = os.path.join(f'{current_path}/dataset/facetracks') # path to respective dataset folder
        self.frames = glob.glob(f"{self.data_path}/*.jpg")
        # self.labels = self.prep_labels()

    # Returns number of items in dataset
    def __len__(self):
        # return len(self.labels.drop_duplicates(subset=['Timestamp']))
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
        face_track_end = int(frame_name.split('_')[-2])
        # vid_id = ''.join(frame_name.split('_')[:-2])
        vid_id = frame_name[:11]

        self.labels = pd.read_csv(f'{ALL_TRAIN_LABELS}{vid_id}-activespeaker.csv')
        self.labels.columns = ['Video_ID', 'Timestamp', 'x1', 'y1', 'x2', 'y2', 'label', 'face_track_id']

        frame = cv2.imread(frame)

        # Gets corresponding label from the timestamp
        # Also remove irrelevant ids/values from label
        p_labels = self.labels.loc[(self.labels['Timestamp'] == timestamp) & (self.labels['Video_ID'] == vid_id)]

        label = ' '
        if len(p_labels) > 1:
            for l in np.array(p_labels):
                face_track_label = l[-1]
                face_track_id = face_track_label.split(':')[-1]
                if face_track_id == face_track_end:
                    label = l[-2]
                    break

        else:
            label = str(p_labels['label'])

        return torch.from_numpy(frame), label


if __name__ == "__main__":
    ds = Train_Loader()
    for i in range(100):
        ds.__getitem__(i)