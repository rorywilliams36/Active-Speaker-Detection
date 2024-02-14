import os, cv2, torch, glob
import pandas as pd
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

current_path = os.getcwd()
ID_LIST = open(f'{current_path}/dataset/ava_speech_file_names_v1.txt', 'r')
ALL_TRAIN_LABELS = f'{current_path}/dataset/ava_activespeaker_train_v1.0/'
TEST_LABELS = f'{current_path}/dataset/ava_activespeaker_val_v1.0/'

class Train_Loader(Dataset):
    def __init__(self, video_id, root_dir: str = 'train'):
        '''
        root_dir: directory holding the dataset
        data_path: path to data
        video_id: id of the video being loaded
        frames: list of all frame files as jpgs
        labels: Preprocessed csv file containing all labels

        '''
        self.root_dir = root_dir # directory holding the data/frames
        self.data_path = os.path.join(f'{current_path}/dataset', self.root_dir) # path to respective dataset folder
        self.video_id = video_id
        self.frames = glob.glob(f"{self.data_path}/*.jpg")
        self.labels = self.prep_labels()

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
        last = float(frame_name.split('_')[-1].split('.jpg')[0])

        # Read csv file and create columns
        labels_df = pd.read_csv(f'{ALL_TRAIN_LABELS}{self.video_id}-activespeaker.csv')
        labels_df.columns = ['Video_ID', 'Timestamp', 'x1', 'y1', 'x2', 'y2', 'label', 'face_track_id']

        # slice to the last frame recorded
        spliced_labels = labels_df.loc[labels_df['Timestamp'] <= last]

        return spliced_labels
            
    def extract_label(labels, index):
        return labels['timestamp'][index], labels['bnd_box'][index], labels['label'][index]

class Val_Loader(Dataset):
    def __init__(self, video_id, root_dir: str = 'test'):
        Train_Loader.__init__()

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

        # Transform frame (recaling and Grayscale)
        frame = transform_frame(frame)

        # Return frames and labels as tensors
        return torch.from_numpy(frame), label

# Transforms the image by resizing and turning to grayscale
def transform_frame(frame):
    H = 300
    frame = cv2.convertScaleAbs(frame, alpha=1.4, beta=3)
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

# Converts each label for the timestamp to tensor
# Since labels contain the coordinates of the face speaking and the actual label
# all values need to be of the same type
def convert_label_to_tensor(label):
    label_tensors = {}
    label_list = list(label.items())
    labels = label_list[0][1]

    if len(labels) > 1:
        # bnd = np.array([[]])
        # for arr in labels:
        #     bnd = np.append(bnd, np.array(arr[1:-1]))

        # print(bnd)

        for i in range(len(labels)):
            label_tensors['timestamp'] = labels[i][0]

            # if 'bnd_box' in label_tensors:
            #     label_tensors['bnd_box'].append(labels[i][1:-1])
            # else:
            label_tensors['bnd_box'] = labels[i][1:-1].astype(dtype=float)

            if 'label' in label_tensors:
                label_tensors['label'].append(labels[i][-1])
            else:
                label_tensors['label'] = [labels[i][-1]]

    else:    
        for arr in labels:
            label_tensors['timestamp'] = arr[0]
            label_tensors['bnd_box'] = arr[1:-1].astype(dtype=float)
            label_tensors['label'] = arr[-1]


    return label_tensors
    



if __name__ == "__main__":
    # ds = Train_Loader(video_id='_mAfwH6i90E')
    ds = Train_Loader(video_id='B1MAUxpKaV8', root_dir='B1MAUxpKaV8')

    for i in range(len(ds.frames)):
        _, labels = ds.__getitem__(i)
        print(labels)

    # show_labels(frame, label)
    # print(ds.labels)
