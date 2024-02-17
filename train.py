import os, argparse, cv2, glob
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from dataLoader import Train_Loader, Val_Loader
from model import ActiveSpeaker
from evaulation import eval_face_detection
from utils import tools

path = os.getcwd()
frames = glob.glob(f"{path}/dataset/train/*.jpg")

def main():
    # parser = argparse.ArgumentParser(description = "Training Stage")
    # parser.add_argument('--epoch', type=int, default=15, help="Number of epochs")
    # parser.add_argument('--face_detect_threshold', type=float, default='0.5', help="Confidence score for face detection")
    # parser.add_argument('--face_detect_model', type=str, default='res10_300x300_ssd_iter_140000.caffemodel', help="OpenCV model used for face detection")
    
    # parser.add_argument('--evaluate', type=bool, default=False, help="Perform Evaluation (True/False)")
    # parser.add_argument('--dataset', type=str, default='AVA Active Speaker Dataset', help="Type of dataset to test and train")

    # parser.add_argument('--train_dir', type=str, default='train', help="Data path for the training data")
    # parser.add_argument('--test_dir', type=str, default='test', help="Data path for the testing data")
    # parser.add_argument('--CUDA', type=bool, default=False, help="Whether to run code on GPU is available")

    # args = parser.parse_args()

    # trainLoader = Train_Loader(video_id='_mAfwH6i90E', root_dir='train')
    trainLoader = Train_Loader(video_id='B1MAUxpKaV8', root_dir='B1MAUxpKaV8')
    trainLoaded = DataLoader(trainLoader, batch_size=64, num_workers=0, shuffle=True)

    num = 0
    t_correct = 0
    t_total = 0
    for images, labels in trainLoaded:
        for i in range(len(images)):
            asd = ActiveSpeaker(images[i])
            faces = asd.model()
            actual_label = trainLoader.extract_labels(trainLoader.labels, labels, i)
            correct, total = eval_face_detection(faces, actual_label)
            t_correct += correct
            t_total += total

            
            # print(faces)
            # img = images[i].numpy()
            # for face in faces:
            #     # tools.plot_frame(face)
            #     x1, y1, x2, y2 = face[3:7] * 300

            #     # Grabs extra pixels around box to account for errors and also check ranges
            #     x1 = max(round(float(x1))-5, 0)
            #     y1 = max(round(float(y1))-5, 0)
            #     x2 = min(round(float(x2))+5, 300)
            #     y2 = min(round(float(y2))+5, 300)

            #     # Extracts and resizes the face detected from the frame
            #     face_region = cv2.resize(img[y1:y2, x1:x2], (64,64))
            #     save_facetracks(face_region, actual_label, trainLoader, face, i)

            
            # tools.plot_frame(images[i].numpy())
            # if num % 100 == 0:
            # tools.plot_faces_detected(images[i].numpy(), faces)
            # tools.plot_actual(images[i].numpy(), actual_label[1])

            num += 1
    
    print('Correct Faces Detected: ', t_correct)
    print('Total Number of Faces: ', t_total)
    print('Percent Correct: ', t_correct/t_total *100)




    




if __name__ == "__main__":
    main()