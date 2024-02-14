import cv2
import pandas as pd
import numpy as np

from faceDetection.faceDetector import FaceDetection
from utils import tools

class ActiveSpeaker():
    def __init__(self, frame):
        self.frame = frame.numpy()

    def model(self):
        # for frame in enumerate(train_features):
        face_detect = FaceDetection(self.frame)
        faces = face_detect.detect()

        self.lip_detection(faces)
        # tools.plot_faces_detected(self.frame, faces)
        

        return faces

 
    def lip_detection(self, faces):
        h, w = self.frame.shape[:2]
        for face in faces:
            # Gets coordinates of bounding box
            x1, y1, x2, y2 = face[3:7] * h

            # Grabs extra pixels around box to account for errors and also check ranges
            x1 = max(round(float(x1))-5, 0)
            y1 = max(round(float(y1))-5, 0)
            x2 = min(round(float(x2))+5, w)
            y2 = min(round(float(y2))+5, h)

            # Extracts and resizes the face detected from the frame
            face_region = cv2.resize(self.frame[y1:y2, x1:x2], (64,64))
            # tools.plot_frame(face_region)


