import cv2
import pandas as pd
import numpy as np

from faceDetection.faceDetector import FaceDetection
from utils import tools

class ActiveSpeaker():
    def __init__(self, frame):
        self.frame = frame

    def model(self):
        # for frame in enumerate(train_features):
        frame = self.frame.numpy()
        face_detect = FaceDetection(frame)
        faces = face_detect.detect()

        height, width = frame.shape[:2]
        # print(faces)
        tools.plot_faces_detected(frame, faces)
