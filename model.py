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
            # chromatic = self.chromatic_vals(face_region)
            lip_pixels = self.define_lip_pixels(face_region)
            tools.plot_frame(lip_pixels)
            tools.plot_frame(face_region)


    def chromatic_vals(self, face_img):
        chromatic = np.zeros((face_img.shape[0], face_img.shape[1], 2)).astype(dtype=np.float64)
        for r in range(len(face_img)):
            for c in range(len(face_img[r])):
                B, G, R  = face_img[r][c].astype(dtype=np.float64)
                den = R + G + B
                if den == 0:
                    chromatic[r][c] = (0, 0)
                else:
                    chromatic[r][c] = (R/den, G/den)

        return chromatic

    def define_lip_pixels(self, face_img):
        lip_pixels = np.zeros(face_img.shape)
        for row in range(len(face_img)):
            for col in range(len(face_img[row])):
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                R, G, B = face_img[row][col].astype(dtype=np.float64)
                I = B + G + R

                if I == 0:
                    r = 0
                    g = 0
                else:
                    r = R / I
                    g = G / I

                discriminant = -0.776 * (r**2) + (0.5601 * r) + 0.2123
                r_lower = -0.776 * (r**2) + (0.5601 * r) + 0.1766
                r_upper =  -1.3767 * (r**2) + (1.0743 * r) + 0.1452

                if g >= r_lower and g <= discriminant:
                    if R >= 10 and B >= 10 and G >= 10:
                        lip_pixels[row][col] = (255,255,255)

        return lip_pixels
