import cv2, dlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from imutils import face_utils
from faceDetection.faceDetector import FaceDetection
from utils import tools

landmarks = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

class SdActiveSpeaker():
    def __init__(self, frame, face_thresh: float = 0.25, angle_thresh: tuple = (0.73, 0.71), prev_angles: list = [], prev_labels: list = []):
        self.frame = frame.numpy()
        self.prev_centre_lip = (0, 0)
        self.prev_angles = prev_angles
        self.prev_labels = prev_labels
        self.face_thresh = face_thresh
        self.angle_thresh = angle_thresh

    def model(self):
        # print(self.prev_angles)
        face_detect = FaceDetection(self.frame, threshold=self.face_thresh)
        faces = face_detect.detect()
        predicted = {'faces' : [], 'label' : []}
        sds = []
        for face in faces:
            face_region, lip_pixels = self.feature_detection(face)

            # Set lip pixels values
            if len(lip_pixels) > 0 and len(face_region) > 0:
                centre_upper = lip_pixels[3]
                centre_lower = lip_pixels[9]
                left = lip_pixels[0]
                right = lip_pixels[6]

                # Sets bounding box for lips
                lip_box = [left[0]-1, centre_upper[1]-2, right[0]+1, centre_lower[1]+2]

                # Get area of lips from face
                lip_region = face_region[lip_box[1]:lip_box[3], lip_box[0]:lip_box[2]]

                b,g,r = cv2.split(lip_region)
                hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
                hist_b= cv2.calcHist([b], [0], None, [256], [0, 256])
                hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
                hist = hist_b + hist_g + hist_r

                sd = np.std(hist)

            return sd
        return None

    # Gets lips using dlib shape predictor model
    def feature_detection(self, face):
        h, w = self.frame.shape[:2]
        H = 64
        # Gets coordinates of bounding box
        x1, y1, x2, y2 = face[3:7] * h

        # Grabs extra pixels around box to account for errors and also check ranges
        x1 = max(round(float(x1))-5, 0)
        y1 = max(round(float(y1))-5, 0)
        x2 = min(round(float(x2))+5, w)
        y2 = min(round(float(y2))+5, h)

        # Extracts and resizes the face detected from the frame
        face_region = cv2.resize(self.frame[y1:y2, x1:x2], (H,H))
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        points = landmarks(gray, dlib.rectangle(3, 3, H-3, H-3))
        points = face_utils.shape_to_np(points)

        try:
            return face_region, points[48:-1]
        except:
            return [], []

    def speaker_detection(self, face_region, lip_pixels, lip_box):
        score = 0
        centre_lip = ((lip_box[0] + lip_box[2])/2, (lip_box[1] + lip_box[-1])/2)
        # centre_lip2 = (round(centre_lip[0]), round(centre_lip[1]))

        centre_upper = lip_pixels[3]
        centre_lower = lip_pixels[9]
        left = lip_pixels[0]
        right = lip_pixels[6]

        #left_to_bot = self.mouth_angle(centre_lower, left, centre_lip)
        left_angle = self.mouth_angle(centre_lower, left, centre_lip) + self.mouth_angle(centre_upper, left, centre_lip)
        right_angle = self.mouth_angle(centre_lower, right, centre_lip) + self.mouth_angle(centre_upper, right, centre_lip)

        return left_angle, right_angle

    def mouth_angle(self, point1, point2, centre_point):
        '''
        Args: 
            Point1: first point (generally centre of upper/lower lip)
            Point2: Second coordinate (generally either far left/right point)
            Centre_point: Centre pixel of the lip

        Returns: Angle of mouth opening from the given points (rads)
        '''

        # Find the opposite and hypotenuse
        opposite = np.linalg.norm(point1 - centre_point)
        hypotenuse = np.linalg.norm(point2 - centre_point)

        # Get angle
        if opposite <= hypotenuse and opposite > 0:
            return np.arcsin(opposite / hypotenuse)
        return 0