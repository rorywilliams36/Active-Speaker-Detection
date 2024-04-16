import cv2, dlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from imutils import face_utils
from faceDetection.faceDetector import FaceDetection
from utils import tools

landmarks = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

class ActiveSpeaker():
    def __init__(self, frame,
                prev_frames: dict = {'Frame' : [], 'Faces' : []}):
        self.frame = frame.numpy()
        self.prev_frames = prev_frames
        self.face_thresh = face_thresh
        self.angle_thresh = angle_thresh

    def model(self):
        face_detect = FaceDetection(self.frame, threshold=self.face_thresh)
        faces = face_detect.detect()
        img_diff = []

        predicted = {'Faces' : [], 'Flow' : [], 'Label' : []}
        for face in faces:
            flow_vector = self.feature_detection(face)    
            predicted['Faces'].append(face[3:7])
            predicted['Flow'].append(flow_vector)

        return predicted

    # Gets lips using dlib shape predictor model
    def feature_detection(self, face):
        h, w = self.frame.shape[:2]
        H = 64
        x1,y1,x2,y2 = self.get_face_coords(face, h, w)

        # Extracts and resizes the face detected from the frame
        face_region = cv2.resize(self.frame[y1:y2, x1:x2], (H,H))
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        points = landmarks(self.frame, dlib.rectangle(x1, y1, x2, y2))
        points = face_utils.shape_to_np(points)

        if len(points) > 0:
            flow_vector = self.dense_optic_flow(face, face_region)

        try:
            return flow_vector
        except:
            return []

    def dense_optic_flow(self, face, face_region):
        H = 64
        prev_face = None
        if len(self.prev_frames['Frame']) > 0:
            x1, y1, x2, y2 = self.get_face_coords(face, 300, 300)
            prev_frame = self.prev_frames['Frame']
            prev_faces = self.prev_frames['Faces' ]

            # Checks if the faces used are in the previous frames
            if len(prev_faces) > 0:
                for face in prev_faces:
                    p_face = self.get_face_coords(face, 300, 300)
                    if self.check_face([x1,y1,x2,y2], p_face): # and self.check_centres([x1,y1,x2,y2], p_face):
                        x1,y1,x2,y2 = p_face
                        break

            # If no face is detected use the same coordinates from before
            prev_face = cv2.resize(prev_frame[y1:y2, x1:x2], (H,H))

            prev_face = cv2.cvtColor(prev_face, cv2.COLOR_BGR2GRAY)
            current_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

            # Gets dense optic flow values
            flow = cv2.calcOpticalFlowFarneback(prev_face, current_face, None, pyr_scale=0.5, levels=1, 
                                                winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

            # Gets and finds the average of the vertical compoments of optic flow
            flow_vertical = flow[..., 1]
            flow_hori = flow[..., 0]
            hori_mean = np.mean(flow_hori, axis=1)
            vert_mean = np.mean(flow_vertical, axis=0)
            flow_vector = np.concatenate((hori_mean, vert_mean), axis=None)


            ### Alternative Methods
            # Gets magnitude and anglular values for the flow values
            # mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

            # Gets average magnitude and angular values per row from flow vector
            # mag_mean1 = np.mean(mag, axis=1).flatten()
            # ang_mean = np.mean(ang, axis=1).flatten()
            # hori_vector = np.concatenate((mag_mean, ang_mean), axis=None)

            # Gets vertical averages
            # mag_mean2 = np.mean(mag, axis=0).flatten()
            # ang_mean = np.mean(ang, axis=0).flatten()
            # vert_vector = np.concatenate((mag_mean, ang_mean), axis=None)

            # flow_vector = np.concatenate((hori_vector, vert_vector), axis=None)
            # flow_vector = np.concatenate((mag_mean1, mag_mean2))

            return flow_vector
        return []

    def get_face_coords(self, face, h, w):
        # Gets coordinates of bounding box
        if len(face) > 4:
            x1, y1, x2, y2 = face[3:7] * h
        else:
            x1, y1, x2, y2 = face * h

        # Grabs extra pixels around bounding box to account for errors and also check ranges
        x1 = max(round(float(x1))-5, 0)
        y1 = max(round(float(y1))-5, 0)
        x2 = min(round(float(x2))+5, w)
        y2 = min(round(float(y2))+5, h)
        return x1, y1, x2, y2

    def check_face(self, current, previous):
        x1,y1,x2,y2 = current
        p_x1, p_y1, p_x2, p_y2 = previous
        return (x1 <= p_x2) and (x2 >= p_x1) and (y1 <= p_y2) and (y2 >= p_y1)
