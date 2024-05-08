import cv2, dlib
import pandas as pd
import numpy as np
from imutils import face_utils

from faceDetection.faceDetector import FaceDetection
from utils import tools
from utils.misc import *

landmarks = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

class ActiveSpeaker():
    def __init__(self, frame,
                prev_frames: dict = {'Frame' : [], 'Faces' : []}, svm: bool = False):
        self.frame = frame.numpy()
        self.prev_frames = prev_frames
        self.svm = svm

    def model(self):
        '''
        Main function to acquire and return features for every face detected in the frame

        returns:
            Dictionary containing the bounding box of face detected, mean optic flow for each face and empty label array
        '''
        face_detect = FaceDetection(self.frame)
        faces = face_detect.detect()
        img_diff = []

        predicted = {'Faces' : [], 'Flow' : [], 'Label' : []}

        # Gets and stores the flow vector for every face detected
        for face in faces:
            flow_vector = self.feature_detection(face)    
            predicted['Faces'].append(face[3:7])
            predicted['Flow'].append(flow_vector)

        return predicted

    # Gets lips using dlib shape predictor model
    def feature_detection(self, face):
        '''
        Function for feature extraction

        Args:
            face: array (7) containing the bounding box for the face detected

        return:
            flow_vector: Mean Optical flow calculated for the given face
        '''

        # Extracts and resizes the face detected from the frame
        h, w = self.frame.shape[:2]
        H = 128
        x1,y1,x2,y2 = get_face_coords(face, h, w)

        # Makes sure face detected is in the frame
        if x1 <= w and y1 <= h and x2 <= w and y1 <= h:
            face_region = cv2.resize(self.frame[y1:y2, x1:x2], (H,H))
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

            # Apply dlib landmarks to face
            points = landmarks(self.frame, dlib.rectangle(x1, y1, x2, y2))
            points = face_utils.shape_to_np(points)

            # Acquires optical flow vector
            if len(points) > 0:
                flow_vector = self.dense_optic_flow(face, face_region)

        try:
            return flow_vector
        except:
            return None


    def dense_optic_flow(self, face, face_region):
        '''
        Function to calculate Gunnar-Farneback Optical flow
        Iterates through previous frames and computes mean flow for every corresponding face in previous frames

        Args:
            face: bounding box for the current face
            face_region: OpenCV image for the current face

        Return:
            mean_flow: Array of mean flow computed from the previous frames/faces
        '''

        H = 128
        prev_face = None
        flows_hori = []
        flows_vert = []
        rates = [1, 0.7, 0.5, 0.2, 0.1]
        all_flows = []

        num_previous = len(self.prev_frames['Frame'])
        if num_previous > 0:
            rate_index = num_previous-1
            for i in range(num_previous):
                x1, y1, x2, y2 = get_face_coords(face, 300, 300)
                prev_frame = self.prev_frames['Frame'][i]
                prev_faces = self.prev_frames['Faces' ][i]

                # Checks if the faces used are in the previous frames
                if len(prev_faces) > 0:
                    for face in prev_faces:
                        p_face = get_face_coords(face, 300, 300)
                        # Change to percent bounding box overlapping
                        if check_face([x1,y1,x2,y2], p_face):
                            x1,y1,x2,y2 = p_face
                            break

                # If no face is detected use the same coordinates from before
                prev_face = cv2.resize(prev_frame[y1:y2, x1:x2], (H,H))

                prev_face = cv2.cvtColor(prev_face, cv2.COLOR_BGR2GRAY)
                current_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

                # Gets dense optic flow values
                flow = cv2.calcOpticalFlowFarneback(prev_face, current_face, None, pyr_scale=0.5, levels=1, 
                                                    winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


                all_flows.append(flow)
                flow_vertical = flow[..., 1]
                flow_hori = flow[..., 0]
                flows_hori.append(flow_hori)
                flows_vert.append(flow_vertical)

            
            if self.svm:
                hori_mean = np.mean(flow_hori, axis=1)
                vert_mean = np.mean(flow_vertical, axis=0)
                mean_flow = np.concatenate((hori_mean, vert_mean), axis=None)

            else:
                mean_flow = np.mean(all_flows, axis=0)
                       
            return mean_flow
        return None