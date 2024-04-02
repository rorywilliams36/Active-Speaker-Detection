import cv2, dlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from imutils import face_utils
from faceDetection.faceDetector import FaceDetection
from utils import tools

landmarks = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

class ActiveSpeaker():
    def __init__(self, frame, face_thresh: float = 0.5, angle_thresh: tuple = (0.78, 0.74), prev_frames: list = []):
        self.frame = frame.numpy()
        self.prev_frames = prev_frames
        self.face_thresh = face_thresh
        self.angle_thresh = angle_thresh

    def model(self):
        face_detect = FaceDetection(self.frame, threshold=self.face_thresh)
        faces = face_detect.detect()
        img_diff = []

        predicted = {'faces' : [], 'label' : []}
        for face in faces:
            face_region, lip_pixels, img_diff = self.feature_detection(face)

            # Set lip pixels values
            speaking = 'NOT_SPEAKING' 
            if len(lip_pixels) > 0 and len(face_region) > 0:
                centre_upper = lip_pixels[3]
                centre_lower = lip_pixels[9]
                left = lip_pixels[0]
                right = lip_pixels[6]

                # Sets bounding box for lips
                lip_box = [left[0]-2, centre_upper[1]-5, right[0]+2, centre_lower[1]+5]
                lip_region = self.frame[lip_box[1]:lip_box[3], lip_box[0]:lip_box[2]]

                speaking = self.speaker_detection(face_region, lip_pixels, lip_box)
                prev_frames = self.update_stacks(self.prev_frames, self.frame, 3)

                predicted['faces'].append(face[3:7])
                predicted['label'].append(speaking)
        
        return predicted, self.prev_frames, img_diff

    def speaker_detection(self, face_region, lip_pixels, lip_box):
        score = 0
        centre_lip = ((lip_box[0] + lip_box[2])/2, (lip_box[1] + lip_box[-1])/2)

        centre_upper = lip_pixels[3]
        centre_lower = lip_pixels[9]
        left = lip_pixels[0]
        right = lip_pixels[6]

        left_angle = self.mouth_angle(centre_lower, left, centre_lip) + self.mouth_angle(centre_upper, left, centre_lip)
        right_angle = self.mouth_angle(centre_lower, right, centre_lip) + self.mouth_angle(centre_upper, right, centre_lip)

        # print(left_angle, right_angle)
        if left_angle > self.angle_thresh[0] or right_angle > self.angle_thresh[1]:
            return 'SPEAKING'
        return 'NOT_SPEAKING'

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

    # Contains the values/items from the last 10 frames
    # Updates stack by adding new items and removing if greater than 10
    def update_stacks(self, stack, item, pointer: int = 10 ):
        if len(stack) > pointer:
            _ = stack.pop(0)
        stack = stack.append(item)
        return stack

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
            diff = self.sparse_optic_flow(points, face_region)
            self.dense_optic_flow(face, face_region)
            # if len(diff) > 0:
                # Gets the maximum vertical movement
                # max_vert = max(abs(diff[:][:, 1]))
                # If video cuts or moves to another speaker
                # if max_vert < 10:
                    # pass


        try:
            return face_region, points[48:-1], diff
        except:
            return [], [], []

    def sparse_optic_flow(self, points, lip_region):
        if len(self.prev_frames) > 0:
            points = points.astype(np.float32)
            prev_frame = self.prev_frames[-1]
            prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            current = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            # Parameters for Lucas Kanade optical flow
            lk_params = dict(
                winSize=(15,15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            )
    
            key_points = np.array([points[0], points[3], points[6], points[9], points[12], points[14], points[16], points[18]]).astype(np.float32)

            flow, status, error = cv2.calcOpticalFlowPyrLK(
                prev_frame, current, points, None, **lk_params
            )

            # print('Prev:', key_points)
            # print('flow:', flow)
            diff = points - flow
            # for i in range(len(diff)):
            #     diff[i][0] /= lip_x
            #     diff[i][1] /= lip_y

            print('diff: ', diff)
            # tools.plot_points(self.frame, key_points)

            return diff
        return []

    def dense_optic_flow(self, face, face_region):
        H = 64
        if len(self.prev_frames) > 0:
            x1, y1, x2, y2 = self.get_face_coords(face, 300, 300)
            points = landmarks(face_region, dlib.rectangle(0,0,64,64))
            points = face_utils.shape_to_np(points)
            mouth_region = points[48:-1]
            if len(points) > 0:
                prev_frame = self.prev_frames[-1]
                prev_face = cv2.resize(self.frame[y1:y2, x1:x2], (H,H))

                ### Change this so that face and frame are stored in the same dict
                prev_face = cv2.cvtColor(prev_face, cv2.COLOR_BGR2GRAY)
                current_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

                # Gets dense optic flow values
                flow = cv2.calcOpticalFlowFarneback(prev_face, current_face, None, pyr_scale=0.5, levels=1, 
                                                    winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

                # Gets magnitude and anglular values for the flow values
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

            else:
                print('speaker change')

    def get_face_coords(self, face, h, w):
        # Gets coordinates of bounding box
        x1, y1, x2, y2 = face[3:7] * h

        # Grabs extra pixels around bounding box to account for errors and also check ranges
        x1 = max(round(float(x1))-5, 0)
        y1 = max(round(float(y1))-5, 0)
        x2 = min(round(float(x2))+5, w)
        y2 = min(round(float(y2))+5, h)
        return x1, y1, x2, y2