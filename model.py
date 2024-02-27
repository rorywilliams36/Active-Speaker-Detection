import cv2, dlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from imutils import face_utils
from faceDetection.faceDetector import FaceDetection
from utils import tools

landmarks = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

class ActiveSpeaker():
    def __init__(self, frame, face_thresh: float = 0.25, angle_thresh: float = 0.315):
        self.frame = frame.numpy()
        self.prev_centre_lip = (0, 0)
        self.face_thresh = face_thresh
        self.angle_thresh = angle_thresh

    def model(self):
        # for frame in enumerate(train_features):
        face_detect = FaceDetection(self.frame, threshold=self.face_thresh)
        faces = face_detect.detect()
        predicted = {'faces' : [], 'label' : []}
        for face in faces:
            face_region, lip_pixels = self.feature_detection(face)
            # tools.plot_faces_detected(self.frame, faces)

            # Set lip pixels values
            speaking = 'NOT_SPEAKING' 
            if len(lip_pixels) > 0 and len(face_region) > 0:
                centre_upper = lip_pixels[3]
                centre_lower = lip_pixels[9]
                left = lip_pixels[0]
                right = lip_pixels[6]

                # Sets bounding box for lips
                lip_box = [left[0]-2, centre_upper[1]-2, right[0]+2, centre_lower[1]+2]

                # Get area of lips from face
                lip_region = face_region[lip_box[1]:lip_box[3], lip_box[0]:lip_box[2]]

                speaking = self.speaker_detection(face_region, lip_pixels, lip_region, lip_box)

                # cv2.circle(face_region, (left[0], left[1]), 1, (255,0,0), -1)
                # cv2.circle(face_region, (centre_lower[0], centre_lower[1]), 1, (255,0,0), -1)
                # cv2.circle(face_region, (right[0], right[1]), 1, (255,0,0), -1)
                # cv2.circle(face_region, (centre_upper[0], centre_upper[1]), 1, (255,0,0), -1)
                # tools.plot_frame(face_region)

                predicted['faces'].append(face[3:7])
                predicted['label'].append(speaking)
                
                # tools.plot_box(face_region, lip_box)
                # lips = cv2.resize(lip_region, (256, 128))
                # cv2.imshow('lips', lips)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

        return predicted

    def speaker_detection(self, face_region, lip_pixels, lip_region, lip_box):
        centre_lip = ((lip_box[0] + lip_box[2])/2, (lip_box[1] + lip_box[-1])/2)
        centre_upper = lip_pixels[3]
        centre_lower = lip_pixels[9]
        left = lip_pixels[0]
        right = lip_pixels[6]

        #left_to_bot = self.mouth_angle(centre_lower, left, centre_lip)
        left_angle = self.mouth_angle(centre_lower, left, centre_lip) + self.mouth_angle(centre_upper, left, centre_lip)
        right_angle = self.mouth_angle(centre_lower, right, centre_lip) + self.mouth_angle(centre_upper, right, centre_lip)

        # print(right_angle)
        # print('-----------')
        if left_angle > 0.6 or right_angle > 0.58:
            return 'SPEAKING'
        return 'NOT_SPEAKING'

    def kalman(self):
        pass

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

        # for i in range(48,68): 
        #     # draw the keypoints on the detected faces
        #     cv2.circle(face_region, (points[i][0], points[i][1]), 1, (0, 255, 0), -1)

        # tools.plot_frame(face_region)
        #tools.plot_frame(lip_pixels)
        #tools.plot_color_space(face_region)
        #tools.plot_hist(face_region)

        try:
            return face_region, points[48:-1]
        except:
            return [], []

    # Gets the chromatic colours of an image and stores in array
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

    # Definces lip pixels using Chiang et al's paper
    # doi:10.1016/j.rti.2003.08.003
    # Unfortunatley doesn't work
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
                    if R >= 20 and B >= 20 and G >= 20:
                        lip_pixels[row][col] = (255,255,255)

        return lip_pixels
