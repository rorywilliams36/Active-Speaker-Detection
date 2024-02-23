import cv2, dlib
import pandas as pd
import numpy as np

from imutils import face_utils
from faceDetection.faceDetector import FaceDetection
from utils import tools

landmarks = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

class ActiveSpeaker():
    def __init__(self, frame):
        self.frame = frame.numpy()
        self.prev_centre_lip = (0,0)

    def model(self):
        # for frame in enumerate(train_features):
        face_detect = FaceDetection(self.frame)
        faces = face_detect.detect()

        face_region, lip_pixels = self.lip_detection(faces)
        # tools.plot_faces_detected(self.frame, faces)

        # Set lip pixels values 
        if len(lip_pixels) > 0:
            centre_upper = lip_pixels[3]
            centre_lower = lip_pixels[9]
            left = lip_pixels[0]
            right = lip_pixels[6]

        # Sets bounding box for lips
        lip_box = [left[0]-5,centre_upper[1]-5, right[0]+5, centre_lower[1]+5]
        lip_area = face_region[lip_box[0]:lip_box[2], lip_box[1]:lip_box[3]]

        # tools.plot_box(face_region, lip_area)

        return faces

    def speaker_detection(self, face_region, lip_pixels, lip_area):
        pass

    def kalman(self):
        pass

    # Gets lips using dlib shape predictor model
    def lip_detection(self, faces):
        h, w = self.frame.shape[:2]
        H = 64
        for face in faces:
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
  
            points = landmarks(gray, dlib.rectangle(0, 0, H, H))
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
