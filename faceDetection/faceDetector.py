'''
Face Detection Model
Uses OpenCV's DNN (Deep Neural Network) module and the DNN face detector caffe model which can be found here:
https://github.com/opencv/opencv/tree/master/samples/dnn

'''

import os, cv2
import numpy as np

path = os.getcwd()
path = os.path.join(path, 'faceDetection/model')

class FaceDetection():
    def __init__(self, frame, threshold: float = 0.25, model: str = 'res10_300x300_ssd_iter_140000.caffemodel'):
        self.frame = frame
        self.face_detector = cv2.dnn.readNetFromCaffe(f"{path}/deploy.prototxt.txt", f"{path}/{model}")
        self.threshold = threshold

    def detect(self):
        # Converts frame to blob and sends it to the model
        blob = cv2.dnn.blobFromImage(self.frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_detector.setInput(blob)

        # Performs forward pass on the frame and returns array of detected faces
        # returns array in shape (1, 1, num_faces_detected, 7)
        # where we are interested in last 7 = [_, _, confidence_score, x1, y1, x2, y2]
        # coordinates are normalised to frame size 
        faces = self.face_detector.forward()
        num_faces = faces.shape[2]

        # Removes faces detected with low confidence values
        # Adds the high confidence faces to a new array conf_faces
        conf_faces = []
        for i in range(num_faces):
            confidence = faces[0, 0, i, 2]

            if confidence > self.threshold:     
                conf_faces.append(faces[0, 0, i, :])

        return np.array(conf_faces)