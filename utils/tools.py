import cv2, torch
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Displays the frame and given label on the image
# Frame must be converted to a numpy array 
def show_labels(frame, labels):
    y_dims, x_dims = frame.shape[:2]
    for i in range(len(labels)):
        x1, y1 = (labels[i][1], labels[i][1])
        x2, y2 = (labels[i][1], labels[i][1])
        speak = labels[i][1]

        x1 = round(float(x1)*x_dims)
        y1 = round(float(y1)*y_dims)
        x2 = round(float(x2)*x_dims)
        y2 = round(float(y2)*y_dims)

        if speak == 0:
            c = (0,0,255)
        else:
            c = (0,255,0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color=c)

    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Shows frame without label
def plot_frame(frame):
    # Scales frame up if frame is too small
    if frame.shape[0] < 100 or frame.shape[1] < 100:
        frame = cv2.resize(frame, (150, 150))

    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Plots bounding boxes on faces
def plot_faces_detected(frame, faces):
    h, w = frame.shape[:2]

    for face in faces:
        x1, y1, x2, y2 = face[3:7] 
        x1 = round(float(x1)*w)
        y1 = round(float(y1)*h)
        x2 = round(float(x2)*w)
        y2 = round(float(y2)*h) 
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Plots single bounding box from coords
def plot_box(frame, coords):
    x1, y1, x2, y2 = coords[:]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_actual(frame, coords):
    if torch.is_tensor(coords):
        x1, y1, x2, y2 = coords[:] * 300
        x1 = round(float(x1))
        y1 = round(float(y1))
        x2 = round(float(x2))
        y2 = round(float(y2)) 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    else:
        for coord in coords:
            x1, y1, x2, y2 = coord * 300
            x1 = round(float(x1))
            y1 = round(float(y1))
            x2 = round(float(x2))
            y2 = round(float(y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Plots scatter graph showing chromatic values of an image
def plot_chromatic(chromatic):
    r = chromatic[:, :, 0]
    g = chromatic[:, :, 1]
    # print(r)
    x = np.linspace(2, -2, 100)
    r_u =  -1.3767 * (x**2) + (1.0743 * x) + 0.1452
    r_l = -0.776 * (x**2) + (0.5601 * x) + 0.1766
    l_r = -0.776 * (x**2) + (0.5601 * x) + 0.2123
    plt.plot(x, r_u, color='red')
    plt.plot(x, r_l, color='red')
    plt.plot(x, l_r, color='green')
    plt.scatter(r, g)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('red chromatic')
    plt.ylabel('green chromatic')
    plt.show()

def plot_side_by_side(lip, image):
    img = cv2.hconcat(lip, image)
    cv2.imshow('images', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
