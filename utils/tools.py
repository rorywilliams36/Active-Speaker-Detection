import cv2, torch
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

'''
Tools file which mainly contains functions for different plots
'''

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

        if speak == 'NOT_SPEAKING':
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
        frame = cv2.resize(frame, (200, 200))

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

    frame = cv2.resize(frame, (400,400))
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

# Plots actual bounding box for frame 
def plot_actual(frame, coords):
    # Plots single bounding box
    if torch.is_tensor(coords):
        x1, y1, x2, y2 = coords[:] * 300
        x1 = round(float(x1))
        y1 = round(float(y1))
        x2 = round(float(x2))
        y2 = round(float(y2)) 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    else:
        # Plots multiple boxes
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
# Shows threshold eq
def plot_chromatic(chromatic):
    r = chromatic[:, :, 0]
    g = chromatic[:, :, 1]
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

# Plots colour histogram of the image
def plot_hist(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.xlabel('intensity')
    plt.ylabel('Occurences')
    plt.show()

# Plots the 3d colour space of the image
def plot_color_space(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb / 255
    B,G,R = cv2.split(img)

    ax = plt.axes(projection='3d')
    ax.scatter(R,G,B, c=img_rgb.reshape((-1, 3)))
    plt.show()

def plot_points(img, points):
    for point in points:
        x = round(point[0])
        y = round(point[1])
        cv2.circle(img, (x, y), 1, color=(0,255,0))
    img = cv2.resize(img, (300,300))
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Plots optical flow representation using current and previous frames
# Credit: https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
def flow_img(flow, face_region, prev_face):
    hsv = np.zeros_like(face_region)
    current_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    flow_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10,5))
    plt.subplot(1,3,1)
    plt.title('Previous Frame')
    plt.axis('off')
    plt.imshow(prev_face, cmap='gray')

    plt.subplot(1,3,2)
    plt.title('Current Frame')
    plt.axis('off')
    plt.imshow(current_face, cmap='gray')

    plt.subplot(1,3,3)
    plt.title('Optic Flow')
    plt.axis('off')
    plt.imshow(flow_img)

    plt.show()

def plot_loss(loss, epochs):
    epoch_list = [i for i in range(epochs)]
    plt.plot(epoch_list, loss)
    plt.title('Loss Function')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show() 


def plot_cross_validation(train_loss, val_loss, epochs):
    epoch_list = [i for i in range(epochs)]
    plt.plot(epoch_list, train_loss, label='Train Loss')
    plt.plot(epoch_list, val_loss, label='Validation Loss')
    plt.title('Loss Function')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show() 

def plot_valid_acc(val_acc, epochs):
    epoch_list = [i for i in range(epochs)]
    plt.plot(epoch_list, val_acc)
    plt.title('Validation Accuracy across Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy %')
    plt.show() 
