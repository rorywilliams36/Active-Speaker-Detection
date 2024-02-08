import cv2
import os

video_id = '_mAfwH6i90E'
current_path = os.getcwd()
output = f'/dataset/{video_id}'
cap = cv2.VideoCapture(f'{current_path}/dataset/{video_id}.mkv')

if not os.path.exists(f'{current_path}/{output}'):
    os.makedirs(f'{current_path}/{output}')

while True:
    cap.set(cv2.CAP_PROP_FPS, 20)
    ret, frame = cap.read()
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)	

    if ret:
        if timestamp >= 900000 and timestamp < 912000:
            norm_time = round((timestamp / 1000), 2) # Converts milliseconds to seconds ot match the timestamps in the labels
            cv2.imwrite(f'{current_path}/{output}/{video_id}_{norm_time}.jpg', frame)
                
        if timestamp > 912000:
            break

    else:
        print('Error occured whilst prepping video')
        break

cap.release()