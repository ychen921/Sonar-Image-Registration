import os
import cv2
import numpy as np
from tqdm import tqdm

clip_path = r"/home/ychen921/808E/final_project/Dataset/videos/video1.avi"
output_path = r"/home/ychen921/808E/final_project/Dataset/Set1"

R = np.arange(0.0025, 3.5025, 0.0025)
Az = np.radians(np.arange(-30, 30, 0.225))

x0 = 959
y0 = 1025
R_bold = 971

X_RES = 267
Y_RES = 1400

R = np.arange(0.0025, 3.5025, 0.0025)
Az = np.radians(np.arange(-30, 30, 0.225))

scale_factor = R_bold / 3.5

def Convert2RA(img):
    r = R[:, np.newaxis]
    theta = Az[np.newaxis, :]
    x = np.rint(x0 + scale_factor * r * np.sin(theta)).astype(int)
    y = np.rint(y0 - scale_factor * r * np.cos(theta)).astype(int)

    # Clip coordinates to image bounds
    x = np.clip(x, 0, img.shape[1] - 1)
    y = np.clip(y, 0, img.shape[0] - 1)

    # Index into img using computed coordinates
    sonar_image = img[y, x]

    # Reshape and post-process the sonar_image
    sonar_image = np.flip(sonar_image, axis=0).astype(np.uint8)
    sonar_image = cv2.resize(sonar_image, (256, 256), interpolation=cv2.INTER_LINEAR)
    
    return sonar_image


def main():
    os.makedirs(output_path, exist_ok=True)
    cap = cv2.VideoCapture(clip_path)

    if not cap.isOpened():
        print(f"Error: Could not open video '{clip_path}'")
        return
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count +=1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        save_frame = Convert2RA(gray_frame)

        cv2.imshow('video', save_frame)
        cv2.waitKey(1)

        frame_filename = os.path.join(output_path, f"{frame_count}.png")
        cv2.imwrite(frame_filename, save_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()