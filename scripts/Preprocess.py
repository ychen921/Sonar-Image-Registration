import os
import cv2
import numpy as np
from tqdm import tqdm

clip_path = r"/home/ychen921/808E/final_project/Dataset/videos/video1.avi"
output_path = r"/home/ychen921/808E/final_project/Dataset/Set1"

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
        gray_frame = gray_frame[260:660, 750:1150]
        cv2.imshow('video', gray_frame)
        cv2.waitKey(1)

    # frame_filename = os.path.join(output_path, f"{frame_count}.png")
    # cv2.imwrite(frame_filename, frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()