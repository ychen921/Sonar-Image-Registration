import os
import cv2
import numpy as np
from tqdm import tqdm

clip_path = r"/home/ychen921/808E/final_project/Dataset/videos/video1.avi"
output_path = r"/home/ychen921/808E/final_project/Dataset/Set1"

ptsa_1 = (0, 499)
ptsa_2 = (0, 194)
ptsa_3 = (175,499)

ptsb_1 = (499, 193)
ptsb_2 = (323, 499)
ptsb_3 = (499,499)

triangle_pts1 = np.array([ptsa_1, ptsa_2, ptsa_3], np.int32)
triangle_pts2 = np.array([ptsb_1, ptsb_2, ptsb_3], np.int32)

def set_color(img, pts):
    pts = pts.reshape((-1, 1, 2))
    triangle_color = (74, 74, 74)
    cv2.fillPoly(img, [pts], triangle_color)
    return img

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
        gray_frame = gray_frame[400:900,710:1210]
        
        gray_frame = set_color(img=gray_frame, pts=triangle_pts1)
        gray_frame = set_color(img=gray_frame, pts=triangle_pts2)

        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('video', gray_frame)
        cv2.waitKey(1)

        frame_filename = os.path.join(output_path, f"{frame_count}.png")
        cv2.imwrite(frame_filename, gray_frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()