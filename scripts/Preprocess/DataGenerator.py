import os
import cv2
import random
from tqdm import tqdm


DataPath = r'/home/ychen921/808E/final_project/Dataset/Set2'
SavePath = r'/home/ychen921/808E/final_project/Dataset/Overfit'
IamgeType = '.png'
NumImg = 3
ShiftRng = 40

def DuplicatedCheck(name, name_list):
    if name in name_list:
        return 0, name_list
    else:
        # Add the name to the list if it's not already there
        name_list.append(name)
        return 1, name_list
    
def GetFrameCount():
    files = os.listdir(SavePath)
    max_x = -1
    last_fixed_image_name = None
    for file_name in files:
        # Check if the file name matches the pattern 'fixed_x.png'
        if file_name.startswith('fixed_') and file_name.endswith('.png'):
            try:
                # Extract the value of 'x' from the file name
                x = int(file_name.split('_')[1].split('.')[0])
                
                # Update the maximum 'x' value found
                if x > max_x:
                    max_x = x
                    last_fixed_image_name = file_name
            except ValueError:
                continue  # Skip if extraction of 'x' fails (e.g., invalid file name format)
    frame_count = last_fixed_image_name.split('.')[0].split('_')[1]

    return int(frame_count)
    
        
def main():
    # If SavePath doesn't exist make the path
    if(not (os.path.isdir(SavePath))):
        os.makedirs(SavePath)
        frame_count = 0
    else:
        frame_count = GetFrameCount()
    print('Start saving from index {}'.format(frame_count))

    
    # Sort image names
    filenames = []
    for filename in os.listdir(DataPath):
        if filename.endswith('.png'):
            filenames.append(int(filename.split('.')[0]))
    
    filenames = sorted(filenames)
    NumImage = len(filenames)

    for img_name in tqdm(filenames):
        img_path = os.path.join(DataPath, f"{str(img_name)}.png")

        ImageFix = cv2.imread(img_path)

        # Randomly generate image pairs for an image frame
        cnt = 0
        check_list = []
        while cnt < NumImg:
            
            if img_name < ShiftRng:
                mov_name = random.randint(img_name+20, img_name+ShiftRng)
                
            elif img_name > (NumImage-1) - ShiftRng:
                mov_name = random.randint(img_name-ShiftRng, img_name-20)
            
            else:
                RND = random.randint(0, 1)
                if RND == 0:
                    mov_name = random.randint(img_name-ShiftRng, img_name-20)
                else:
                    mov_name = random.randint(img_name+20, img_name+ShiftRng)

            flag, check_list = DuplicatedCheck(mov_name, check_list)

            # Save image pair if not duolicated
            if flag == 1:
                ImageMov = cv2.imread(os.path.join(DataPath, f"{str(mov_name)}.png"))
                # print(ImageMov)

                fixed_path = os.path.join(SavePath, f"fixed_{frame_count}.png") # fixed image path
                moving_path = os.path.join(SavePath, f"moving_{frame_count}.png") # moving image path
                
                cv2.imwrite(fixed_path, ImageFix)
                cv2.imwrite(moving_path, ImageMov)

                frame_count += 1
                cnt += 1


if __name__ == '__main__':
    main()