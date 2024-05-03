import os
import re
import torch
import random
import sys
from PIL import Image
from torch.utils.data import Dataset

sys.dont_write_bytecode = True

class SonarPairDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.dir = data_folder
        self.transform = transform
        self.image_names = self.sort_filenames_by_number([filename for filename in os.listdir(self.dir) if filename.endswith('.png')])
        self.num_images = len(self.image_names)
        self.shift = 20
       
    def __len__(self):
        return self.num_images-1

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        current_img_name = self.image_names[idx]

        if idx < self.shift:
            idx2 = random.randint(idx+1, idx+self.shift)
        elif idx > self.num_images-self.shift:
            idx2 = random.randint(idx-self.shift, idx-1)
        else:
            while True:
                idx2 = random.randint(idx-self.shift, idx+self.shift)
                if idx != idx2: break

        fixed_img_name = os.path.join(self.dir, current_img_name)
        moving_img_name = os.path.join(self.dir, self.image_names[idx2])
        # print(fixed_img_name, moving_img_name)
        
        fixed_img = Image.open(fixed_img_name)
        moving_img = Image.open(moving_img_name)

        if self.transform:
            fixed_img = self.transform(fixed_img)
            moving_img = self.transform(moving_img)

        return fixed_img, moving_img#, fixed_img_name, moving_img_name
    
    def extract_number(self, filename):
        # Use regular expression to extract the number part of the filename
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        return None

    def sort_filenames_by_number(self, filenames):
        # Sort filenames based on the extracted number using the `extract_number` function
        return sorted(filenames, key=self.extract_number)

        
