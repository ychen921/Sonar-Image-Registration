import os
import re
import torch
from PIL import Image
from torch.utils.data import Dataset


class SonarPairDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.dir = data_folder
        self.transform = transform
        self.image_names = self.sort_filenames_by_number([filename for filename in os.listdir(self.dir) if filename.endswith('.png')])
        self.num_images = len(self.image_names)
       
    def __len__(self):
        return self.num_images-1

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        current_img_name = self.image_names[idx]

        fixed_img_name = os.path.join(self.dir, current_img_name)
        moving_img_name = os.path.join(self.dir, self.image_names[idx+1])
        # print(fixed_img_name, moving_img_name)
        
        fixed_img = Image.open(fixed_img_name)
        moving_img = Image.open(moving_img_name)

        if self.transform:
            fixed_img = self.transform(fixed_img)
            moving_img = self.transform(moving_img)

        return fixed_img, moving_img
    
    def extract_number(self, filename):
        # Use regular expression to extract the number part of the filename
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        return None

    def sort_filenames_by_number(self, filenames):
        # Sort filenames based on the extracted number using the `extract_number` function
        return sorted(filenames, key=self.extract_number)

        
