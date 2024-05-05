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
        self.image_names = [filename for filename in os.listdir(self.dir) if filename.startswith('fixed_')]
        self.num_images = len(self.image_names)
        
       
    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        fixed_image_name = self.image_names[idx]
        moving_image_name = f"moving_{fixed_image_name.split('_')[1]}"

        fixed_image_path = os.path.join(self.dir, fixed_image_name)
        moving_image_path = os.path.join(self.dir, moving_image_name)

        fixed_image = Image.open(fixed_image_path).convert('L')
        moving_image = Image.open(moving_image_path).convert('L')

        if self.transform:
            fixed_image = self.transform(fixed_image)
            moving_image = self.transform(moving_image)

        return fixed_image, moving_image