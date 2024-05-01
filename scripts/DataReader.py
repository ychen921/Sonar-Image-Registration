import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import os


class SonarPairDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.dir = data_folder
        self.transform = transform
        self.image_names = [filename for filename in os.listdir(self.dir) if filename.endswith('.png')]
        self.num_images = len(self.image_names)
       
    def __len__(self):
        return self.num_images-1

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        current_img_name = self.image_names[idx]

        fixed_img_name = os.path.join(self.dir, current_img_name)
        moving_img_name = os.path.join(self.dir, self.image_names[idx+1])
        
        fixed_img = Image.open(fixed_img_name)
        moving_img = Image.open(moving_img_name)

        if self.transform:
            fixed_img = self.transform(fixed_img)
            moving_img = self.transform(moving_img)

        return fixed_img, moving_img

        
