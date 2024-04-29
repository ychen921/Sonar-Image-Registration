import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import cv2
import os


class SonarPairDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.dir = data_folder
        self.image_names = [filename for filename in os.listdir(self.dir) if filename.endwith('.png')]
        self.num_images = len(self.image_names)

    def __len__(self):
        return self.num_images-1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx2 = random.randint(0, self.num_images-1)

        img_fixed = os.path.join(self.dir, self.image_names[idx])
        img_moved = os.path.join(self.root_dir, self.image_filenames[idx2])

        
