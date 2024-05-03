import torch
import torch.nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as T

import argparse
from tqdm import tqdm
import sys
import os
import time
from Solver import Solver
from DataReader.DataReader import SonarPairDataset


# Don't generate pyc codes
sys.dont_write_bytecode = True

dtype = torch.float32

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)


def main():

    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataPath', default="/home/ychen921/808E/final_project/Dataset/Set1", 
                        help='Base path of images, Default:/home/ychen921/808E/final_project/Dataset/Set1')
    Parser.add_argument('--NumEpochs', type=int, default=20, 
                        help='Number of Epochs to Train for, Default:20')
    Parser.add_argument('--MiniBatchSize', type=int, default=64, 
                        help='Size of the MiniBatch to use, Default:64')

    Args = Parser.parse_args()
    DataPath = Args.DataPath
    NumEpochs = Args.NumEpochs
    MiniBatchSize = Args.MiniBatchSize

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to 256x256
        transforms.ToTensor()            # Convert images to tensors
    ])

    SonarPair = SonarPairDataset(data_folder=DataPath, transform=transform)
    data_loader = DataLoader(SonarPair, batch_size=MiniBatchSize, shuffle=True)

    solver = Solver(DataLoader=data_loader, epochs=NumEpochs, learning_rate=1e-3, device=device)
    solver.train()

if __name__ == '__main__':
    main()