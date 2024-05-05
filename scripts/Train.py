import torch
import torch.nn
from torchsummary import summary
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms as T

import os
import sys
import argparse
from Solver import Solver
from Networks.AIRNet_v2 import AIRNet_v2
from Networks.AIRNet import AIRNet
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
    Parser.add_argument('--DataPath', default="/home/ychen921/808E/final_project/Dataset/Train", 
                        help='Base path of images, Default:/home/ychen921/808E/final_project/Dataset/Train')
    Parser.add_argument('--NumEpochs', type=int, default=10, 
                        help='Number of Epochs to Train for, Default:10')
    Parser.add_argument('--MiniBatchSize', type=int, default=32, 
                        help='Size of the MiniBatch to use, Default:32')

    Args = Parser.parse_args()
    DataPath = Args.DataPath
    NumEpochs = Args.NumEpochs
    MiniBatchSize = Args.MiniBatchSize

    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to 128x128
        transforms.ToTensor()            # Convert images to tensors
    ])

    # Initialize Data Loader
    SonarPair = SonarPairDataset(data_folder=DataPath, transform=transform)
    data_loader = DataLoader(SonarPair, batch_size=MiniBatchSize, shuffle=False)
    
    model = AIRNet().to(device)
    # model = AIRNet_v2().to(device)
    # summary(model, (1, 128, 128))

    # Train the model
    solver = Solver(model=model, DataLoader=data_loader, epochs=NumEpochs, learning_rate=1e-3, device=device)
    solver.train()

if __name__ == '__main__':
    main()