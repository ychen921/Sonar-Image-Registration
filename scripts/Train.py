import torch
import torch.nn
from torchsummary import summary
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms as T

import os
import sys
import argparse
from Mics.utils import FindLatestModel

from Solver import Solver
from Networks.AIRNet import AIRNet
from Networks.InceptionNet import InceptionNet
from Networks.DenseNet import DenseNet
from DataReader.DataReader import SonarPairDataset


# Don't generate pyc codes
sys.dont_write_bytecode = True

dtype = torch.float32

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def pretty_print(NumEpoch, Batchsize, LR, DecayStep):
    print('Training Epoch: {}'.format(NumEpoch))
    print('Training Batch Size: {} '.format(Batchsize))
    print('Initial Learning Rate: {}'.format(LR))
    print('Period of learning rate decay: {} steps'.format(DecayStep))
    print('Using device:', device)


def main():

    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataPath', default="/home/ychen921/808E/final_project/Dataset/Overfit", 
                        help='Base path of images, Default:/home/ychen921/808E/final_project/Dataset/Train')
    Parser.add_argument('--CheckPointPath', default='/home/ychen921/808E/final_project/Checkpoints/', 
                        help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--NumEpochs', type=int, default=50, 
                        help='Number of Epochs to Train for, Default:10')
    Parser.add_argument('--MiniBatchSize', type=int, default=32, 
                        help='Size of the MiniBatch to use, Default:32')
    Parser.add_argument('--LearningRate', type=int, default=1e-3, 
                        help='Size of the MiniBatch to use, Default:0.005')
    Parser.add_argument('--LrDecayStep', type=int, default=10, 
                        help='Period of learning rate decay, Default:15')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0,
                         help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')

    Args = Parser.parse_args()
    DataPath = Args.DataPath
    CheckPointPath = Args.CheckPointPath
    NumEpochs = Args.NumEpochs
    MiniBatchSize = Args.MiniBatchSize
    lr = Args.LearningRate
    DecayStep = Args.LrDecayStep
    LoadCheckPoint = Args.LoadCheckPoint

    # Print training info.
    pretty_print(NumEpochs, MiniBatchSize, lr, DecayStep)
    
    # Create checkpoint directory
    os.makedirs(CheckPointPath, exist_ok=True)

    # Find Latest Checkpoint
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    print('Latest file: ', LatestFile)  

    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to 128x128
        transforms.ToTensor()            # Convert images to tensors
    ])

    # Initialize Data Loader
    SonarPair = SonarPairDataset(data_folder=DataPath, transform=transform)
    data_loader = DataLoader(SonarPair, batch_size=MiniBatchSize, shuffle=False)
    
    # model = AIRNet().to(device)
    # model = InceptionNet().to(device)
    model = DenseNet().to(device)


    # Train the model
    solver = Solver(model=model, DataLoader=data_loader, epochs=NumEpochs,
                     learning_rate=lr, device=device, DecayStep=DecayStep, CheckPointPath=CheckPointPath)
    solver.train()

if __name__ == '__main__':
    main()