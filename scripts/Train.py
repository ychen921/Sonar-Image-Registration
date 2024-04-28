import torch
import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as T

import argparse
from tqdm import tqdm
import sys
import os
import time


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

if __name__ == '__main__':
    main()