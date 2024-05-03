import torch
import torch.nn as nn
from Mics.metrics import NCC
from Networks.AIRNet import AIRNet
from tqdm import tqdm

class Solver(object):
    def __init__(self, DataLoader, epochs, learning_rate=1e-3, device=torch.device('cpu')):
        self.model = AIRNet(kernels=16)
        self.DataLoader = DataLoader
        self.epochs = epochs
        self.lr = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, amsgrad=True)
        self.loss_func = NCC()
        self.device = device
        
    def train(self):
        
        loss_values = []
        for epoch in range(self.epochs):
            loss_epoch = []

            for i, (fix_img, mov_img) in enumerate(tqdm(self.DataLoader)):
                fix_img = (fix_img/255.0).to(self.device)
                mov_img = (mov_img/255.0).to(self.device)

                self.optimizer.zero_grad()

                wraped, AffineParams = self.model(fix_img, mov_img)

                loss = self.metric(fix_img, wraped)

                loss.backward()

                self.optimizer.step()