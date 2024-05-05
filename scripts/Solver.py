import torch
import torch.nn as nn
import sys

from Mics.utils import plot_loss
from Mics.metrics import NCC
from Networks.AIRNet import AIRNet
from tqdm import tqdm

sys.dont_write_bytecode = True

class Solver(object):
    def __init__(self, DataLoader, epochs, learning_rate=1e-3, device=torch.device('cpu')):
        self.device = device
        self.epochs = epochs
        self.lr = learning_rate
        self.DataLoader = DataLoader
        self.model = AIRNet(kernels=16).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        self.loss_func = NCC()
        
    def train(self):
        
        loss_values = []
        for epoch in range(self.epochs):
            loss_epoch = []
            
            print(f"Epoch {epoch+1}, Learning Rate: {self.optimizer.param_groups[0]['lr']}")

            for i, (fix_img, mov_img) in enumerate(tqdm(self.DataLoader)):
                fix_img = (fix_img/255.0).to(self.device)
                mov_img = (mov_img/255.0).to(self.device)

                # Zero gradients for every batch
                self.optimizer.zero_grad()

                # Make predictions for this batch
                wraped, _ = self.model(fix_img, mov_img)

                # Compute loss and its gradient
                loss = self.loss_func(fix_img, wraped)
                loss_epoch.append(loss.item())

                # Backpropation
                loss.backward()

                # Adjust learning rate
                self.optimizer.step()

            # Update the learning rate
            self.scheduler.step()

            LossThisEpoch = sum(loss_epoch) / len(loss_epoch)
            loss_values.append(LossThisEpoch)
            print('Epoch:{}, NCC Loss:{}\n'.format(epoch+1, LossThisEpoch))

        plot_loss(loss_values)