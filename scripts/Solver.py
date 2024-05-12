import torch
import torch.nn as nn
import sys

from Mics.utils import plot_loss
from Mics.metrics import NCC, dice_score
from tqdm import tqdm

sys.dont_write_bytecode = True

class Solver(object):
    def __init__(self, model, DataLoader, CheckPointPath, epochs, learning_rate=1e-3, 
                 device=torch.device('cpu'), DecayStep=10):
        self.device = device
        self.epochs = epochs
        self.lr = learning_rate
        self.DataLoader = DataLoader
        self.CheckPointPath = CheckPointPath

        self.model = model.to(self.device)

        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=DecayStep, gamma=0.5)
        self.loss_func = NCC()
        
    def train(self):
        
        loss_values = []
        dice_values = []
        for epoch in range(self.epochs):
            loss_epoch = []
            dice_epoch = []
            
            print(f"Epoch {epoch+1}, Learning Rate: {self.optimizer.param_groups[0]['lr']}")

            for i, (fix_img, mov_img) in enumerate(tqdm(self.DataLoader)):
                fix_img = (fix_img/255.0).to(self.device)
                mov_img = (mov_img/255.0).to(self.device)
                
                # Zero gradients for every batch
                self.optimizer.zero_grad()

                # Make predictions for this batch
                wraped, _ = self.model(fix_img, mov_img)
                # wraped = self.model(mov_img, fix_img)

                # Compute loss and its gradient
                loss = self.loss_func(fix_img, wraped)
                dice = dice_score(wraped, fix_img)

                dice_epoch.append(dice.item())
                loss_epoch.append(loss.item())

                # Backpropation
                loss.backward()

                # Adjust learning rate
                self.optimizer.step()

            # Update the learning rate
            self.scheduler.step()

            # Compute NCC and Dice loss of epoch 
            LossThisEpoch = sum(loss_epoch) / len(loss_epoch)
            loss_values.append(LossThisEpoch)

            DiceThisEpoch = sum(dice_epoch) / len(dice_epoch)
            dice_values.append(DiceThisEpoch*1e3)

            print('Epoch:{}, NCC Loss:{}, Dice:{}'.format(epoch+1, LossThisEpoch, DiceThisEpoch*1e3))

            # Save epoch model
            SaveName = self.CheckPointPath + str(epoch) + '_model.pt'
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        }, SaveName)
            print(SaveName + ' Model Saved...\n')

        plot_loss(loss_values, dice_values)