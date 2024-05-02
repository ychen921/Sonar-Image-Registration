import torch
import torch.nn as nn
import torch.optim as optim

class Solver(object):
    def __init__(self, model, epochs, loss_func, learning_rate=1e-3):
        self.model = model
        self.epochs = epochs
        self.lr = learning_rate
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
    def train(self):
        pass