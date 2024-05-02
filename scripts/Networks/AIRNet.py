import torch
import torch.nn as nn

class ConvNet(nn.Module):
  def __init__(self, output_dim=10):
    super(ConvNet, self).__init__()
    self.layers = nn.Sequential(

        # Conv layer with batch normalization
        # and maxpool layer.
        nn.Conv2d(3, 1024, kernel_size=7, padding=1),
        nn.ReLU(),
        # nn.BatchNorm2d(1024),
        nn.MaxPool2d(kernel_size=2),

        # Conv layer with batch normalization
        # and maxpool layer.
        nn.Conv2d(1024, 512, kernel_size=5, padding=1),
        nn.ReLU(),
        # nn.BatchNorm2d(512),
        nn.MaxPool2d(kernel_size=2),

        # Conv layer with batch normalization
        # and maxpool layer.
        nn.Conv2d(512, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        # nn.BatchNorm2d(256),
        nn.MaxPool2d(kernel_size=2),

        # Flatten then pass to 3 fully connected
        # layers, each FCL follow with a dropout
        # layer
        nn.Flatten(),
        nn.Linear(2304, 1024),
        nn.ReLU(),
        # nn.Dropout(0.7),

        nn.Linear(1024, 512),
        nn.ReLU(),
        # nn.Dropout(0.7),

        nn.Linear(512, 128),
        nn.ReLU(),
        # nn.Dropout(0.7),

        # Output 10 classes
        nn.Linear(128, output_dim)
        )

  def forward(self, input):
    out = self.layers(input)
    return out



class AIRNet(nn.Module):
    def __init__(self, output_dim=10):
        super().__init__(AIRNet, self)
        
        pass