import torch
import torch.nn as nn

class AIRNet(nn.Module):
    def __init__(self, 
                 kernal_size=3, 
                 kernals=32, 
                 linear_nodes=64,
                 num_conv_layers=5,
                 num_dense_layers=2,
                 num_downsamplings=4,
                 ndim=2
                 ):
        
        super().__init__()
        
        assert (num_dense_layers >= 1), "Number of dense layers should at least be 1 (excluding the final dense output layer)."
        
        self.ndim = ndim