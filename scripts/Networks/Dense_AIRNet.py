import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(Bottleneck, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.inter_channel = channel_out * 4
        
        self.layer = nn.Sequential(
            nn.BatchNorm2d(self.channel_in),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.channel_in,  out_channels=self.inter_channel, 
                               kernel_size=1, stride=1, padding=0, bias=False),
            
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.inter_channel, out_channels=self.channel_out, 
                                   kernel_size=3, stride=1, padding=1, bias=False)
        )
        
    def forward(self, x):
        input_x = x
        x = self.layer(x)
        # Add all togather
        x = torch.cat([input_x, x], 1)
        return x
    
class Dense_Block(nn.Module):
    def __init__(self, in_channel, num_layers, growth_rate):
        super(Dense_Block, self).__init__()
        self.in_channel = in_channel #12
        self.num_layers = num_layers #6
        self.growth_rate = growth_rate #12
        
        self.block = self._make_block(self.in_channel, self.num_layers, self.growth_rate)
        
    def _make_block(self, in_channel, num_layers, growth_rate):
        blocks = []
        for i in range(num_layers):
            blocks.append(Bottleneck(in_channel+(growth_rate*i), growth_rate))
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        x = self.block(x)
        return x
    
        
class Trans_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Trans_Block, self).__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.layer = nn.Sequential(nn.BatchNorm2d(self.in_channel),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel, 
                                   kernel_size=1, bias=False),
                                   nn.AvgPool2d(2))
    def forward(self, x):
        x = self.layer(x)
        return x
    
class Dense_Net(nn.Module):
    def __init__(self, num_layers=[6, 12, 24, 16], in_channels=32, growth_rate=24, num_classes=10):
        super(Dense_Net, self).__init__()
        
        self.conv_1 = nn.Conv2d(3, in_channels, 7, padding=3, bias=False)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Dense block 1
        self.Dense_Block1 = Dense_Block(in_channel=in_channels, num_layers=num_layers[0], growth_rate=growth_rate)
        in_channels = int(in_channels+num_layers[0]*growth_rate)
        self.trans1 = Trans_Block(in_channels, in_channels//2)
        in_channels = in_channels//2
        
        # Dense block 2
        self.Dense_Block2 = Dense_Block(in_channel=in_channels, num_layers=num_layers[1], growth_rate=growth_rate)
        in_channels = int(in_channels+num_layers[1]*growth_rate)
        self.trans2 = Trans_Block(in_channels, in_channels//2)
        in_channels = in_channels//2
        
        # Average pooling layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.batchnorm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_channels, num_classes)
        
                
    def forward(self, x):
        x = self.conv_1(x)
        x = self.max_pool(x)
        
        x = self.Dense_Block1(x)
        x = self.trans1(x)
        
        x = self.Dense_Block2(x)
        x = self.trans2(x)
        
        x = self.avg_pool(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x