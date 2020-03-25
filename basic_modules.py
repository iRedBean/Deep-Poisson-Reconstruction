import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        ret = F.leaky_relu(self.conv(x))
        return ret

class DeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(DeConvBlock, self).__init__()

        self.de_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        ret = F.relu(self.de_conv(x))
        return ret

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()

        self.conv_1 = ConvBlock(channels, channels, kernel_size, stride=1, padding=padding)
        self.conv_2 = nn.Conv2d(channels, channels, kernel_size, stride=1, padding=padding)

    def forward(self, x):
        ret = x + self.conv_2(self.conv_1(x))
        return ret

class FullyConnectedBlocks(nn.Module):
    def __init__(self, features_list, activation_list, dropout = None):
        super(FullyConnectedBlocks, self).__init__()

        fc_blocks = []
        for i in range(1, len(features_list)):
            if dropout:
                fc_blocks += [nn.Dropout(dropout)]
            fc_blocks += [nn.Linear(features_list[i-1], features_list[i])]
            if activation_list[i-1] == 'l':
                fc_blocks += [nn.LeakyReLU()]
            elif activation_list[i-1] == 't':
                fc_blocks += [nn.Tanh()]
            elif activation_list[i-1] == 's':
                fc_blocks += [nn.Sigmoid()]
        self.fc_blocks = nn.Sequential(*fc_blocks)

    def forward(self, x):
        ret = self.fc_blocks(x)
        return ret

class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()

        self.args = args
    
    def forward(self, x):
        ret = x.view(self.args)
        return ret
