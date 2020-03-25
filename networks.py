import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_modules import ConvBlock, DeConvBlock, ResidualBlock

class GradNet(nn.Module):
    def __init__(self):
        super(GradNet, self).__init__()

        # encoder for I_b & F
        self.encoder_1 = ConvBlock(8, 32)
        self.encoder_2 = ConvBlock(32, 64)
        self.encoder_3 = ConvBlock(64, 128)

        # encoder for I_dx & I_dy
        self.g_encoder_1 = ConvBlock(2, 32)
        self.g_encoder_2 = ConvBlock(32, 64)
        self.g_encoder_3 = ConvBlock(64, 128)

        # residual blocks
        self.residual_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )

        # decoder
        self.decoder_1 = DeConvBlock(512, 128)
        self.decoder_2 = DeConvBlock(256, 64)
        self.decoder_3 = nn.Sequential(
            DeConvBlock(128, 32),
            nn.Conv2d(32, 1, 3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x, g):
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(x1)
        x3 = self.encoder_3(x2)

        g1 = self.g_encoder_1(g)
        g2 = self.g_encoder_2(g1)
        g3 = self.g_encoder_3(g2)

        x_cat = torch.cat([x3, g3], 1)
        x_cat = self.residual_blocks(x_cat)

        r1 = self.decoder_1(torch.cat([x_cat, x3, g3], 1))
        r2 = self.decoder_2(torch.cat([r1, x2, g2], 1))
        r3 = self.decoder_3(torch.cat([r2, x1, g1], 1))

        return r3

class GNet(nn.Module):
    def __init__(self):
        super(GNet, self).__init__()

        self.model = nn.Sequential(
            ConvBlock(10, 64, stride=1),
            ConvBlock(64, 64, stride=1),
            nn.Conv2d(64, 7, 3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)
