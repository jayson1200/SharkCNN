import torch
import torch.nn as nn
import torch.nn.functional as F

import low_yolo_mods as mods


class Backbone(nn.Module):

    def __init__(self, 
                 in_channels):
        self.conv1 = mods.Conv(kernel_size=3,
                               in_channels=3,
                               out_channels=42,
                               stride=2,
                               padding=1)
        
        self.conv2 = mods.Conv(kernel_size=3,
                               in_channels=in_channels,
                               out_channels=in_channels,
                               stride=2,
                               padding=1)
        
        self.c2f1 = mods.C2F(in_channels=in_channels,
                             shortcut=True,
                             num_bottlenecks=2)
        
        self.conv3 = mods.Conv(kernel_size=3,
                        in_channels=in_channels,
                        out_channels=in_channels,
                        stride=2,
                        padding=1)
        
        self.c2f2_out = mods.C2F(in_channels=in_channels,
                                 shortcut=True,
                                 num_bottlenecks=4)
        
        self.conv4 = mods.Conv(kernel_size=3,
                               in_channels=in_channels,
                               out_channels=in_channels,
                               stride=2,
                               padding=1)
        
        self.c2f3_out = mods.C2F(in_channels=42,
                                 shortcut=True,
                                 num_bottlenecks=4)
        
        self.conv5 = mods.Conv(kernel_size=3,
                               in_channels=in_channels,
                               out_channels=in_channels,
                               stride=2,
                               padding=1)
        
        self.c2f4 = mods.C2F(in_channels=in_channels,
                                 shortcut=True,
                                 num_bottlenecks=2)
        
        self.sppf1 = mods.SPPF(in_channels=in_channels,
                               kernel_size=5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.c2f1(x)
        x = self.conv3(x)
        x_low = self.c2f2_out(x)

        x = self.conv4(x_low)
        x_mid = self.c2f3_out(x)

        x = self.conv5(x_mid)
        x = self.c2f4(x)
        x_high = self.sppf1(x)

        return (x_low, x_mid, x_high)

class PAN(nn.Module):

    def __init__(self):
        pass

    def forward(self, x_low, x_mid, x_high):
        pass

class Detect(nn.Module):

    def __init__(self):
        pass

    def forward(self, x):
        pass