import torch
import torch.nn as nn

import low_yolo_mods as mods


class Backbone(nn.Module):

    def __init__(self, 
                 in_channels):
        super().__init__()
        self.conv1 = mods.Conv(kernel_size=3,
                               in_channels=3,
                               out_channels=in_channels,
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

        return x_low, x_mid, x_high

class PANONE(nn.Module):

    def __init__(self, x_low_shape, x_mid_shape):
        super().__init__()
        self.upsample_high_mid = nn.Upsample(size=x_mid_shape)
        self.c2f1_out = mods.C2F(in_channels=x_mid_shape[1] * 2, shortcut=False, num_bottlenecks=2)

        self.upsample_mid_low = nn.Upsample(size=x_low_shape)
        self.c2f2_out = mods.C2F(in_channels=x_low_shape[1] * 2, shortcut=False, num_bottlenecks=2)

    def forward(self, x_low, x_mid, x_high):
        high_mid_up = self.upsample_high_mid(x_high)
        high_mid_concat = torch.cat((x_mid, high_mid_up), dim=1)
        
        x_mid_out = self.c2f1_out(high_mid_concat)
        
        mid_low_up = self.upsample_mid_low(x_mid_out)
        mid_low_concat = torch.cat((x_low, mid_low_up), dim=1)

        x_low_out = self.c2f2_out(mid_low_concat)

        return x_low_out, x_mid_out, x_high
    
class PANTWO(nn.Module):

    def __init__(self, x_low_shape, x_mid_shape, x_high_shape):
        super().__init__()
        self.conv1 = mods.Conv(kernel_size=3,
                               in_channels=x_low_shape[1],
                               out_channels=x_low_shape[1],
                               stride=2,
                               padding=1)
        
        concat_1_chn_size = x_low_shape[1] + x_mid_shape[1]
        
        self.c2f1_out = mods.C2F(in_channels=concat_1_chn_size,
                                 shortcut=False,
                                 num_bottlenecks=2)
        
        self.conv2 = mods.Conv(kernel_size=3,
                               in_channels=concat_1_chn_size,
                               out_channels=concat_1_chn_size,
                               stride=2,
                               padding=1)
        
        concat_2_chn_size = x_high_shape[1] + x_mid_shape[1]

        self.c2f2_out = mods.C2F(in_channels=concat_2_chn_size,
                                 shortcut=False,
                                 num_bottlenecks=2)
        


    def forward(self, x_low, x_mid, x_high):
        
        x_low_out = x_low

        x1_conv = self.conv1(x_low_out)
        low_mid_concat = torch.concat((x_mid, x1_conv), dim=1)
        x_mid_out = self.c2f1_out(low_mid_concat)

        x2_conv = self.conv2(x_mid_out)
        mid_high_concat = torch.concat((x_high, x2_conv), dim=1)
        x_high_out = self.c2f2_out(mid_high_concat)

        return x_low_out, x_mid_out, x_high_out
    

class PAN(nn.Module):
    def __init__(self, x_low_shape, x_mid_shape, x_high_shape):
        super().__init__()

        self.pan_one = PANONE(x_low_shape=x_low_shape, 
                              x_high_shape=x_high_shape)
        
        self.pan_two = PANONE(x_low_shape=x_low_shape,
                              x_mid_shape=x_mid_shape, 
                              x_high_shape=x_high_shape)

    def forward(self, x_low, x_mid, x_high):

        x_low, x_mid, x_high = self.pan_one(x_low, x_mid, x_high)
        x_low, x_mid, x_high = self.pan_two(x_low, x_mid, x_high)

        return x_low, x_mid, x_high
