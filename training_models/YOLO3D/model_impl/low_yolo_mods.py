import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Conv(nn.Module):

    def __init__(self,
                 kernel_size, 
                 in_channels, 
                 out_channels,
                 stride,
                 padding):
        
        super().__init__()
        
        self.conv3d = nn.Conv3d(in_channels=in_channels, 
                                out_channels=out_channels, 
                                kernel_size=kernel_size, 
                                stride = stride, 
                                padding = padding)
        
        self.batchnorm3d = nn.BatchNorm3d(out_channels)
        self.silu = nn.SiLU()

        

    def forward(self, x):
        out = self.conv3d(x)
        out = self.batchnorm3d(out)
        out = self.silu(out)
        
        return out
    
class Conv2D(nn.Module):

    def __init__(self,
                 kernel_size, 
                 in_channels, 
                 out_channels,
                 stride,
                 padding):
        
        super().__init__()
        
        self.conv2d = nn.Conv2d(in_channels=in_channels, 
                                out_channels=out_channels, 
                                kernel_size=kernel_size, 
                                stride = stride, 
                                padding = padding)
        
        self.batchnorm2d = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

        

    def forward(self, x):
        out = self.conv2d(x)
        out = self.batchnorm2d(out)
        out = self.silu(out)
        
        return out
    
class Bottleneck(nn.Module):

    def __init__(self, 
             in_channels,
             shortcut):
        
        super().__init__()
        self.shortcut = shortcut
        
        self.conv1 = Conv(in_channels=in_channels, 
                          out_channels=in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1)
        
        self.conv2 = Conv(in_channels=in_channels, 
                          out_channels=in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = (out + x) if self.shortcut else out

        return out
    
class C2F(nn.Module):

    def __init__(self, 
             in_channels,
             shortcut,
             num_bottlenecks):
        super().__init__()

        self.elem_dist = self.distrib_elems(in_channels)

        self.first_split_end = self.elem_dist[0]
        self.second_split_end = self.elem_dist[0] + self.elem_dist[1]
        self.num_bottles = num_bottlenecks

        self.out_channels = self.elem_dist[0] + self.elem_dist[1] + (self.num_bottles * self.elem_dist[2])

        self.conv1 = Conv(in_channels=in_channels, 
                          out_channels=in_channels,
                          kernel_size=1, 
                          padding=0, 
                          stride=1)
        
        self.bottlenecks = nn.ModuleList([Bottleneck(in_channels=self.elem_dist[2], shortcut=shortcut) for _ in range(self.num_bottles)])

        self.conv2 = Conv(in_channels=self.out_channels, 
                          out_channels=in_channels,
                          kernel_size=1, 
                          padding=0, 
                          stride=1)

    def forward(self, x):
        out = self.conv1(x)
    

        first_split = out[:, 0:self.first_split_end,:, :,:]
        second_split = out[:, self.first_split_end:self.second_split_end,:, :,:]
        third_split = out[:, self.second_split_end:,:, :,:]
        
        in_shape = list(x.shape)
        in_shape[1] = self.elem_dist[2] * self.num_bottles

        bottle_out = torch.empty(tuple(in_shape))
        
        curr_bottle_out_start = 0
        curr_bottle_out_end = self.elem_dist[2]

        for bottleneck in self.bottlenecks:
            third_split = bottleneck(third_split)
            bottle_out[:, curr_bottle_out_start:curr_bottle_out_end, :, :, :] = third_split

            curr_bottle_out_start = curr_bottle_out_end
            curr_bottle_out_end += self.elem_dist[2]

        out = torch.cat((first_split, second_split, bottle_out), dim=1)
        out = self.conv2(out)

        return out
    
    def distrib_elems(self, num_elements):
        base, remainder = divmod(num_elements, 3)
        distribution = np.array([base] * 3)

        distribution[:remainder] += 1
        
        return distribution

class SPPF(nn.Module):

    def __init__(self, 
                 in_channels, 
                 kernel_size):
        super().__init__()

        self.conv1 = Conv(kernel_size=3,
                          in_channels=in_channels,
                          out_channels=in_channels,
                          stride=1,
                          padding=1)
        
        padding = int(kernel_size * 0.5)

        self.max_pool1 = nn.MaxPool3d(kernel_size=kernel_size,
                                      padding=padding,
                                      stride=1)
        self.max_pool2 = nn.MaxPool3d(kernel_size=kernel_size,
                                      padding=padding,
                                      stride=1)
        self.max_pool3 = nn.MaxPool3d(kernel_size=kernel_size,
                                      padding=padding,
                                      stride=1)

        self.conv2 = Conv(kernel_size=3,
                          in_channels=in_channels * 4,
                          out_channels=in_channels,
                          stride=1,
                          padding=1)
        
    def forward(self, x):
        out1 = self.conv1(x)

        out2 = self.max_pool1(out1)
        out3 = self.max_pool2(out2)
        out4 = self.max_pool3(out3)
        
        concat_out = torch.cat((out1, out2, out3, out4), dim=1)
        
        out5 = self.conv2(concat_out)

        return out5

 
class Detect_Box(nn.Module):
    
    def __init__(self, in_channels, boxes_per_cell, num_frames):
        super().__init__()
        
        self.lin_comb_kernel = (num_frames, 1, 1)

        self.lin_comb_conv = nn.Conv3d(in_channels=in_channels, 
                                       out_channels=in_channels, 
                                       kernel_size=self.lin_comb_kernel)
        
        self.conv_box1 = Conv2D(in_channels=in_channels, 
                                out_channels=in_channels, 
                                kernel_size=3, 
                                stride=1, 
                                padding=1)
        
        self.conv_box2 = Conv2D(in_channels=in_channels, 
                                out_channels=in_channels, 
                                kernel_size=3, 
                                stride=1, 
                                padding=1)

        self.conv_box_final = nn.Conv2d(in_channels=in_channels, 
                                        out_channels=5 * boxes_per_cell, 
                                        kernel_size=1, 
                                        stride=1)


    
    def forward(self, x):
        x = self.lin_comb_conv(x)
        x = torch.squeeze(x, dim=2)
        x = self.conv_box1(x)
        x = self.conv_box2(x)
        x = self.conv_box_final(x)

        return x
        
class Detect_Class(nn.Module):
    
    def __init__(self, in_channels, boxes_per_cell, num_frames):
        super().__init__()

        self.lin_comb_kernel = (num_frames, 1, 1)

        self.lin_comb_conv = nn.Conv3d(in_channels=in_channels, 
                                       out_channels=in_channels, 
                                       kernel_size=self.lin_comb_kernel)
        
        self.conv_class1 = Conv2D(in_channels=in_channels, 
                                out_channels=in_channels, 
                                kernel_size=3, 
                                stride=1, 
                                padding=1)
        
        self.conv_class2 = Conv2D(in_channels=in_channels, 
                                out_channels=in_channels, 
                                kernel_size=3, 
                                stride=1, 
                                padding=1)
    
        self.conv_class_final = nn.Conv2d(in_channels=in_channels, 
                                          out_channels = boxes_per_cell, 
                                          kernel_size=1, 
                                          stride=1)


    
    def forward(self, x):
        x = self.lin_comb_conv(x)
        x = torch.squeeze(x, dim=2)
        x = self.conv_class1(x)
        x = self.conv_class2(x)
        x = self.conv_class_final(x)
        
        return x

class Detect(nn.Module):

    def __init__(self, in_channels, boxes_per_cell, num_frames):
        
        super().__init__()
        self.detect_box = Detect_Box(in_channels=in_channels, 
                                     boxes_per_cell=boxes_per_cell, 
                                     num_frames=num_frames)
        
        self.detect_class = Detect_Class(in_channels=in_channels, 
                                       boxes_per_cell=boxes_per_cell, 
                                       num_frames=num_frames)
        

    def forward(self, x):
        print(x.shape)
        box_out = self.detect_box(x)
        class_out = self.detect_class(x)

        return box_out, class_out