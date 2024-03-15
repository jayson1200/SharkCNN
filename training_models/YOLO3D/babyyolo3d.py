"""
    This is the Conv block of the YOLOv8 model. It takes an array of size (1, C, D, H, W), where
    C is the channel dimension D is the temporal component, listing sequential frames. H is the height
    component of the image and W is the width component representing the width of the image. 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):

    def __init__(self,
                 kernel_size = 3, 
                 in_channels = 3, 
                 out_channels = 42,
                 stride = 1,
                 padding = 1):
        
        super().__init__()
        
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride = stride, padding = padding)
        self.batchnorm3d = nn.BatchNorm3d(out_channels)
        self.silu = nn.SiLU()

        

    def forward(self, x):
        out = self.conv3d(x)
        out = self.batchnorm3d(out)

        return self.silu(out)
    
class Bottleneck(nn.Module):

    def __init__(self, 
             in_channels):
        
        super().__init__()
        
        self.conv1 = Conv(in_channels=in_channels, out_channels=in_channels)
        self.conv2 = Conv(in_channels=in_channels, out_channels=in_channels)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out
    
class C2F(nn.Module):

    def __init__(self, 
             in_channels):
        super().__init__()
        self.elem_dist = distrib_elems(in_channels)

        self.conv1 = Conv(kernel_size=1,padding=0)
        self.bottleneck1 = Bottleneck(in_channel=self.elem_dist[2])
        self.bottleneck2 = Bottleneck(in_channel=self.elem_dist[2])
        self.conv2 = Conv(kernel_size=1, in_channels = in_channels, padding=0)

    def forward(self, x):
        out = self.conv1(x)

        first_split = out[:, 0:self.elem_dist[0],:, :,:]
        second_split = out[:, self.elem_dist[0]:self.elem_dist[1],:, :,:]
        third_split = out[:, self.elem_dist[1]:,:, :,:]

        bottle_out_1 =  self.bottleneck1(third_split)
        bottle_out_2 = self.bottleneck2(bottle_out_1)

        out = torch.cat((first_split, second_split, bottle_out_1, bottle_out_2), dim=1)

        return self.conv2(out)

 
class Detect_Box(nn.Module):
    
    def __init__(self, in_channels, boxes_per_cell, num_frames):
        super().__init__()
        self.convbox1 = Conv(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.convbox2 = Conv(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        
        self.coefficients = nn.Parameter(torch.randn(num_frames, 1))
        self.convboxfinal = nn.Conv2d(in_channels, out_channels=5 * boxes_per_cell, kernel_size=1, stride=1)


    
    def forward(self, x):
        x = self.convbox1(x)
        x = self.convbox2(x)
        
        original_shape = x.shape
        x = x.view(-1, original_shape[2], original_shape[3], original_shape[4])
        
        x = x.permute(0, 2, 3, 1)
        x = torch.matmul(x, self.coefficients)  
        x = x.permute(0, 3, 1, 2) 

        x = x.view(original_shape[0], original_shape[1], original_shape[3], original_shape[4])

        return self.convboxfinal(x)
        
class Detect_Class(nn.Module):
    
    def __init__(self, in_channels, boxes_per_cell, num_frames):
        super().__init__()
        self.convclass1 = Conv(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.convclass2 = Conv(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        
        self.coefficients = nn.Parameter(torch.randn(num_frames, 1))
        self.convclassfinal = nn.Conv2d(in_channels, out_channels = boxes_per_cell, kernel_size=1, stride=1)


    
    def forward(self, x):
        x = self.convclass1(x)
        x = self.convclass2(x)
        
        original_shape = x.shape
        x = x.view(-1, original_shape[2], original_shape[3], original_shape[4])
        
        x = x.permute(0, 2, 3, 1)
        x = torch.matmul(x, self.coefficients)  
        x = x.permute(0, 3, 1, 2) 

        x = x.view(original_shape[0], original_shape[1], original_shape[3], original_shape[4])

        return self.convclassfinal(x)


class BabyYolo(nn.Module):

    def __init__(self, num_frames, boxes_per_cell):
        super().__init__()
        self.conv1 = Conv(kernel_size=3, in_channels=3, out_channels=42, stride=4, padding=1)

        self.conv2 = Conv(kernel_size=3, in_channels=42, out_channels=42, stride=4, padding=1)
        self.c2f1 = C2F(in_channels=42)

        self.conv3 = Conv(kernel_size=3, in_channels=42, out_channels=42, stride=4, padding=1)
        self.c2f2 = C2F(in_channels=42)

        self.conv4 = Conv(kernel_size=3, in_channels=42, out_channels=42, stride=4, padding=1)
        self.c2f3 = C2F(in_channels=42)

        self.conv5 = Conv(kernel_size=3, in_channels=42, out_channels=42, stride=4, padding=1)
        self.c2f4 = C2F(in_channels=42)

        self.conv6 = Conv(kernel_size=3, in_channels=42, out_channels=42, stride=4, padding=1)
        self.c2f5 = C2F(in_channels=42)

        self.conv7 = Conv(kernel_size=3, in_channels=42, out_channels=42, stride=4, padding=1)
        self.c2f6 = C2F(in_channels=42)
        
        self.detect_box = Detect_Box(42, 1, 6)
        self.detect_class = Detect_Class(42, 1, 6)


    def forward(self, x):
        self.con

 
def distrib_elems(num_elements):
    base, remainder = divmod(num_elements, 3)
    distribution = [base, base, base]
    for i in range(remainder):
        distribution[i] += 1
    return distribution
