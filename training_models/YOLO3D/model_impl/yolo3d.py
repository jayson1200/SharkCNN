import torch
import torch.nn as nn

import sys

sys.path.append("/home/meribejayson/Desktop/Projects/SharkCNN/training_models/YOLO3D/model_impl")

import low_yolo_mods as lowmods
import high_yolo_mods as highmods


class YOLO3D(nn.Module):
    
    def __init__(self, num_frames, num_features):
        super().__init__()

        self.num_frames = num_frames
        self.num_features = num_features

        self.x_low_shape = (1, 128, 4, 135, 240)
        self.x_mid_shape = (1, 384, 2, 68, 120)
        self.x_high_shape = (1, 576, 1, 34, 60)

        self.backbone = highmods.Backbone(in_channels=self.num_features)

        self.neck = highmods.PAN(x_low_shape=self.x_low_shape, 
                                x_mid_shape=self.x_mid_shape,
                                x_high_shape=self.x_high_shape)
        
        low_in_channels = self.x_high_shape[1] + self.x_mid_shape[1] + self.x_low_shape[1]
        mid_in_channels = self.x_high_shape[1] + self.x_mid_shape[1] + low_in_channels
        high_in_channels = self.x_high_shape[1] + mid_in_channels

        self.detect_low = lowmods.Detect(in_channels=low_in_channels,
                                         boxes_per_cell=1,
                                         num_frames=self.num_frames)

        self.detect_mid = lowmods.Detect(in_channels=mid_in_channels,
                                         boxes_per_cell=1,
                                         num_frames=self.num_frames)

        self.detect_high = lowmods.Detect(in_channels=high_in_channels,
                                         boxes_per_cell=1,
                                         num_frames=self.num_frames)


    def forward(self, x):
        x_low, x_mid, x_high = self.backbone(x)
        x_low, x_mid, x_high = self.neck(x_low, x_mid, x_high)
        
        low_preds = self.detect_low(x_low)
        mid_preds = self.detect_mid(x_mid)
        high_preds = self.detect_high(x_high)

        return low_preds, mid_preds, high_preds