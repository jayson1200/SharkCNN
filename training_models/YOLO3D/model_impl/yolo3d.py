import torch
import torch.nn as nn

import high_yolo_mods as highmods
import low_yolo_mods as lowmods

class YOLO3D(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.x_low_shape = (None, 42, None, None, None)
        self.x_mid_shape = (None, 42, None, None, None)
        self.x_high_shape = (None, 42, None, None, None)

        self.backbone = highmods.Backbone(in_channels=42)

        self.neck = highmods.PAN(x_low_shape=self.x_low_shape, 
                                x_mid_shape=self.x_mid_shape,
                                x_high_shape=self.x_high_shape)
        
        self.detect_low = lowmods.Detect(in_channels=42,
                                         boxes_per_cell=1,
                                         num_frames=6)

        self.detect_mid = lowmods.Detect(in_channels=42,
                                         boxes_per_cell=1,
                                         num_frames=6)

        self.detect_high = lowmods.Detect(in_channels=42,
                                         boxes_per_cell=1,
                                         num_frames=6)


    def forward(self, x):
        x_low, x_mid, x_high = self.backbone(x)
        x_low, x_mid, x_high = self.neck(x_low, x_mid, x_high)
        
        low_preds = self.detect_low(x_low)
        mid_preds = self.detect_mid(x_mid)
        high_preds = self.detect_high(x_high)

        return low_preds, mid_preds, high_preds