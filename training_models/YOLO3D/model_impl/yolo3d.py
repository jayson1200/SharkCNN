import torch
import torch.nn as nn

import high_yolo_mods as highmods
import low_yolo_mods as lowmods

class YOLO3D(nn.Module):
    
    def __init__(self, num_frames, image_features):
        super().__init__()

        self.num_frames = num_frames
        self.image_features = image_features

        self.x_low_shape = (1, 42, self.num_frames, 135, 240)
        self.x_mid_shape = (1, 42, self.num_frames, 68, 120)
        self.x_high_shape = (1, 42, self.num_frames, 34, 60)

        self.backbone = highmods.Backbone(in_channels=self.image_features)

        self.neck = highmods.PAN(x_low_shape=self.x_low_shape, 
                                x_mid_shape=self.x_mid_shape,
                                x_high_shape=self.x_high_shape)
        
        self.detect_low = lowmods.Detect(in_channels=self.image_features,
                                         boxes_per_cell=1,
                                         num_frames=self.num_frames)

        self.detect_mid = lowmods.Detect(in_channels=self.image_features,
                                         boxes_per_cell=1,
                                         num_frames=self.num_frames)

        self.detect_high = lowmods.Detect(in_channels=self.image_features,
                                         boxes_per_cell=1,
                                         num_frames=self.num_frames)


    def forward(self, x):
        x_low, x_mid, x_high = self.backbone(x)
        x_low, x_mid, x_high = self.neck(x_low, x_mid, x_high)
        
        low_preds = self.detect_low(x_low)
        mid_preds = self.detect_mid(x_mid)
        high_preds = self.detect_high(x_high)

        return low_preds, mid_preds, high_preds