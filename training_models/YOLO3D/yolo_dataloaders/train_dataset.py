import torch
import sys
sys.path.append("/home/meribejayson/Desktop/Projects/SharkCNN/training_models/YOLO3D/utils")

import utils
from torch.utils.data import Dataset

import os
import cv2
import numpy as np

datasets_path = "/home/meribejayson/Desktop/Projects/SharkCNN/datasets-reduced/"
image_target_shape = (1, 3, 1, 1080, 1920)

box_high_shape = (1, 5, 17, 30)
box_mid_shape = (1, 5, 34, 60)
box_low_shape = (1, 5, 68, 120)

                   
class SharkYOLODataset(Dataset):
    def __init__(self, 
                 num_frames,
                 device):
        
        self.num_frames = num_frames

        self.device = device
        
        self.recordings = ["dji114", "dji136", "dji206", "dji246", "dji343", "dji344", 
                           "dji345", "dji349", "dji358", "dji359", "nodriz,mar,c,laura"]
        
        self.recordings_len = []   
        self.num_of_total_imgs = 0

        for recording_name in self.recordings:
            recording_path = datasets_path + recording_name + "/train/images"
            
            num_images_in_folder = len(os.listdir(recording_path))

            self.recordings_len.append(num_images_in_folder)
            self.num_of_total_imgs += num_images_in_folder

        self.total_training_examples = self.num_of_total_imgs - ((num_frames - 1) * len(self.recordings))

    def __len__(self):
        return self.total_training_examples
    
    def generate_ground_truth(self, recording_name, out_shape):
        label_path = datasets_path + "/train/labels/" + recording_name
        
        return utils.get_ground_truth_boxes(label_path=label_path, box_output_shape=out_shape, device=self.device)

    def __getitem__(self, idx):
        target_frame_idx = idx + self.num_frames
        frame_end_idx = 0
        target_record = ""

        frames = []

        for len, name in zip(self.recordings_len, self.recordings):
            if(len >= target_frame_idx):
                frame_end_idx = target_frame_idx - 1
                target_record = name
                break
            
            target_frame_idx -= len 
        
        for i in range(self.num_frames):
            img_name = f"{frame_end_idx - i}.jpg"
            img_path = datasets_path + target_record +  "/train/images/" + img_name
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)
            img = torch.from_numpy(np.expand_dims(np.expand_dims(img.transpose(2,0,1), axis=0), axis=2))
            frames.append(img)

        exp = torch.cat(tuple(frames), dim=2)

        high = self.generate_ground_truth(img_name, box_high_shape)
        mid = self.generate_ground_truth(img_name, box_mid_shape)
        low = self.generate_ground_truth(img_name, box_low_shape)

        return torch.squeeze(exp).to(device=self.device), (high, mid, low)



