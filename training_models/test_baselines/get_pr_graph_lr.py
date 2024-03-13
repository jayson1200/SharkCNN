import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import albumentations as A
import cv2
import numpy as np
import skimage as ski

import matplotlib.pyplot as plt
import os
import copy

from tqdm import tqdm
from IPython.display import clear_output

import psutil
import pynvml
import sys

import struct

import sklearn.metrics as metrics

import gc

sys.path.append('/home/meribejayson/Desktop/Projects/SharkCNN/training_models/dataloaders/')


from test_dataset import SharkDatasetTest as SharkDataset


output_file_path = 'preds_labels.dat'

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

image_width = 1920
image_height = 1080

target_iters = 300
images_per_iter = 5

target_sample = image_height * image_width * target_iters * images_per_iter

def read_from_binary_file(file_path):
    dt = np.dtype([('ann_pred', np.float32), ('lr_pred', np.float32), ('label', np.uint32)])
    
    record_size = dt.itemsize

    total_records = os.path.getsize(file_path) // record_size

    quarter_records = total_records // 8

    data = np.memmap(file_path, dtype=dt, mode='r', shape=(quarter_records,))
    
    ann_preds = data['ann_pred']
    lr_preds = data['lr_pred']
    labels = data['label'].astype(int)
    
    return ann_preds, lr_preds, labels

preds_ann, preds_lr, all_labels = read_from_binary_file(output_file_path)

# Create larger plot for better readability
plt.figure(figsize=(10, 8))

# Calculate precision-recall curve
precision_lr, recall_lr, _ = metrics.precision_recall_curve(all_labels, preds_lr)

# Plot the precision-recall curve
plt.plot(recall_lr, precision_lr, color='blue')

plt.title('Precision-Recall Curve for LR', fontsize=16)
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)

# Set the limits for the X and Y axes
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 0.2])

# Show the plot
plt.show()

# Clear the current figure
plt.clf()