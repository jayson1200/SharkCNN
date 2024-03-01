"""
Should be run in megaset labels directory 
"""
import os
import numpy as np
import cv2
from tqdm import tqdm

files = os.listdir("./")

total_area_in_boxes = 0
total_area = len(files)


for label_name in tqdm(files):
    objects = []
    
    try:
        with open(label_name, "r") as labels_doc:
            objects = labels_doc.read().splitlines()
    except:
        raise Exception(f"Failed at {label_name}")

    for object in objects:
        box_keypoints = [float(keypoint_string) for keypoint_string in object.split(" ")]
        
        points = np.array([[box_keypoints[1], box_keypoints[2]], 
                            [box_keypoints[3], box_keypoints[4]], 
                            [box_keypoints[5], box_keypoints[6]], 
                            [box_keypoints[7], box_keypoints[8]]], dtype=np.float32)

        total_area_in_boxes += cv2.contourArea(points, oriented=True)

total_area_not_in_boxes = total_area - total_area_in_boxes
ratio = total_area_not_in_boxes / total_area_in_boxes

print(f"Total Positive Area: {total_area_in_boxes}\nTotal Negative Area: {total_area_not_in_boxes}\nRatio: {ratio}")
