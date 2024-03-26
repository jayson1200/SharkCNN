import numpy as np
import cv2
import torch


def keypoints_to_xywh_theta(keypoints):
    points = np.array(keypoints, dtype=np.float32)

    rect = cv2.minAreaRect(points)

    center, size, angle = rect

    return {
        'x': center[0],
        'y': center[1],
        'w': size[0],
        'h': size[1],
        'theta': (angle / 180) * np.pi
    }

# This can definetly written more efficiently. However, I don't have time to figure out the math
def get_max_intersecting_iou_grid_loc(box_loc, output_shape):
    grid_box_h = 1 / output_shape[2]
    grid_box_w = 1 / output_shape[3]
    
    grid_box_area = grid_box_h * grid_box_w
    box_loc_area = cv2.contourArea(box_loc)
    
    grid_label_union = box_loc_area + grid_box_area

    int_ind = []
    ious = []

    for y in range(output_shape[2]):
        for x in range(output_shape[3]):
            
            key_1 = [grid_box_w * x, grid_box_h * y]
            key_2 = [grid_box_w * (x+1), grid_box_h * y]
            key_3 = [grid_box_w * (x+1), grid_box_h * (y+1)]
            key_4 = [grid_box_w * x, grid_box_h * (y+1)]

            grid_coords = np.array([key_1, key_2, key_3, key_4], dtype=np.float32)

            int_area, _ = cv2.intersectConvexConvex(grid_coords, box_loc, handleNested=True)

            if(int_area > 0):
                int_ind.append([x, y])
                ious.append(int_area / grid_label_union)

    return int_ind[np.argmax(ious)], int_ind

def get_intersecting_grid_loc(box_loc, output_shape):
    grid_box_h = 1 / output_shape[2]
    grid_box_w = 1 / output_shape[3]

    int_ind = []

    for y in range(output_shape[2]):
        for x in range(output_shape[3]):
            
            key_1 = [grid_box_w * x, grid_box_h * y]
            key_2 = [grid_box_w * (x+1), grid_box_h * y]
            key_3 = [grid_box_w * (x+1), grid_box_h * (y+1)]
            key_4 = [grid_box_w * x, grid_box_h * (y+1)]

            grid_coords = np.array([key_1, key_2, key_3, key_4], dtype=np.float32)

            int_area, _ = cv2.intersectConvexConvex(grid_coords, box_loc, handleNested=True)

            if(int_area > 0):
                int_ind.append([x, y])

    return int_ind


def get_coords_from_label_file(label_path):
    labels = []

    try:
        with open(label_path, "r") as labels_doc:
            labels = labels_doc.read().splitlines()
    except:
        return None
    
    label_points = []

    for labels_string in labels:
        box_keypoints = [float(keypoint_string) for keypoint_string in labels_string.split(" ")]

        label_points.append(np.array([[box_keypoints[1], box_keypoints[2]], 
                                             [box_keypoints[3], box_keypoints[4]], 
                                             [box_keypoints[5], box_keypoints[6]], 
                                             [box_keypoints[7], box_keypoints[8]]], 
                                             dtype=np.float32))
    
    
    return label_points if label_points != [] else None

"""
Example box output: torch.Size([1, 5, 68, 120]) 
Example class prob output: torch.Size([1, 1, 68, 120])
"""
def get_ground_truth_boxes_idx(label_path, box_output_shape):
    class_output_shape = (box_output_shape[0], 1, box_output_shape[2], box_output_shape[3])

    box_outputs = np.zeros(box_output_shape, dtype=np.float32)
    class_outputs = np.zeros(class_output_shape, dtype=np.float32)

    coords = get_coords_from_label_file(label_path)

    if (coords == None):
        return box_outputs, class_outputs
    
    for coord in coords:
        formated_coord_dict = keypoints_to_xywh_theta(coord)
        formated_vec = np.array([formated_coord_dict["x"],
                                                        formated_coord_dict["y"],
                                                        formated_coord_dict["w"], 
                                                        formated_coord_dict["h"], 
                                                        formated_coord_dict["theta"]])
        
        idx, intersecting_idxs = get_max_intersecting_iou_grid_loc(coord, box_output_shape)

        int_idxs = np.array(intersecting_idxs)
        
        box_outputs[0, :, int_idxs[:, 1], int_idxs[:, 0]] = formated_vec
        class_outputs[0, :, int_idxs[:, 1], int_idxs[:, 0]] = 1


    return box_outputs, class_outputs


def get_ground_truth_boxes(label_path, box_output_shape, device):
    class_output_shape = (box_output_shape[0], 1, box_output_shape[2], box_output_shape[3])

    box_outputs = np.zeros(box_output_shape, dtype=np.float32)
    class_outputs = np.zeros(class_output_shape, dtype=np.float32)

    coords = get_coords_from_label_file(label_path)

    if (coords == None):
        return torch.from_numpy(box_outputs).to(device), torch.from_numpy(class_outputs).to(device), torch.tensor(0).to(device)
    
    total_pos = 0
    for coord in coords:
        formated_coord_dict = keypoints_to_xywh_theta(coord)
        formated_vec = np.array([formated_coord_dict["x"],
                                                        formated_coord_dict["y"],
                                                        formated_coord_dict["w"], 
                                                        formated_coord_dict["h"], 
                                                        formated_coord_dict["theta"]])
        
        intersecting_idxs = get_intersecting_grid_loc(coord, box_output_shape)
        
        total_pos += len(intersecting_idxs) 
        int_idxs = np.array(intersecting_idxs)
        
        box_outputs[0, :, int_idxs[:, 1], int_idxs[:, 0]] = formated_vec
        class_outputs[0, :, int_idxs[:, 1], int_idxs[:, 0]] = 1


    return (torch.from_numpy(box_outputs).to(device), torch.from_numpy(class_outputs).to(device), torch.tensor(total_pos).to(device))

