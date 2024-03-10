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

import torch

megaset_train_images_path = "/home/meribejayson/Desktop/Projects/SharkCNN/datasets-reduced/megaset/train/images/"
megaset_train_labels_path = "/home/meribejayson/Desktop/Projects/SharkCNN/datasets-reduced/megaset/train/labels/"

image_width = 1920
image_height = 1080



class SharkImageFeatureGen():

    def __init__(self):
        self.image_names = os.listdir(megaset_train_images_path)
        self.num_images = len(self.image_names)

    """
        The following return a numpy array representing the image in BGR format
    """
    def get_image(self, image_name):
        file_path = megaset_train_images_path + image_name
        image = cv2.imread(file_path)

        return (image_name, image)
    
    """
        Assumes input image is in grayscale
    """
    def get_canny_output(self, image):
        return cv2.Canny(image, 50, 100)
    
    """
        Expecting image in BGR format
    """
    def get_color_gradients(self, image):
        blue_dy = ski.filters.sobel_h(image[:, :, 0])
        blue_dx = ski.filters.sobel_v(image[:, :, 0])

        green_dy = ski.filters.sobel_h(image[:, :, 1])
        green_dx = ski.filters.sobel_v(image[:, :, 1])

        red_dy = ski.filters.sobel_h(image[:, :, 2])
        red_dx = ski.filters.sobel_v(image[:, :, 2])

        return (blue_dx, blue_dy, green_dx, green_dy, red_dx, red_dy)
    
    def get_direction(self, image_dx, image_dy):
        peturbation = 2e-125

        image_dy_peturb = copy.deepcopy(image_dy)
        image_dx_peturb = copy.deepcopy(image_dx)

        image_dy_peturb[image_dy == 0.0] = peturbation
        image_dx_peturb[image_dx == 0.0] = peturbation

        return np.arctan(image_dy_peturb / image_dx_peturb)

    """
        Expecting image in BGR format

        return gabor feature image per channel (8, 1080, 1920)
    """
    def get_gabor_feature_vector(self, image):
        size = 20
        pi_2 = np.pi / 2

        sigma = np.array([1.0, 1.0, 3.0, 1.0, 3.0, 1.0, 2.0, 1.0])
        theta = np.array([0.0, 0.0, 0.2617993877991494, 0.5235987755982988, 0.2617993877991494, 0.2617993877991494, 0.0, 0.0])
        lambd = np.array([10.0, 15.0, 20.0, 15.0, 20.0, 20.0, 15.0, 20.0])
        gamma = np.array([0.5, 1.5, 1.5, 1.5, 1.0, 1.5, 1.5, 1.0])
        psi = np.array([pi_2, pi_2, pi_2, pi_2, pi_2, pi_2, pi_2, pi_2])

        R = []
        G = []
        B = []

        for idx in range(sigma.shape[0]): 
            kern = cv2.getGaborKernel((size, size), sigma[idx], theta[idx], lambd[idx], gamma[idx], psi[idx], ktype=cv2.CV_32F)
            B.append(cv2.filter2D(image[:, :, 0], -1, kern))
            G.append(cv2.filter2D(image[:, :, 1], -1, kern))
            R.append(cv2.filter2D(image[:, :, 2], -1, kern))
            
        
        R_np = np.array(R)
        G_np = np.array(G)
        B_np = np.array(B)
         
        return (B_np, G_np, R_np)
    
    def get_gabor_feature_gradients(self, B_np, G_np, R_np):
        R_dx = []
        G_dx = []
        B_dx = []

        R_dy = []
        G_dy = []
        B_dy = []

        for i in range(B_np.shape[0]):
            B_dx.append(ski.filters.sobel_v(B_np[i, :, :]))
            B_dy.append(ski.filters.sobel_h(B_np[i, :, :]))

            G_dx.append(ski.filters.sobel_v(G_np[i, :, :]))
            G_dy.append(ski.filters.sobel_h(G_np[i, :, :]))

            R_dx.append(ski.filters.sobel_v(R_np[i, :, :]))
            R_dy.append(ski.filters.sobel_h(R_np[i, :, :]))
        
        R_dx_np = np.array(R_dx)
        R_dy_np = np.array(R_dy)

        G_dx_np = np.array(G_dx)
        G_dy_np = np.array(G_dy)

        B_dx_np = np.array(B_dx)
        B_dy_np = np.array(B_dy)

        return (B_dx_np, B_dy_np, G_dx_np, G_dy_np, R_dx_np, R_dy_np) 

    def is_pixel_in_box(self, image_name, pixel_loc_x, pixel_loc_y):
        label_name = image_name.split(".")[0] + ".txt"
        label_path = megaset_train_labels_path + label_name

        labels = []
        
        # I assume images that don't have an sharks
        try:
            with open(label_path, "r") as labels_doc:
                labels = labels_doc.read().splitlines()
        except FileNotFoundError:
            return False  

        
        for labels_string in labels:
            box_keypoints = [float(keypoint_string) for keypoint_string in labels_string.split(" ")]
            points = np.array([[box_keypoints[1], box_keypoints[2]], 
                               [box_keypoints[3], box_keypoints[4]], 
                               [box_keypoints[5], box_keypoints[6]], 
                               [box_keypoints[7], box_keypoints[8]]], dtype=np.float32)
            
            pixel_norm_loc = np.array([pixel_loc_x / image_width, pixel_loc_y / image_height])

            if(cv2.pointPolygonTest(points,  pixel_norm_loc, False) >= 0):
                return True
            
        return False
    


    def test_shark_dataset_functions(self):
        # Test get_random_image
        image_name, image = self.get_random_image()
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Test transform_image
        transformed_image = self.transform_image(image)
        plt.subplot(1, 2, 2)
        plt.title("Transformed Image")
        plt.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
        plt.show()
        
        # Test get_canny_output (Assuming the transformed image is suitable)
        canny_output = self.get_canny_output(cv2.cvtColor(transformed_image, cv2.COLOR_RGB2GRAY))
        plt.figure()
        plt.title("Canny Output")
        plt.imshow(canny_output, cmap='gray')
        plt.show()
        
        # Test get_color_gradients
        blue_dx, blue_dy, green_dx, green_dy, red_dx, red_dy = self.get_color_gradients(image)
        # Visualizing one of the gradients
        plt.figure(figsize=(10, 5))
        plt.title("Gradient - Blue Channel DX")
        plt.imshow(blue_dx, cmap='gray')
        plt.show()
        
        # No direct visualization for get_direction as it's more of a computation method
        
        # Test get_gabor_feature_vector
        B_np, G_np, R_np = self.get_gabor_feature_vector(image)
        # Visualize one of the Gabor feature images
        plt.figure()
        plt.title("Gabor Feature - Blue Channel")
        plt.imshow(B_np[0], cmap='gray')
        plt.show()
        
        # Test get_gabor_feature_gradients (Using Gabor features from previous step)
        B_dx_np, B_dy_np, G_dx_np, G_dy_np, R_dx_np, R_dy_np = self.get_gabor_feature_gradients(B_np, G_np, R_np)
        # Visualize one of the gradient images
        plt.figure()
        plt.title("Gabor Feature Gradient - Blue Channel DX")
        plt.imshow(B_dx_np[0], cmap='gray')
        plt.show()

        # Testing is_pixel_in_box function
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlay = image.copy()
        
        # Iterate over each pixel in the image
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if self.is_pixel_in_box(image_name, x, y):
                    overlay[y, x] = [255, 0, 0]  # Red for inside box

        # Blend the original image with the overlay
        alpha = 0.25  # Transparency factor
        output_image = cv2.addWeighted(image_rgb, 1 - alpha, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), alpha, 0)

        # Display the original and the output image
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image_rgb)
        plt.title('Original Image')
        
        plt.subplot(1, 2, 2)
        plt.imshow(output_image)
        plt.title('Overlay Image')

        plt.show()

    # I  standardize each image because each image has a different feature distribution, which will make it hard for our model to learn weights that generalize across all images
    def standardize(self, mat):
        return(mat - np.mean(mat)) / np.std(mat)
    
    def generate_image_features(self, image_name):
        image_name, image = self.get_image(image_name)

        # Canny
        image_canny_stand = self.standardize(self.get_canny_output(image))

        # Color Gradient
        blue_dx, blue_dy, green_dx, green_dy, red_dx, red_dy = self.get_color_gradients(image)
        
        blue_dx_stand = self.standardize(blue_dx)
        blue_dy_stand = self.standardize(blue_dy)
        green_dx_stand = self.standardize(green_dx)
        green_dy_stand = self.standardize(green_dy)
        red_dx_stand = self.standardize(red_dx)
        red_dy_stand =  self.standardize(red_dy)

        # Gradient Direction
        blue_direction = self.get_direction(blue_dx, blue_dy)
        green_direction = self.get_direction(green_dx, green_dy)
        red_direction = self.get_direction(red_dx, red_dy)


        # Texture Feature Vectors and Gradients
        blue_text_stand, green_text_stand, red_text_stand = self.get_gabor_feature_vector(image)

        blue_text_dx_stand, blue_text_dy_stand, green_text_dx_stand, green_text_dy_stand, red_text_dx_stand, red_text_dy_stand = self.get_gabor_feature_gradients(blue_text_stand, green_text_stand, red_text_stand)

        for i in range(blue_text_stand.shape[0]):
            blue_text_stand[i, :, :] = self.standardize(blue_text_stand[i, :, :])
            green_text_stand[i, :, :] = self.standardize(green_text_stand[i, :, :])
            red_text_stand[i, :, :] = self.standardize(red_text_stand[i, :, :])

        for i in range(blue_text_dx_stand.shape[0]):
            blue_text_dx_stand[i, :, :] = self.standardize(blue_text_dx_stand[i, :, :])
            blue_text_dy_stand[i, :, :] = self.standardize(blue_text_dy_stand[i, :, :])

            green_text_dx_stand[i, :, :] = self.standardize(green_text_dx_stand[i, :, :])
            green_text_dy_stand[i, :, :] = self.standardize(green_text_dy_stand[i, :, :])

            red_text_dx_stand[i, :, :] = self.standardize(red_text_dx_stand[i, :, :])
            red_text_dy_stand[i, :, :] = self.standardize(red_text_dy_stand[i, :, :])

        image_per_pixel_feats = np.empty((image_height * image_width, 86))
        
        # Color
        blue_stand = self.standardize(image[:, :, 0])
        green_stand = self.standardize(image[:, :, 1])
        red_stand = self.standardize(image[:, :, 2])

        # Creating features
        curr_idx = 0

        for y in range(image_height):
            for x in range(image_width):
                curr_y = 1 if self.is_pixel_in_box(image_name, x, y) else 0 

                pixel_features = np.array([red_stand[y, x], green_stand[y, x], blue_stand[y, x], 
                                           image_canny_stand[y, x], 
                                           red_dx_stand[y, x], red_dy_stand[y, x], green_dx_stand[y, x], green_dy_stand[y, x], blue_dx_stand[y, x], blue_dy_stand[y, x], 
                                           red_direction[y, x], green_direction[y, x], blue_direction[y, x], 
                                           red_text_stand[0, y, x], red_text_stand[1, y, x], red_text_stand[2, y, x], red_text_stand[3, y, x], red_text_stand[4, y, x], red_text_stand[5, y, x], red_text_stand[6, y, x], red_text_stand[7, y, x],
                                           blue_text_stand[0, y, x], blue_text_stand[1, y, x], blue_text_stand[2, y, x], blue_text_stand[3, y, x], blue_text_stand[4, y, x], blue_text_stand[5, y, x], blue_text_stand[6, y, x], blue_text_stand[7, y, x],
                                           green_text_stand[0, y, x], green_text_stand[1, y, x], green_text_stand[2, y, x], green_text_stand[3, y, x], green_text_stand[4, y, x], green_text_stand[5, y, x], green_text_stand[6, y, x], green_text_stand[7, y, x],
                                           red_text_dx_stand[0, y, x], red_text_dx_stand[1, y, x], red_text_dx_stand[2, y, x], red_text_dx_stand[3, y, x], red_text_dx_stand[4, y, x], red_text_dx_stand[5, y, x], red_text_dx_stand[6, y, x], red_text_dx_stand[7, y, x],
                                           blue_text_dx_stand[0, y, x], blue_text_dx_stand[1, y, x], blue_text_dx_stand[2, y, x], blue_text_dx_stand[3, y, x], blue_text_dx_stand[4, y, x], blue_text_dx_stand[5, y, x], blue_text_dx_stand[6, y, x], blue_text_dx_stand[7, y, x],
                                           green_text_dx_stand[0, y, x], green_text_dx_stand[1, y, x], green_text_dx_stand[2, y, x], green_text_dx_stand[3, y, x], green_text_dx_stand[4, y, x], green_text_dx_stand[5, y, x], green_text_dx_stand[6, y, x], green_text_dx_stand[7, y, x],
                                           red_text_dy_stand[0, y, x], red_text_dy_stand[1, y, x], red_text_dy_stand[2, y, x], red_text_dy_stand[3, y, x], red_text_dy_stand[4, y, x], red_text_dy_stand[5, y, x], red_text_dy_stand[6, y, x], red_text_dy_stand[7, y, x],
                                           blue_text_dy_stand[0, y, x], blue_text_dy_stand[1, y, x], blue_text_dy_stand[2, y, x], blue_text_dy_stand[3, y, x], blue_text_dy_stand[4, y, x], blue_text_dy_stand[5, y, x], blue_text_dy_stand[6, y, x], blue_text_dy_stand[7, y, x],
                                           green_text_dy_stand[0, y, x], green_text_dy_stand[1, y, x], green_text_dy_stand[2, y, x], green_text_dy_stand[3, y, x], green_text_dy_stand[4, y, x], green_text_dy_stand[5, y, x], green_text_dy_stand[6, y, x], green_text_dy_stand[7, y, x],
                                           curr_y])
                
                image_per_pixel_feats[curr_idx, :] = pixel_features

                curr_idx += 1
        
        return torch.from_numpy(image_per_pixel_feats)