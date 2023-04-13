#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 21:55:46 2022

@author: jin
"""

import numpy as np
import os
import meshio

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def np_perm():
    
    imagedir = "ML_micro/2_train/data/train_images"
    maskdir = "ML_micro/2_train/data/train_masks"
    
    images = []
    masks = []
    for file in os.listdir(imagedir):
        if file.endswith(".npy"):
            images.append(file)
    for file in os.listdir(maskdir):
        if file.endswith(".npy"):
            masks.append(file)
    images = sorted(images)
    masks = sorted(masks)
    
    num_aug = 19

    for imagefilename in images:
        imagepath = os.path.join(imagedir, imagefilename)
        image = np.load(imagepath)
        
        for i in range(num_aug):
            image0 = image == 0
            image1 = image == 1
            image2 = image == 2
            image3 = image == 3
            image4 = image == 4
            image5 = image == 5
            image6 = image == 6
            image7 = image == 7
            image8 = image == 8
            image9 = image == 9
            image10 = image == 10
            image11 = image == 11
            image12 = image == 12
            image13 = image == 13
            image14 = image == 14
            image15 = image == 15
            image16 = image == 16
            image17 = image == 17
            image18 = image == 18
            image19 = image == 19
            image[image0] = 1
            image[image1] = 2
            image[image2] = 3
            image[image3] = 4
            image[image4] = 5 
            image[image5] = 6 
            image[image6] = 7
            image[image7] = 8 
            image[image8] = 9
            image[image9] = 10
            image[image10] = 11
            image[image11] = 12
            image[image12] = 13
            image[image13] = 14
            image[image14] = 15
            image[image15] = 16
            image[image16] = 17
            image[image17] = 18
            image[image18] = 19
            image[image19] = 0
            np.save(os.path.join(imagedir, f"{imagefilename.split('.')[0]}_{i}.npy"), image)
            print(f"image {i} saved")

    for maskfilename in masks:
        maskpath = os.path.join(maskdir, maskfilename)
        mask = np.load(maskpath)
        
        for i in range(num_aug):
            mask0 = mask == 0
            mask1 = mask == 1
            mask2 = mask == 2
            mask3 = mask == 3
            mask4 = mask == 4
            mask5 = mask == 5
            mask6 = mask == 6
            mask7 = mask == 7
            mask8 = mask == 8
            mask9 = mask == 9
            mask10 = mask == 10
            mask11 = mask == 11
            mask12 = mask == 12
            mask13 = mask == 13
            mask14 = mask == 14
            mask15 = mask == 15
            mask16 = mask == 16
            mask17 = mask == 17
            mask18 = mask == 18
            mask19 = mask == 19
            mask[mask0] = 1
            mask[mask1] = 2
            mask[mask2] = 3
            mask[mask3] = 4
            mask[mask4] = 5 
            mask[mask5] = 6 
            mask[mask6] = 7
            mask[mask7] = 8 
            mask[mask8] = 9
            mask[mask9] = 10
            mask[mask10] = 11
            mask[mask11] = 12
            mask[mask12] = 13
            mask[mask13] = 14
            mask[mask14] = 15
            mask[mask15] = 16
            mask[mask16] = 17
            mask[mask17] = 18
            mask[mask18] = 19
            mask[mask19] = 0
            np.save(os.path.join(maskdir, f"{maskfilename.split('.')[0]}_{i}.npy"), mask)
            print(f"mask {i} saved")

if __name__ == "__main__":
    np_perm()
