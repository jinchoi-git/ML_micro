#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 16:14:14 2022

@author: jin
"""
import os
import torch
from torch.utils.data import Dataset
import random
from scipy import ndimage
import numpy as np

class PolyDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.images = sorted(self.images)
        self.masks = os.listdir(mask_dir)
        self.masks = sorted(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        image = np.load(img_path)
        mask = np.load(mask_path)
        #print(f"image shape {image.shape}")
        #print(f"mask shape {mask.shape}")
        
        image_fname = self.images[index].split('.')[-2]
        mask_fname = self.masks[index].split('.')[-2]
        
        image[0,:,:,:] = image[0,:,:,:]/19.0
        image[1:,:,:,:] = (image[1,:,:,:]-300.0)/1700.0
        
        angle = random.randint(0, 359)
        #angle = random.choice([0, 180])
        #print(f"angle is {angle}")

        rotated_image = np.zeros((3,80,80,32))
        rotated_image[0,:,:,:] = ndimage.rotate(image[0,:,:,:], angle, axes=(0, 1), reshape=False, output=None, order=0, mode='nearest', prefilter=False)
        rotated_image[1,:,:,:] = ndimage.rotate(image[1,:,:,:], angle, axes=(0, 1), reshape=False, output=None, order=0, mode='nearest', prefilter=False)
        rotated_image[2,:,:,:] = ndimage.rotate(image[2,:,:,:], angle, axes=(0, 1), reshape=False, output=None, order=0, mode='nearest', prefilter=False)
        rotated_mask = ndimage.rotate(mask, angle, axes=(0, 1), reshape=False, output=None, order=0, mode='nearest', prefilter=False)

        t_image = torch.Tensor(rotated_image)
        t_mask = torch.Tensor(rotated_mask)
        # t_image = torch.Tensor(image)
        # t_mask = torch.Tensor(mask)
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"] 

        return t_image, t_mask, image_fname, mask_fname
