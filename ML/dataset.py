#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 16:14:14 2022

@author: jin
"""
import os
import torch
from torch.utils.data import Dataset
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

        #image[0,:,:] = image[0,:,:]/19
        #image[1:,:,:] = image[1:,:,:]-300
        #image[1:,:,:] = image[1:,:,:]/1691
        
        image[0,:,:,:] = image[0,:,:,:]/20
        image[1:,:,:,:] = image[1,:,:,:]-299.9
        image[1:,:,:,:] = image[1:,:,:,:]/1700
        
        # image = np.swapaxes(image,0,1)
        # image = np.swapaxes(image,1,2)
        
        t_image = torch.Tensor(image)
        t_mask = torch.Tensor(mask)
        
        # print(f"image shape {image.shape}")
        # print(f"mask shape {mask.shape}")
       
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"] 

        return t_image, t_mask
