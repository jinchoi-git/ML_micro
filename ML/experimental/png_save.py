#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:43:04 2022

@author: jin
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import os

imagedir = "/home/jin/Desktop/experimental/data"
maskdir = "/home/jin/Desktop/experimental/data"

images = []
masks = []

for file in os.listdir(imagedir):
    if file.startswith("image_"):
        images.append(file)
for file in os.listdir(maskdir):
    if file.startswith("mask"):
        masks.append(file)
images = sorted(images)
masks = sorted(masks)

for imagefilename in images:
    imagepath = os.path.join(imagedir, imagefilename)
    image = np.load(imagepath)
    
    ind0 = image[0,45,:,:]
    T0 = image[1,45,:,:]
    T1 = image[2,45,:,:]
    
    matplotlib.image.imsave(f"/home/jin/Desktop/experimental/data/images/{imagefilename}_ind0.png", ind0)
    matplotlib.image.imsave(f"/home/jin/Desktop/experimental/data/images/{imagefilename}_T0.png", T0)
    matplotlib.image.imsave(f"/home/jin/Desktop/experimental/data/images/{imagefilename}_T1.png", T1)
    
        
for maskfilename in masks:
    maskpath = os.path.join(maskdir, maskfilename)
    mask = np.load(maskpath)
    
    ind1 = mask[45,:,:]
    
    matplotlib.image.imsave(f"/home/jin/Desktop/experimental/data/images/{maskfilename}_ind1.png", ind1)
    
    
    