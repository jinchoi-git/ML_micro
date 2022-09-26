#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 13:21:56 2022

@author: jin
"""

import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="0"

npfilepath = "/home/jyc3887/my_unet_1.1/saved_npy/"
imagefilepath = "/home/jyc3887/my_unet_1.1/saved_img/"

images = os.listdir(npfilepath)

ys = []
preds = []
for file in os.listdir(npfilepath):
    if file.startswith("y"):
        ys.append(file)
    if file.startswith("preds"):
        preds.append(file)
ys = sorted(ys)
preds = sorted(preds)

for i in range(len(ys)):
    y = np.load(os.path.join(npfilepath, ys[i]))
    pred = np.load(os.path.join(npfilepath, preds[i]))
    # y = y*255.0/19.0
    # pred = pred*255.0/19.0

    plt.figure(1)
    plt.subplot(121)
    plt.imshow(y)
    plt.subplot(122)
    plt.imshow(pred)
    plt.savefig(os.path.join(imagefilepath, f"pred_{i}"))
    plt.close('all')
    print(f"saved image {i}/{len(ys)}")
    
# pred_image = Image.fromarray(pred, 'L')
    
    # y_image.save(os.path.join(imagefilepath, f"y_{i}.png"))
    # pred_image.save(os.path.join(imagefilepath, f"pred_{i}.png"))
