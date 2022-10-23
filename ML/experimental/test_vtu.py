#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 13:58:09 2022

@author: jin
"""
import os
import numpy as np
import torch
import meshio
import matplotlib.pyplot as plt


filename = "/home/jin/Documents/ML_micro/ML_micro/ML/experimental/fd_mesh.vtu"
mask_data = meshio.read(filename)
mask_data.cell_data['ori_ind'] = [np.random.randint(0, 20, size=(46,46,46)).flatten()]
meshio.write(os.path.join("/home/jin/Documents/ML_micro/ML_micro/ML/experimental/", "fd_mesh_with_ori.vtu"), mask_data)

# pred_data = meshio.read(filename)
# pred_dict = pred_data.cell_data
# pred_dict['ori_inds'][0] = pred
# pred_data.cell_data = pred_dict
# meshio.write(os.path.join(vtufilepath, f"pred_{i}.vtu"), pred_data)


test = meshio.read("/home/jin/Documents/ML_micro/ML_micro/ML/experimental/fd_mesh_with_ori.vtu")