# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:26:33 2022

@author: Jin
"""
import os
import sys
import numpy as np
import torch
import meshio
from model import UNET
from yaml_parser import pf_parse

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device is {DEVICE}")

# load model
case_name = "large"
MAIN_DIR = "ML_micro/3_predict"
ML_filename = "model_state_dict"
model = UNET(in_channels=3, out_channels=20).cuda()
model.load_state_dict(torch.load(os.path.join(MAIN_DIR, ML_filename)))
model.eval()

def read_path(pf_args):
    x_corners = pf_args['laser_path']['x_pos']
    y_corners = pf_args['laser_path']['y_pos']
    z_corners = pf_args['laser_path']['z_pos']
    power_control = pf_args['laser_path']['switch'][:-1]

    ts, xs, ys, zs, ps, mov_dir = [], [], [], [], [], []
    t_pre = 0.
    for i in range(len(x_corners) - 1):
        moving_direction = np.array([x_corners[i + 1] - x_corners[i], 
                                      y_corners[i + 1] - y_corners[i],
                                      z_corners[i + 1] - z_corners[i]])
        traveled_dist = np.linalg.norm(moving_direction)
        unit_direction = moving_direction/traveled_dist
        traveled_time = traveled_dist/pf_args['vel']
        ts_seg = np.arange(t_pre, t_pre + traveled_time, pf_args['dt'])
        xs_seg = np.linspace(x_corners[i], x_corners[i + 1], len(ts_seg))
        ys_seg = np.linspace(y_corners[i], y_corners[i + 1], len(ts_seg))
        zs_seg = np.linspace(z_corners[i], z_corners[i + 1], len(ts_seg))
        ps_seg = np.linspace(power_control[i], power_control[i], len(ts_seg))
        ts.append(ts_seg)
        xs.append(xs_seg)
        ys.append(ys_seg)
        zs.append(zs_seg)
        ps.append(ps_seg)
        mov_dir.append(np.repeat(unit_direction[None, :], len(ts_seg), axis=0))
        t_pre = t_pre + traveled_time

    ts, xs, ys, zs, ps, mov_dir = np.hstack(ts), np.hstack(xs), np.hstack(ys), np.hstack(zs), np.hstack(ps), np.vstack(mov_dir)  
    print(f"Total number of time steps = {len(ts)}")
    return ts, xs, ys, zs, ps, mov_dir

def get_T_laser(pf_args, centroids, x_laser, y_laser, z_laser, power, unit_mov_dir):
    """Analytic T from https://doi.org/10.1016/j.actamat.2021.116862
    """
    Q = 25 * power
    alpha = 5.2e-6
    kappa = 27
    X = centroids[:, 0] - x_laser
    Y = centroids[:, 1] - y_laser
    Z = centroids[:, 2] - z_laser
    R = np.sqrt(X**2 + Y**2 + Z**2)
    projection = X*unit_mov_dir[0] + Y*unit_mov_dir[1] + Z*unit_mov_dir[2]
    T = pf_args['T_ambient'] + Q / (2 * np.pi * kappa) / R * np.exp(-pf_args['vel'] / (2*alpha) * (R + projection))
    T = np.where(T > 2000., 2000., T)
    return T[:, None]

def rve_picker(x, y, z, large_array):
    rve = np.zeros((80, 80, 32))
    rve = large_array[x-40:x+40, y-40:y+40, z-32:z]
    return rve

def large_arr_updater(x, y, z, rve, large_array):
    large_array[x-40:x+40, y-40:y+40, z-32:z] = rve
    return large_array

def predictor(t_image):
    model.eval()
    x = t_image
    x[:, 0, :, :, :] = x[:, 0, :, :, :] 
    x[:, 1:, :, :, :] = x[:, 1:, :, :, :] 

    x = x.cuda()
    with torch.no_grad():
        output = model(x)
        preds = torch.argmax(output, dim=1)
        #predss = preds.to("cpu").detach().numpy()
        #predss = predss.squeeze(0)
    return preds.squeeze(0)#, predss
    
# initialize
skip = 100
max_t = 13650
Nx = 464
Ny = 139 
Nz = 46 
cell_size = 1e-3/464.0

pf_args = pf_parse(os.path.join(MAIN_DIR, '3_track.yaml'))
centroids = np.load(os.path.join(MAIN_DIR, '808032_centroids.npy')) #for local temperature field
large_array = np.load(os.path.join(MAIN_DIR, 'large_initial.npy'))
large_array = large_array.reshape((Nx, Ny, Nz), order="F")
t_large_array = torch.Tensor(large_array)

t_image = torch.zeros(3, 80, 80, 32)
t_image = t_image[None, :, :, :, :]

ts, xs, ys, zs, ps, mov_dir = read_path(pf_args)
x = round(xs[0]/cell_size)
y = round(ys[0]/cell_size)
z = round((zs[0])/cell_size)

T0 = get_T_laser(pf_args, centroids, 0.0862e-3, 0.0862e-3, 0.0689655e-3, ps[0], mov_dir[0]) 
T1 = get_T_laser(pf_args, centroids, 0.0862e-3, 0.0862e-3, 0.0689655e-3, ps[0+skip], mov_dir[0+skip]) 
T0 = T0.reshape((80, 80, 32), order="F")
T1 = T1.reshape((80, 80, 32), order="F")

I0 = rve_picker(x, y, z, large_array)
I0 = I0[None, None, :, :, :]
T0 = T0[None, None, :, :, :]
T1 = T1[None, None, :, :, :]

image = np.concatenate((I0, T0, T1), axis=1)
t_image = torch.Tensor(image)
np.save(os.path.join(MAIN_DIR, 'saved_npy', f'large_array_start'), large_array)

print("start loop")
for i in range(0, max_t-skip, skip):
      
    x = round(xs[i]/cell_size)
    y = round(ys[i]/cell_size)
    z = round((zs[i])/cell_size)
    #print(f"x={x}, y={y}, z={z}")
    
    shift = mov_dir[i]/np.linalg.norm(mov_dir[i])*19
    shift = np.rint(shift)
    x = int(x - shift[0])
    y = int(y - shift[1])
    print(f"after shift x={x}, y={y}, z={z}")
            
    laser_mov = mov_dir[i]/np.linalg.norm(mov_dir[i])*1e-7
    laser_mov = np.rint(laser_mov)

    T0 = get_T_laser(pf_args, centroids, 0.0862e-3, 0.0862e-3, 0.0689655e-3, ps[i], mov_dir[i]) 
    T1 = get_T_laser(pf_args, centroids, 0.0862e-3+laser_mov[0], 0.0862e-3+laser_mov[1], 0.0689655e-3, ps[i+skip], mov_dir[i+skip]) 
    T0 = T0.reshape((80, 80, 32), order="F")
    T1 = T1.reshape((80, 80, 32), order="F")
    t_Tr0 = torch.Tensor(T0)
    t_Tr1 = torch.Tensor(T1)    
    t_I0 = rve_picker(x, y, z, t_large_array)
   
    t_image[:, 0, :, :, :] = t_I0 /19.0 #normalization
    t_image[:, 1, :, :, :] = (t_Tr0-300) /1700.0
    t_image[:, 2, :, :, :] = (t_Tr1-300) /1700.0
   
    preds = predictor(t_image)
    t_large_array = large_arr_updater(x, y, z, preds, t_large_array)
    large_array = t_large_array.to('cpu').detach().numpy()
    
    if i % 10000 == 0:
        np.save(os.path.join(MAIN_DIR, 'saved_npy', f'{case_name}_{i:08d}'), large_array.astype(np.int8))

print("done all")