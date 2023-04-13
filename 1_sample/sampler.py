# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 21:33:02 2022

@author: jinch
"""

import os
import sys
import numpy as np
from yaml_parser import pf_parse

os.environ["CUDA_VISIBLE_DEVICES"]="0"
MAIN_DIR = "ML_micro/1_sample"
pf_args = pf_parse(os.path.join(MAIN_DIR, '3_track.yaml'))

IDS = [1, 2, 3, 4, 5]
QS = [20,22,24,26,28,30]
Nx = 464
Ny = 139
Nz = 46        
t_skip = 100
cell_size = 1e-3/464.0

for ID in IDS:
    for Q in QS:
        q = Q
        pf_args['id'] = ID
        case_name = f"ID{ID}_Q{Q}"
        ind_path = f"jax-am/applications/phase_field/multi_scan/data/{case_name}/numpy/pf/sols"
        
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
            #combined = np.hstack((ts[:, None], xs[:, None], ys[:, None], zs[:, None], ps[:, None], mov_dir))
            # print(combined)
            return ts, xs, ys, zs, ps, mov_dir
             
        def rve_picker(x, y, z, large_array):
            rve = np.zeros((80, 80, 32))
            # rve = large_array[x-57:x+18, y-27:y+28, z-28:z]
            rve = large_array[x-59:x+21, y-40:y+40, z-32:z]
            return rve
        
        #initialize here
        inds = []
        Ts = []
        ts, xs, ys, zs, ps, mov_dir = read_path(pf_args)
        
        for file in os.listdir(ind_path):
            if file.startswith("cell_ori_"):
                inds.append(file)
            if file.startswith("T_"):
                Ts.append(file)
        inds = sorted(inds)
        Ts = sorted(Ts)

        #loop start here
        for i, ind in enumerate(inds[:-1]):
            T0 = np.load(os.path.join(ind_path, Ts[i]))
            T1 = np.load(os.path.join(ind_path, Ts[i+1]))
            T0 = T0.reshape((Nx, Ny, Nz), order="F")
            T1 = T1.reshape((Nx, Ny, Nz), order="F")
            
            I0 = np.load(os.path.join(ind_path, inds[i]))
            I1 = np.load(os.path.join(ind_path, inds[i+1]))
            I0 = I0.reshape((Nx, Ny, Nz), order="F")
            I1= I1.reshape((Nx, Ny, Nz), order="F")
            
            x = round(xs[i*t_skip]/cell_size)
            y = round(ys[i*t_skip]/cell_size)
            z = round(zs[i*t_skip]/cell_size)
            print(f"{x}, {y}, {z}")
                 
            T0 = rve_picker(x, y, z, T0)
            T1 = rve_picker(x, y, z, T1)
            I0 = rve_picker(x, y, z, I0)
            I1 = rve_picker(x, y, z, I1)
    
            I0 = I0[None, :, :, :]
            T0 = T0[None, :, :, :]
            T1 = T1[None, :, :, :]
            
            image = np.concatenate((I0, T0, T1), axis=0)
            # mask = I1
            np.save(os.path.join(MAIN_DIR, "saved_npy", f"{case_name}_image_{i+1}"), image.astype(np.float32))
            np.save(os.path.join(MAIN_DIR, "saved_npy", f"{case_name}_mask_{i+1}"), I1.astype(np.int8))
            #print(f"saving {i}/{len(inds)}")

