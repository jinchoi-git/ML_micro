#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 16:15:42 2022

@author: jin
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as TNF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=20, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Down part of UNET
        self.downs.append(DoubleConv(in_channels, 64))
        self.downs.append(DoubleConv(64, 128))
        self.downs.append(DoubleConv(128, 256))
        self.downs.append(DoubleConv(256, 512))
        
        # Up part of UNET
        self.ups.append(nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2))
        self.ups.append(DoubleConv(1024, 512))
        self.ups.append(nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2))
        self.ups.append(DoubleConv(512, 256))
        self.ups.append(nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2))
        self.ups.append(DoubleConv(256, 128))
        self.ups.append(nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2))
        self.ups.append(DoubleConv(128, 64))

        # middle part
        self.bottleneck = DoubleConv(512, 1024)
        
        # final part
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        attach_array = []

        for down in self.downs:
            x = down(x)
            attach_array.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        attach_array = attach_array[::-1] #reverse list

        for idx in range(0, len(self.ups), 2): 
            x = self.ups[idx](x) #upsample
            attach = attach_array[idx//2] #get skip connections by 1 index

            if x.shape != attach.shape:
                # x = TF.resize(x, size=attach.shape[2:])
                x = TNF.interpolate(x, size=attach.shape[2:])

            concat_skip = torch.cat((attach, x), dim=1) #attach the skip connect
            x = self.ups[idx+1](concat_skip) #do double conv

        return self.final_conv(x)       
