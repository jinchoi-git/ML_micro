#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 16:19:30 2022

@author: jin
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
def check_accuracy(loader, model, loss_fn, device="cuda"):
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.long().to(device=device)
            output = model(x)
            loss = loss_fn(output, y)
            
            output = output.long()
            pred = torch.argmax(output, dim=1)
            pred = pred.long()
            acc = (pred == y).float().mean()
            accc = acc.to("cpu").detach().numpy()
    print(f"ACCURACY = {accc}")
    model.train()
    return accc, loss.item()

def save_predictions_as_npys(loader, model, BATCH_SIZE, folder="saved_npy/", device="cpu"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device, dtype=torch.float)
        with torch.no_grad():
            output = model(x)
            preds = torch.argmax(output, dim=1)
            predss = preds.to("cpu").detach().numpy()#.squeeze(0)
            yy = y.to("cpu").detach().numpy()
          
        for i in range(BATCH_SIZE):
            np.save(os.path.join(folder, f'preds_{idx}_{i}.npy'), predss[i])
            np.save(os.path.join(folder, f'y_{idx}_{i}.npy'), yy[i])
    model.train()

def save_npys_as_imgs(work_dir="/work_dir"):
    npfilepath = os.path.join(work_dir, "saved_npy")
    imagefilepath = os.path.join(work_dir, "saved_img")

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

        plt.figure(1)
        plt.subplot(121)
        plt.imshow(y)
        plt.subplot(122)
        plt.imshow(pred)
        plt.savefig(os.path.join(imagefilepath, f"pred_{i}"))
        plt.close('all')
        print(f"saved image {i+1}/{len(ys)}")
        
def save_loss_acc_plot(losstrains, lossvals, accs, work_dir="/work_dir"):
    plt.figure(1, figsize=(5,5))
    plt.plot(losstrains, color='r', label='Training loss')
    plt.plot(lossvals, color='g', label='Validation loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.legend()
    plt.savefig(os.path.join(work_dir, "loss_hist"))

    plt.figure(2, figsize=(5,5))
    plt.plot(accs, color='b', label='Pixel accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy History")
    plt.legend()
    plt.savefig(os.path.join(work_dir, "accuracy_hist"))
    plt.close('all')
        
    np.save(os.path.join(work_dir, "training_loss.npy"), losstrains)
    np.save(os.path.join(work_dir, "validation_loss.npy"), lossvals)
    np.save(os.path.join(work_dir, "accuracy.npy"), accs)
