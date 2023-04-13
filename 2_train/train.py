import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
from dataset import PolyDataset
from model import UNET
from utils import (
    check_accuracy,
    save_predictions_as_npys,
    save_npys_as_imgs,
    save_loss_acc_plot,
    npy_to_vtu,
    )

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 50
NUM_EPOCHS = 10
PIN_MEMORY = True
LOAD_MODEL = False

# directories
MAIN_DIR = "ML_micro/2_train"
TRAIN_IMG_DIR = os.path.join(MAIN_DIR, "data/train_images")
TRAIN_MASK_DIR = os.path.join(MAIN_DIR, "data/train_masks")
VAL_IMG_DIR = os.path.join(MAIN_DIR, "data/val_images")
VAL_MASK_DIR = os.path.join(MAIN_DIR, "data/val_masks")

# train function
def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)
    fl_loss = []
    # print("loop start")
    for batch_idx, (data, targets, i_name, f_name) in enumerate(loop):
        data = data.to(device=DEVICE, dtype=torch.float)
        targets = targets.squeeze(1).long().to(device=DEVICE)
        
        # forward
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
        fl_loss.append(float(loss.item()))
    return fl_loss

# main function
def main():
    # get dataset, loaders
    train_ds = PolyDataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR)
    print(f"dataset length is {len(train_ds)}")
    val_ds = PolyDataset(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    # create work diretory
    work_dir = os.path.join(MAIN_DIR, "runs", f"LR{LEARNING_RATE}_BS{BATCH_SIZE}_NE{NUM_EPOCHS}_TS{len(train_ds)}_VS{len(val_ds)}")
    NPY_DIR = os.path.join(work_dir, "saved_npy")
    IMG_DIR = os.path.join(work_dir, "saved_img")
    VTU_DIR = os.path.join(work_dir, "saved_vtu")
    os.makedirs(NPY_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(VTU_DIR, exist_ok=True)
    
    model = UNET(in_channels=3, out_channels=20).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
       
    print("starting training loop")
    losstrains = []
    lossvals = []
    accs = []
    for epoch in range(NUM_EPOCHS):
        losstrain = train_fn(train_loader, model, optimizer, loss_fn)
        losstrains.append(losstrain)
        acc, lossval = check_accuracy(val_loader, model, loss_fn, device=DEVICE)
        accs.append(acc)
        lossvals.append(lossval)
        torch.save(model.state_dict(), os.path.join(work_dir, f"model_state_dict_{epoch}"))
        np.save(os.path.join(work_dir, f"losstrains_{epoch}.npy"), losstrains)
        np.save(os.path.join(work_dir, f"accs_{epoch}.npy"), accs)
        np.save(os.path.join(work_dir, f"lossvals_{epoch}.npy"), lossvals)
        print(f"looping epoch {epoch}/{NUM_EPOCHS}")
    
    # save everything
    save_predictions_as_npys(val_loader, model, BATCH_SIZE, folder=NPY_DIR, device=DEVICE)
    save_loss_acc_plot(losstrains, lossvals, accs, work_dir=work_dir)
    # save_npys_as_imgs(work_dir=work_dir)
    npy_to_vtu(work_dir=work_dir)
    # torch.save(model.state_dict(), os.path.join(work_dir, "model_state_dict"))

if __name__ == "__main__":
    main()
