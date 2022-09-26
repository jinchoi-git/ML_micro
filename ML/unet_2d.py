import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="0"

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
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
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        self.downs.append(DoubleConv(in_channels, 64))
        self.downs.append(DoubleConv(64, 128))
        self.downs.append(DoubleConv(128, 256))
        self.downs.append(DoubleConv(256, 512))
        
        # Up part of UNET
        self.ups.append(nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2))
        self.ups.append(DoubleConv(1024, 512))
        self.ups.append(nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2))
        self.ups.append(DoubleConv(512, 256))
        self.ups.append(nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2))
        self.ups.append(DoubleConv(256, 128))
        self.ups.append(nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2))
        self.ups.append(DoubleConv(128, 64))

        # middle part
        self.bottleneck = DoubleConv(512, 1024)
        
        # final part
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

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
                x = TF.resize(x, size=attach.shape[2:])

            concat_skip = torch.cat((attach, x), dim=1) #attach the skip connect
            x = self.ups[idx+1](concat_skip) #do double conv

        return self.final_conv(x)

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

        image[0,:,:] = image[0,:,:]/19
        image[1:,:,:] = image[1:,:,:]-309
        image[1:,:,:] = image[1:,:,:]/1691
        
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

# print("classes loaded")

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
BATCH_SIZE = 50
NUM_EPOCHS = 20
IMAGE_HEIGHT = 93
IMAGE_WIDTH = 93  
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "/home/jyc3887/my_unet_1.1/data/train_images/"
TRAIN_MASK_DIR = "/home/jyc3887/my_unet_1.1/data/train_masks/"
VAL_IMG_DIR = "/home/jyc3887/my_unet_1.1/data/val_images/"
VAL_MASK_DIR = "/home/jyc3887/my_unet_1.1/data/val_masks/"
   
# print("hyperparameters loaded")

# train_transform = A.Compose([ToTensorV2()])
# val_transforms = A.Compose([ToTensorV2()])
 
train_ds = PolyDataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR)
val_ds = PolyDataset(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# print("ds and loaders loaded")

def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)
    # print("loop start")
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE, dtype=torch.float)
        targets = targets.squeeze(1).long().to(device=DEVICE)
        
        # forward
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
    return loss.item()
     

model = UNET(in_channels=12, out_channels=20).to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# scaler = torch.cuda.amp.GradScaler()

def save_predictions_as_imgs(loader, model, folder="/home/jyc3887/my_unet_1.1/saved_npy/", device="cpu"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device, dtype=torch.float)
        with torch.no_grad():
            output = model(x)
            preds = torch.argmax(output, dim=1)
            predss = preds.to("cpu").detach().numpy()#.squeeze(0)
            yy = y.to("cpu").detach().numpy()

            
        for i in range(BATCH_SIZE):
            np.save(f'/home/jyc3887/my_unet_1.1/saved_npy/preds_{idx}_{i}.npy', predss[i])
            np.save(f'/home/jyc3887/my_unet_1.1/saved_npy/y_{idx}_{i}.npy', yy[i])
        # print(f"saved predictions idx {i}")
    model.train()

def check_accuracy(loader, model, loss_fn, device="cuda"):
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.long().to(device=DEVICE)
            output = model(x)
            loss = loss_fn(output, y)
            
            output = output.long()
            pred = torch.argmax(output, dim=1)
            pred = pred.long()
            # print(f"y shape is {y.shape}")
            # print(f"preds shape is {pred.shape}")
            # num_correct = (preds == y).sum()/BATCH_SIZE
            # num_pixels = torch.numel(preds)/BATCH_SIZE
            # acc = num_correct/num_pixels
            acc = (pred == y).float().mean()
            accc = acc.to("cpu").detach().numpy()
    print(f"ACCURACY = {accc}")
    
    model.train()
    return accc, loss.item()
    

print("starting training loop")
losstrains = []
lossvals = []
accs = []
for epoch in range(NUM_EPOCHS):
    losstrain = train_fn(train_loader, model, optimizer, loss_fn)
    losstrains.append(losstrain)
    print(f"looping epoch {epoch}/{NUM_EPOCHS}")
    acc, lossval = check_accuracy(val_loader, model, loss_fn, device=DEVICE)
    accs.append(acc)
    lossvals.append(lossval)
    save_predictions_as_imgs(val_loader, model, folder="/home/jyc3887/my_unet_1.1/saved_npy/", device=DEVICE)
    print("saved predictions as imgs")

plt.figure(1, figsize=(12,4))
plt.subplot(131)
plt.plot(losstrains, 'bo')
plt.subplot(132)
plt.plot(lossvals, 'go')
plt.subplot(133)
plt.plot(accs, 'ro')
plt.savefig("/home/jyc3887/my_unet_1.1/loss_acc")


# if __name__ == "__main__":
#     test()
