# %%
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image #save a ggiven tensor into img file
from torchvision.utils import make_grid #make a grid of imgs
import matplotlib.pyplot as plt


# %%
# Define the path to the "train" folder
train_folder = os.path.normpath("C:/Users/gaoan/Desktop/FS2K-main/tools/FS2K/train")


size=256
batch_size=32
mean = 0.5
std = 0.5 #normalize all pixel tensor vals between -1 to 1


# Define the transformation to apply to the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size),
    transforms.CenterCrop(size),
    transforms.Normalize((0.5,), (0.5,))
])

# Create the dataset
train_dataset = ImageFolder(train_folder, transform=transform)

# Create the dataloader
train_dataloader = DataLoader(train_dataset, 
                        batch_size=batch_size, 
                        shuffle=True,
                        num_workers = os.cpu_count(),
                        pin_memory = True)


#get the shape of training imgs 
test_img = next(iter(train_dataloader))
print(test_img[0].shape, test_img[1].shape)
#sample output is 
# torch.Size([32, 3, 256, 256]) torch.Size([32])
# batch_size, channel, H, W


# %%
def denorm(img_tensors):
    return img_tensors * std + mean

def show_images(images, nmax=64):
  fig, ax = plt.subplots(figsize=(8, 8))
  ax.set_xticks([]); ax.set_yticks([])
  ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break

show_batch(train_dataloader)
# %%

