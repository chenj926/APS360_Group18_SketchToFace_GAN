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
from torch.utils.tensorboard import SummaryWriter


#Load dataset with ImageFolder
size=64
batch_size=32
mean = 0.5
std = 0.5 #normalize all pixel tensor vals between -1 to 1

train_data_transform = transforms.Compose([
    transforms.Resize(size),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

train_dataset = ImageFolder(root='C:/Users/gaoan/Desktop/FS2K-main/tools/FS2K/train/photo',
                            transform = train_data_transform)

train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers = os.cpu_count(),
                              pin_memory = True)


