import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_SKETCH_DIR = 'train/sketch'
TRAIN_TARGET_DIR = 'train/photo'
TRAIN_TRANS_SKETCH_DIR = 'train/transformed_sketch'
TRAIN_TRANS_TARGET_DIR = 'train/transformed_photo'


VAL_DIR = "data/val"
LEARNING_RATE = 2e-4
BETAS = (0.5, 0.999)
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
NUM_EPOCHS = 500
LOAD_MODEL = True
SAVE_MODEL = False
CHECKPOINT_DISC = "Pix2Pix_Weights_Colorize_Anime/disc.pth.tar"
CHECKPOINT_GEN = "Pix2Pix_Weights_Colorize_Anime/gen.pth.tar"

#transform on both sketch and target img
#like flip, resize, normalize etc..
#additional_targets is used to apply same transformation to multiple images (label, type)

'''
both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"target": "image"},
)
'''

both_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE, always_apply=True),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE,
            min_width=IMAGE_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
            always_apply=True
        ),
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
    ], additional_targets={"target": "image"},
)



#transform only apply to sketch img 
transform_only_sketch = A.Compose(
    [
        #A.HorizontalFlip(p=0.5),
        #A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

#transform only apply to target img
transform_only_tar = A.Compose(
    [
        
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)