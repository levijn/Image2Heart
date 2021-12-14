
# importing stuff
from torchvision import transforms, datasets, models
from torchvision.models.segmentation import fcn_resnet50
import torch
import os
import inspect
import numpy as np
import PIL
import matplotlib.pyplot as plt
import sys
import time
import copy

import torchvision.transforms.functional as F
from torchvision.transforms.functional import convert_image_dtype, resize
from torchvision.transforms import Resize

import torch.optim as optim 
from torch.optim import lr_scheduler

from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path

# making directories
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.dirname(current_dir))

from config import data_dir, preprocessing_dir
sys.path.append(preprocessing_dir)

arrays_dir = os.path.join(data_dir, "slice_arrays")


# veel te lang mee zitten kloten om dit uit preprocess.py te halen dus nu maar gekopieerd
def get_filenames(directory) -> list:
    """Returns a list of all the filenames in the directory"""
    for (_, _, filenames) in os.walk(directory):
        return filenames


# inserting data
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'validation':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ]),
}


# pathing for training files and validation files aren't made, so this won't yet (dictionaries are empty)
image_datasets = {
    'train': 
    get_filenames(os.path.join(arrays_dir, "testing_set")),
    'validation': 
    get_filenames(os.path.join(arrays_dir, "validation_set"))
}


dataloaders = {
    'train':
    torch.utils.data.DataLoader(image_datasets['train'],
                                batch_size=32,
                                shuffle=True,
                                num_workers=0), 
    'validation':
    torch.utils.data.DataLoader(image_datasets['validation'],
                                batch_size=32,
                                shuffle=False,
                                num_workers=0)
}


# creating network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True).to(device)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2)).to(device)

# train model