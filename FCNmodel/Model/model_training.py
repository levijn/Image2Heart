
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



current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

sys.path.append(os.path.dirname(current_dir))

import config

data_dir = config.data_dir
preprocessing_dir = config.preprocessing_dir
arrays_dir = os.path.join(data_dir, "slice_arrays")
images_dir = os.path.join(data_dir, "slice_images")
print(arrays_dir, images_dir)

sys.path.insert(1, preprocessing_dir)

from preprocess import plot_slice_with_lbl
from slicedataset import main




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(data_dir)