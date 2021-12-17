# import stuff
import torch
import os
import inspect
import numpy as np
import PIL
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50
import torchvision.transforms.functional as F
from torchvision.transforms.functional import convert_image_dtype, resize
from torchvision.transforms import Resize
from torchvision.utils import make_grid
from torchvision.io import read_image


# directories set-up
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
prepdir = os.path.join(parentdir, "Preprocessing")
modeldir = os.path.join(parentdir, "Model")

sys.path.append(prepdir)
sys.path.append(parentdir)
sys.path.append(modeldir)

# import stuff without Kfold
from slicedataset import Dataloading
from model_training import training_model, running_model

# preparing data for KFold