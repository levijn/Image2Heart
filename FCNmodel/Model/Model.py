# importing stuff
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50
import torch
import os
import inspect
import numpy as np
import PIL
import matplotlib.pyplot as plt
import sys

import torchvision.transforms.functional as F
from torchvision.transforms.functional import convert_image_dtype, resize
from torchvision.transforms import Resize
        
from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

sys.path.append(os.path.dirname(current_dir))

import config
data_dir = config.data_dir
preprocessing_dir = config.preprocessing_dir
print(preprocessing_dir)
sys.path.insert(1, preprocessing_dir)



from preprocess import (get_filenames, 
                        create_indexed_file_dict,
                        load_slice_array,
                        plot_slice_with_lbl,
                        get_all_shapes_hw)
