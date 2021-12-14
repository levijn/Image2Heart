from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torch.utils import data
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import sys
import copy
import inspect
import config

root_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
data_dir = os.path.join(os.path.dirname(os.path.dirname(root_dir)), "Data")
hyme_data_dir = os.path.join(data_dir, "hymenoptera_data")
slice_array = os.path.join(data_dir, "slice_arrays")
slice_images = os.path.join(data_dir, "slice_images")


from preprocess import (get_filenames, 
                        create_indexed_file_dict,
                        load_slice_array,
                        plot_slice_with_lbl,
                        get_all_shapes_hw)

from slicedataset import (SliceDataset, 
                        PadImage, 
                        SudoRGB, 
                        ToTensor)


plt.ion()

def run():
    array_path = os.path.join(config.data_dir, "slice_arrays")
    data_dict = create_indexed_file_dict(array_path)
    padding = 428, 512
    padder = PadImage(padding)
    sudorgb_converter = SudoRGB()
    to_tensor = ToTensor()
    composed_transform = transforms.Compose([padder,sudorgb_converter])
    slicedata = SliceDataset(array_path, data_dict, transform=composed_transform)
    slice = slicedata[4]


    image_datasets = datasets.ImageFolder(os.path.join(slice_images))
    dataloader1 = data.DataLoader(slicedata, batch_size=8, shuffle=True, num_workers=8)

    class_names = ('0', '1', '2', '3')

    print(dataloader1)




if __name__ == '__main__':
    run()    