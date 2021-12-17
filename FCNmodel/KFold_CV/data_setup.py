"""Steps:
1. make k amount of dataloaders
2. loop over range(k) with:
    1 assign i'th dataloader as test datasets
    2 assign the others as training datasets
    3 train on train data
    4 test on test data
    5 calculate loss
3. average all the losses
"""
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

from sklearn.model_selection import KFold


# directories set-up
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
prepdir = os.path.join(parentdir, "Preprocessing")
modeldir = os.path.join(parentdir, "Model")

sys.path.append(prepdir)
sys.path.append(parentdir)
sys.path.append(modeldir)

# make dataloaders
class Dataloading:
    """Creates K dataloaders.
    Args:
        K: the amount of dataloaders
        array_path: path to the folder containing the arrayfiles per slice.
        max_zoom: the maximum amount of zoom
        padding: the size of the largest image
        batch_size: size of the batches
        shuffle: "True" to enable shuffle, "False" to disable shuffle
    """

    def __init__(self, number_of_K, array_path, max_zoom=10, padding=(264, 288), batch_size=4, shuffle=False) -> None:
        self.number_of_K = number_of_K
        self.array_path = array_path
        self.max_zoom = max_zoom
        self.padding = padding
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.create_dicts()
        self.create_transforms()
        self.create_dataloaders()

    def create_dicts(self):
        for K in self.number_of_K:
            # create dicts
        self.data_dict = create_indexed_file_dict(self.array_path)
        self.train_data_dict = {key: self.data_dict[key] for i, key in enumerate(self.data_dict.keys()) if i < (1-self.test_size)*len(self.data_dict)}
        self.test_data_dict = {key: self.data_dict[key] for i, key in enumerate(self.data_dict.keys()) if i >= (1-self.test_size)*len(self.data_dict)}

    def create_transforms(self):
        randomzoom = RandomZoom(self.max_zoom)
        padder = PadImage(self.padding)
        sudorgb_converter = SudoRGB()
        to_tensor = ToTensor()
        normalizer = Normalizer()
        encoder = OneHotEncoder()
        self.composed_transform = transforms.Compose([normalizer, padder, sudorgb_converter, to_tensor])
    
    def create_dataloaders(self):
        self.train_slicedata = SliceDataset(self.array_path, self.train_data_dict, transform=self.composed_transform)
        self.test_slicedata = SliceDataset(self.array_path, self.test_data_dict, transform=self.composed_transform)

        self.train_dataloader = data.DataLoader(self.train_slicedata, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4)
        self.test_dataloader = data.DataLoader(self.test_slicedata, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4)
    
    def __iter__(self):
        pass

    def __next__(self):
        pass

    @staticmethod
    def remove_padding(batch) -> list:
        """Removes the padding from a batch and returns them as a list of dictionaries.
        The batch will not be a stacked anymore.\n
        Args:
            batch: a single batch loaded from a dataloader using the SliceDataset.
        """
    
        img_b, lbl_b, size_b = batch["image"], batch["label"], batch["size"]
        samples = []
        pad_deleter = RemovePadding()
        for i in range(img_b.size(dim=0)):
            sample = {"image": img_b[i,:,:,:], "label": lbl_b[i,:,:], "size": size_b[i,:]}
            samples.append(pad_deleter(sample))
        return samples
    


