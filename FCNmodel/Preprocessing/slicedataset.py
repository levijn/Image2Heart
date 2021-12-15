"""
Custom dataset class used by the DataLoader class.
"""

import os
import sys
import inspect
import numpy as np
from torch.utils import data
from torchvision import transforms 
import torch
import torch.nn.functional as F
import random

# Import the path of different folders
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.dirname(current_dir))

import config
from preprocess import (get_filenames, 
                        create_indexed_file_dict,
                        load_slice_array,
                        plot_slice_with_lbl,
                        get_all_shapes_hw)


class SliceDataset(data.Dataset):
    """Slices Dataset"""
    def __init__(self, data_dir, idx_dict, transform=None) -> None:
        """
        Args:
            data_dir (string): Path to the folder with slice array files.
            root_dir (string): Dictonary with index pointing to a dictonary with "img_data_file" and "lbl_data_file"
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.data_dir = data_dir
        self.idx_dict = idx_dict
        self.transform = transform
    
    def __len__(self):
        return len(self.idx_dict)

    def __getitem__(self, idx):
        #get filenames
        slice = self.idx_dict[idx]
        img_data_file = slice["img_data_file"]
        lbl_data_file = slice["lbl_data_file"]
        
        #load files
        img_array = load_slice_array(os.path.join(self.data_dir, img_data_file))
        lbl_array = load_slice_array(os.path.join(self.data_dir, lbl_data_file))

        org_size = np.asarray(img_array.shape[:2])

        sample = {"image": img_array,
                  "label": lbl_array,
                  "size": org_size}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class PadImage(object):
    """Adds padding to the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size: tuple):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        image, label, size = sample["image"], sample["label"], sample["size"]

        h, w = size
        new_h, new_w = self.output_size
        h_pad = new_h - h
        w_pad = new_w - w
        
        new_image = np.pad(image, ((0,h_pad), (0,w_pad)), "constant", constant_values=(0,0))
        new_label = np.pad(label, ((0,h_pad), (0,w_pad)), "constant", constant_values=(0,0))
        
        return {"image": new_image, "label": new_label, "size": size}


class SudoRGB(object):
    def __init__(self):
        pass
    
    def __call__(self, sample):
        image, label, size = sample["image"], sample["label"], sample["size"]
        
        rgb_img = np.stack([image]*3, axis=0)
    
        return {"image": rgb_img, "label": label, "size": size}


class ToTensor(object):
    def __init__(self):
        pass
    
    def __call__(self, sample):
        image, label, size = sample["image"], sample["label"], sample["size"]
        
        if type(image) == type(np.ndarray):
            image = torch.from_numpy(image)
        if type(label) == type(np.ndarray):
            label = torch.from_numpy(label)
        if type(size) == type(np.ndarray):
            size = torch.from_numpy(size)
            
        return {"image": image, "label": label, "size": size}


class RemovePadding(object):
    """Removing the padding from images and labels"""    
    def __call__(self, sample):
        image, label, size = sample["image"], sample["label"], sample["size"]
        
        orig_img = torch.narrow(image, 1, 0, size[0])       #Deleting the padding rows from image
        orig_img = torch.narrow(orig_img, 2, 0, size[1])       #Deleting the padding columns from image
        
        orig_lbl = torch.narrow(label, 0, 0, size[0])       #Deleting the padding rows from label
        orig_lbl = torch.narrow(orig_lbl, 1, 0, size[1])       #Deleting the padding columns from label

        return {"image": orig_img, "label": orig_lbl, "size": size}


class OneHotEncoder(object):
    def __call__(self, sample):
        image, label, size = sample["image"], sample["label"], sample["size"]
        
        lbl = torch.from_numpy(label)
        onehot_label = F.one_hot(lbl.to(torch.int64), num_classes=4)
        
        return {"image": image, "label": onehot_label, "size": size}


class RandomZoom(object):
    def __init__(self, max_zoom):
        self.max_zoom = max_zoom
        
    
    def __call__(self, sample):
        image, label, size = sample["image"], sample["label"], sample["size"]
        zoom = 1-random.randint(0, self.max_zoom)/100
        
        new_h = int(zoom*size[0])
        new_w = int(zoom*size[1])
        
        h_del = size[0]-new_h
        w_del = size[1]-new_w
        
        new_image = image[int(h_del/2):int(new_h-h_del/2),int(w_del/2):int(new_w-w_del/2)]
        new_label = label[int(h_del/2):int(new_h-h_del/2),int(w_del/2):int(new_w-w_del/2)]
        
        return {"image": new_image, "label": new_label, "size": np.asarray([new_h, new_w])}

def main():
    array_path = os.path.join(config.data_dir, "slice_arrays")
    
    data_dict = create_indexed_file_dict(array_path)
    
    # Use to calculate h and w for the image padding. Only run again if data changes.
    # heights, widths = get_all_shapes_hw(array_path, data_dict)
    # print(max(heights), max(widths))
    
    padding = 428, 512
    
    randomzoom = RandomZoom(20)
    padder = PadImage(padding)
    sudorgb_converter = SudoRGB()
    remove_padding = RemovePadding()
    to_tensor = ToTensor()
    encoder = OneHotEncoder()
    composed_transform = transforms.Compose([sudorgb_converter, to_tensor])
    composed_zoomtransform = transforms.Compose([randomzoom, sudorgb_converter, to_tensor])
    
    slicedata = SliceDataset(array_path, data_dict, transform=composed_transform)
    zoomslicedata = SliceDataset(array_path, data_dict, transform=composed_zoomtransform)
    slice = slicedata[1]
    zoom_slice = zoomslicedata[600]
    print(slice["image"].shape, zoom_slice["image"].shape)

    plot_slice_with_lbl(zoom_slice["image"][1,:,:], zoom_slice["label"])
    
    dataloader = data.DataLoader(slicedata, batch_size=1, shuffle=False, num_workers=8)
    
    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch, sample_batched['image'].size(),
    #       sample_batched['label'].size(),
    #       sample_batched["size"])
    

if __name__ == '__main__':
    main()