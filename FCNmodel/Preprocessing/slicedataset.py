"""
Custom dataset class used by the DataLoader class.
"""

import os
import sys
import inspect
import numpy as np
from torch.utils import data

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
        
        org_size = img_array.shape[:2]
        print(org_size)
        
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


def main():
    array_path = os.path.join(config.data_dir, "slice_arrays")
    
    data_dict = create_indexed_file_dict(array_path)
    
    # Use to calculate h and w for the image padding. Only run again if data changes.
    # heights, widths = get_all_shapes_hw(array_path, data_dict)
    # print(max(heights), max(widths))
    
    padding = 428, 512
    
    padder = PadImage(padding)
    slicedata = SliceDataset(array_path, data_dict, transform=padder)

    dataloader = data.DataLoader(slicedata, batch_size=4, shuffle=True, num_workers=0)
    
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
          sample_batched['label'].size(),
          sample_batched["size"])
    
    


if __name__ == '__main__':
    main()