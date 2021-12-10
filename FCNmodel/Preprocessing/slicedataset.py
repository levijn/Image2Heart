import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.dirname(current_dir))

import config
from preprocess import (get_filenames, 
                        create_indexed_file_dict,
                        load_slice_array,
                        plot_slice_with_lbl)
from torch.utils import data


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
        slice = self.idx_dict[idx]
        img_data_file = slice["img_data_file"]
        lbl_data_file = slice["lbl_data_file"]
        
        img_array = load_slice_array(os.path.join(self.data_dir, img_data_file))
        lbl_array = load_slice_array(os.path.join(self.data_dir, lbl_data_file))
        
        return img_array, lbl_array


def main():
    array_path = os.path.join(config.data_dir, "slice_arrays")
    data_dict = create_indexed_file_dict(array_path)
    slicedataset = SliceDataset(array_path, data_dict)
    img_array, lbl_array = slicedataset[40]
    plot_slice_with_lbl(img_array, lbl_array)
    


if __name__ == '__main__':
    main()