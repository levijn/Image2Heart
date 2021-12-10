import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.dirname(current_dir))

import config
from preprocess import get_filenames, create_indexed_file_dict
from torch.utils import data


class SliceDataset(data.Dataset):
    def __init__(self, data_dir, idx_dict, transform=None) -> None:
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


def main():
    array_path = os.path.join(config.data_dir, "slice_arrays")
    data_dict = create_indexed_file_dict(array_path)
    slicedataset = SliceDataset(array_path, data_dict)


if __name__ == '__main__':
    main()