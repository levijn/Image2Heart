"""
Contains a dataset of a single nifti file.
This can be used to precict the 3D image of a heart.
"""

import os
import sys
from pathlib import Path
import nibabel as nib
import numpy as np
from torch.utils import data
from torchvision import transforms


# Import the path of different folders
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

import config
from slicedataset import (RandomZoom,
                          PadImage,
                          SudoRGB,
                          ToTensor,
                          Normalizer)


class OnePatientDataset(data.Dataset):
    """Slices Dataset"""
    def __init__(self, nifti_file, nifti_label_file, transform=None) -> None:
        """
        Args:
            nifti_file (string): Path to the nifti file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.nifti_file = nifti_file
        self.nifti_lbl_file = nifti_label_file
        self.transform = transform
        
        self.slices = nib.load(self.nifti_file).get_fdata()
        self.labels = nib.load(self.nifti_lbl_file).get_fdata()
        
        
    
    def __len__(self):
        return self.slices.shape[2]

    def __getitem__(self, idx):
        #get filenames
        slice = self.slices[:,:,idx]
        lbl = self.labels[:,:,idx]
        org_size = np.asarray(slice.shape[:2])

        sample = {"image": slice,
                  "label": lbl,
                  "size": org_size}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def get_patient_dataloader(img_path, lbl_path):
    """Returns a dataloader with the right transforms"""
    randomzoom = RandomZoom(10)
    padder = PadImage((264, 288))
    sudorgb_converter = SudoRGB()
    to_tensor = ToTensor()
    normalizer = Normalizer()
    composed_transform = transforms.Compose([randomzoom, padder, sudorgb_converter, to_tensor, normalizer])
    
    dataset = OnePatientDataset(img_path, lbl_path, transform=composed_transform)
    dataloader = data.DataLoader(dataset, batch_size=16)
    
    return dataloader


def main():
    img_path = os.path.join(config.data_dir, "simpledata", "patient001_frame01.nii.gz")
    lbl_path = os.path.join(config.data_dir, "simpledata", "patient001_frame01_gt.nii.gz")
    
    randomzoom = RandomZoom(10)
    padder = PadImage((264, 288))
    sudorgb_converter = SudoRGB()
    to_tensor = ToTensor()
    normalizer = Normalizer()
    composed_transform = transforms.Compose([randomzoom, padder, sudorgb_converter, to_tensor, normalizer])
    
    dataset = OnePatientDataset(img_path, lbl_path, transform=composed_transform)
    dataloader = data.DataLoader(dataset, batch_size=16)
    
    for batch in dataloader:
        print(batch)
    
if __name__ == '__main__':
    main()