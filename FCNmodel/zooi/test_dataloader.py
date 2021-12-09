from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from nifti import MriFrame
from os import path, walk
import nibabel as nib
import json

class ACDCDataset(Dataset):
    """ACDC dataset"""
    
    def __init__(self, root_path, transform=None) -> None:
        super().__init__()
        self.root_path = root_path
        self.filenames = []
        for (dirpath, dirnames, filenames) in walk(root_path):
            self.filenames.extend(filenames)
            break
        
        self.transform = transform
        
        self.current_img_index = 1
        self.loaded_lbl_file = nib.load(path.join(root_path, filenames[self.current_img_index-1])).get_fdata()
        self.loaded_img_file = nib.load(path.join(root_path, filenames[self.current_img_index])).get_fdata()
        
        self.current_slice = 0
        self.last_slice_index = self.loaded_img_file.shape[2]
    
    def get_patient_str(self):
        patient_str = str(self.current_patient)
        while len(patient_str) < 3:
            patient_str = "0" + patient_str
        return patient_str
    
    def get_current_path(self):
        patient_num = self.get_patient_str()
        folder_path = "patient" + patient_num
    
    def __getitem__(self, idx):
        """get an item at a certain index"""
        pass
    
    def __len__(self):
        #TODO probably usefull to precalculate the amount of available slices.
        #TODO (otherwise all img have to be loaded to check the available slices)
        pass
        