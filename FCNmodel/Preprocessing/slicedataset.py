"""
Custom dataset class used by the DataLoader class.
Also contains the transforms.
"""

import os
import sys
import inspect
import numpy as np
from torch.utils import data
from torch.utils.data.dataset import ConcatDataset
from torchvision import transforms 
import torch
import torch.nn.functional as nnF
import torchvision.transforms.functional as tF
import torchvision.transforms as trans
import random
import matplotlib.pyplot as plt
from PIL import Image

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
        keys = self.idx_dict.keys()
        self.start_index = min(keys)
        self.transform = transform
    
    def __len__(self):
        return len(self.idx_dict)

    def __getitem__(self, idx):
        #get filenames
        slice = self.idx_dict[idx+self.start_index]
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
        # print("Padding...")
        image, label, size = sample["image"], sample["label"], sample["size"]
        # print(f"Padder: Initial size: {size}")
        # print(image.shape)
        h, w = (len(image), len(image[0]))
        new_h, new_w = self.output_size
        h_pad = int(new_h - h)
        w_pad = int(new_w - w)
        
        new_image = np.pad(image, ((0,h_pad), (0,w_pad)), "constant", constant_values=(0,0))
        new_label = np.pad(label, ((0,h_pad), (0,w_pad)), "constant", constant_values=(0,0))
        
        # print(f"Padder: Final size: {new_image.shape}")

        return {"image": new_image, "label": new_label, "size": np.array([h, w])}


class SudoRGB(object):
    def __call__(self, sample):
        # print("RGB...")
        image, label, size = sample["image"], sample["label"], sample["size"]
        
        rgb_img = np.stack([image]*3, axis=0)
    
        return {"image": rgb_img, "label": label, "size": size}


class ToTensor(object):
    def __call__(self, sample):
        # print("Tensoring...")
        image, label, size = sample["image"], sample["label"], sample["size"]
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        size = torch.from_numpy(size).float()
            
        return {"image": image, "label": label, "size": size}


class RemovePadding(object):
    """Removing the padding from images and labels"""    
    def __call__(self, sample):
        # print("Remove padding...")
        image, label, size = sample["image"], sample["label"], sample["size"]

        # size = torch.from_numpy(size)
        
        orig_img = torch.narrow(image, 1, 0, int(size[0]))     #Deleting the padding rows from image
        orig_img = torch.narrow(orig_img, 2, 0, int(size[1]))                    #Deleting the padding columns from image
        
        orig_lbl = torch.narrow(label, 0, 0, int(size[0]))                       #Deleting the padding rows from label
        orig_lbl = torch.narrow(orig_lbl, 1, 0, int(size[1]))                    #Deleting the padding columns from label

        return {"image": orig_img, "label": orig_lbl, "size": size}


class OneHotEncoder(object):
    def __call__(self, sample):
        # print("OneHotEncoder...")
        image, label, size = sample["image"], sample["label"], sample["size"]
        
        lbl = torch.from_numpy(label)
        onehot_label = nnF.one_hot(lbl.to(torch.int64), num_classes=4)
        
        return {"image": image, "label": onehot_label, "size": size}


class RandomZoom(object):
    def __init__(self, max_zoom):
        self.max_zoom = max_zoom
        
    
    def __call__(self, sample):
        # print("Random zoom...")
        image, label, size = sample["image"], sample["label"], sample["size"]
        # print(f"Zoom: initial size: {size}")
        zoom = 1-random.randint(0, self.max_zoom)/100
        
        new_h = int(zoom*size[0])
        new_w = int(zoom*size[1])
        
        h_del = size[0]-new_h
        w_del = size[1]-new_w
        
        new_image = image[int(h_del/2):int(new_h-h_del/2),int(w_del/2):int(new_w-w_del/2)]
        new_label = label[int(h_del/2):int(new_h-h_del/2),int(w_del/2):int(new_w-w_del/2)]

        # print(f"Zoom: final size: {[new_h, new_w]}")
        
        return {"image": new_image, "label": new_label, "size": [new_h, new_w]}

class Normalizer(object):
    def __call__(self, sample):
        # print("Normalizing...")
        image, label, size = sample["image"], sample["label"], sample["size"]
        
        # if type(image) == type(np.ndarray):
        #     image = image       
        # else:
        #     image = np.array(image)

        # norm = np.linalg.norm(image)
        # norm_img = image/norm

        norm_img = tF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return {"image": norm_img, "label": label, "size": size}

class Resize(object):
    def __init__(self, output_size: tuple):
        assert isinstance(output_size, tuple)
        self.output_size = output_size
        
    def __call__(self, sample):
        # print("Resizing...")
        image, label, size = sample["image"], sample["label"], sample["size"]

        new_img = np.array(Image.fromarray(image).resize(size=self.output_size))
        new_lbl = np.array(Image.fromarray(label).resize(size=self.output_size))
        return {"image": new_img, "label": new_lbl, "size": np.array(self.output_size)}

class Dataloading:
    """Creates two dataloaders. A train and test dataloader.
    Args:
        test_size: fraction of data used for testing.
        array_path: path to the folder containing the arrayfiles per slice.
        max_zoom: the maximum amount of zoom
        padding: the size of the largest image
        batch_size: size of the batches
        shuffle: "True" to enable shuffle, "False" to disable shuffle
    """

    def __init__(self, test_size, array_path, max_zoom=10, padding=(264, 288), min_size=(170, 200), batch_size=4, shuffle=False) -> None:
        self.test_size = test_size
        self.array_path = array_path
        self.max_zoom = max_zoom
        self.padding = padding
        self.min_size = min_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.create_dicts()
        self.create_transforms()
        self.create_dataloaders()

    def create_dicts(self):
        self.data_dict = create_indexed_file_dict(self.array_path)
        self.train_data_dict = {key: self.data_dict[key] for i, key in enumerate(self.data_dict.keys()) if i < (1-self.test_size)*len(self.data_dict)}
        self.test_data_dict = {key: self.data_dict[key] for i, key in enumerate(self.data_dict.keys()) if i >= (1-self.test_size)*len(self.data_dict)}

    def create_transforms(self):
        randomzoom = RandomZoom(self.max_zoom)
        padder = PadImage(self.padding)
        removepadder = RemovePadding()
        resizer = Resize(output_size=self.min_size)
        sudorgb_converter = SudoRGB()
        to_tensor = ToTensor()
        normalizer = Normalizer()
        encoder = OneHotEncoder()

        self.train_composed_transform = transforms.Compose([randomzoom, padder, sudorgb_converter, to_tensor, normalizer])
        self.test_composed_transform = transforms.Compose([padder, sudorgb_converter, to_tensor, normalizer])
    def create_dataloaders(self):
        self.train_slicedata = SliceDataset(self.array_path, self.train_data_dict, transform=self.train_composed_transform)
        self.test_slicedata = SliceDataset(self.array_path, self.test_data_dict, transform=self.test_composed_transform)

        self.train_dataloader = data.DataLoader(self.train_slicedata, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4)
        self.test_dataloader = data.DataLoader(self.test_slicedata, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4)
        self.dataloaders_combined = ConcatDataset([self.train_dataloader, self.test_dataloader])
    
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
    


def main():
    array_path = os.path.join(config.data_dir, "slice_arrays")
    
    data_dict = create_indexed_file_dict(array_path)

    dataloading = Dataloading(0.1, array_path)

    onefile = ""
    
    for i_batch, sample_batched in enumerate(dataloading.test_dataloader):
        print(i_batch, sample_batched['image'].size(),
           sample_batched['label'].size(),
           sample_batched["size"])
        onefile = sample_batched
        break
    print(onefile)
    img = tF.to_pil_image(onefile["image"][0,0,:,:])
    plt.imshow(img, cmap="gray")
    plt.show()
    # test_size = 0.2

    # train_data_dict = {key: data_dict[key] for i, key in enumerate(data_dict.keys()) if i < (1-test_size)*len(data_dict)}
    # test_data_dict = {key: data_dict[key] for i, key in enumerate(data_dict.keys()) if i >= (1-test_size)*len(data_dict)}
    
    
    # Use to calculate h and w for the image padding. Only run again if data changes.
    # heights, widths = get_all_shapes_hw(array_path, data_dict)
    # print(max(heights), max(widths))
    
    # padding = 428, 512
    
    # randomzoom = RandomZoom(20)
    # padder = PadImage(padding)
    # sudorgb_converter = SudoRGB()
    # remove_padding = RemovePadding()
    # to_tensor = ToTensor()
    # encoder = OneHotEncoder()
    # composed_transform = transforms.Compose([sudorgb_converter, to_tensor])
    # composed_zoomtransform = transforms.Compose([randomzoom, sudorgb_converter, to_tensor])


    # train_slicedata = SliceDataset(array_path, train_data_dict, transform=composed_transform)
    # test_slicedata = SliceDataset(array_path, test_data_dict, transform=composed_transform)
    
    # slicedata = SliceDataset(array_path, data_dict, transform=composed_transform)
    # zoomslicedata = SliceDataset(array_path, data_dict, transform=composed_zoomtransform)
    # slice = slicedata[1]
    # zoom_slice = zoomslicedata[600]
    # print(slice["image"].shape, zoom_slice["image"].shape)

    # plot_slice_with_lbl(zoom_slice["image"][1,:,:], zoom_slice["label"])
    
    # dataloader = data.DataLoader(slicedata, batch_size=1, shuffle=False, num_workers=8)
    
    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch, sample_batched['image'].size(),
    #       sample_batched['label'].size(),
    #       sample_batched["size"])
    

if __name__ == '__main__':
    main()