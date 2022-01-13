import os
import sys
import inspect
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from numpy.lib.npyio import save
from torchvision import transforms
from torch.utils import data
import torch

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.dirname(current_dir))

import config
from slicedataset import (SliceDataset,
                          PadImage,
                          SudoRGB,
                          ToTensor,
                          RemovePadding)
from preprocess import create_indexed_file_dict


class BatchProcessor:
    def __init__(self) -> None:
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
    
    @staticmethod
    def plot_batch(batch, show=True, save=False, save_loc="") -> Figure:
        """Plots a batch in a grid format. Any batch size should work
        Args:
            batch: a single batch loaded from a dataloader using the SliceDataset.
            show: if you want to show the images
            save: if you want the image
        """
        batch_list = BatchProcessor.remove_padding(batch)
        batch_size = len(batch_list)
        
        fig = plt.figure()
        
        gs = fig.add_gridspec(batch_size, 2, hspace=0, wspace=0)
        axis = gs.subplots()
        for i in range(batch_size):
            axis[i,0].set_axis_off()
            axis[i,1].set_axis_off()
            axis[i,0].imshow(batch_list[i]["image"][0,:,:], cmap="gray", interpolation='nearest')
            axis[i,1].imshow(batch_list[i]["label"], cmap="gray", interpolation='nearest')
        
        if show:
            plt.subplots_adjust(wspace=0, hspace=0, left=0, right=0.1, bottom=0, top=.7)
            plt.show()
        
        if save:
            fig.savefig("testsave")
        
        return fig
            
    

def main():
    array_path = os.path.join(config.data_dir, "slice_arrays")
    data_dict = create_indexed_file_dict(array_path)
    
    padding = 428, 512
    padder = PadImage(padding)
    sudorgb_converter = SudoRGB()
    remove_padding = RemovePadding()
    to_tensor = ToTensor()
    composed_transform = transforms.Compose([padder, sudorgb_converter, to_tensor])
    
    slicedata = SliceDataset(array_path, data_dict, transform=composed_transform)
    
    dataloader = data.DataLoader(slicedata, batch_size=8, shuffle=True, num_workers=8)
    
    
    for i_batch, sample_batched in enumerate(dataloader):
        print(BatchProcessor.remove_padding(sample_batched))
        BatchProcessor.plot_batch(sample_batched, save=True)
        break


if __name__ == '__main__':
    main()