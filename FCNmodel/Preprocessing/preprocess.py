"""
Preprocesses the data and save the individual slices in folders. It saves them as a png file and as csv file
"""

import os
import sys
import inspect
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#add the parent folder to the path so modules can be imported
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.dirname(current_dir))

import config

data_dir = config.data_dir
sdata_dir = os.path.join(data_dir, "simpledata")

def get_filenames(directory):
    for (_, _, filenames) in os.walk(directory):
        return filenames


def plot_slice(slice_array):
    """Plots a slice of a frame"""
    fig = plt.figure(1)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(slice_array, cmap="gray")
    plt.show()


def convert_nifti_to_slices(img, label) -> tuple:
    """Returns a tuple of dictionaries. One dictionary for the 2D images and one for the corresponding labels"""
    img_dict = {}
    lbl_dict = {}
    for i in range(img.shape[2]):
        slice_name = f"slice{i+1}"
        img_dict[slice_name] = img[:,:,i]
        lbl_dict[slice_name] = label[:,:,i]
    return (img_dict, lbl_dict)
        

def save_slice_img(slice, name, location, format):
    """Makes an image from the slice, then saves it as a file at the given location"""
    path = os.path.join(location, name)
    max_value = np.max(slice)
    normalized_slice = np.uint8(slice/max_value * 255)
    im = Image.fromarray(normalized_slice, mode="L")
    im.save(f"{path}.{format}")


def save_slice_array(slice, name, location):
    path = os.path.join(location, name)
    np.savetxt(path, slice, delimiter=", ")


def save_all_slices_array(array_location, img_location):
    nifti_files = sorted(get_filenames(sdata_dir))
    print("test")
    for i in range(0, len(nifti_files), 2):
        img_file = nifti_files[i]
        lbl_file = nifti_files[i+1]
        
        img_path = os.path.join(sdata_dir, img_file)
        lbl_path = os.path.join(sdata_dir, lbl_file)

        img = nib.load(img_path)
        lbl = nib.load(lbl_path)
        
        img_array = img.get_fdata()
        lbl_array = lbl.get_fdata()
        img_slices, lbl_slices = convert_nifti_to_slices(img_array, lbl_array)
        
        for slice_name in img_slices:
            name = f"patient{int(1+i/4)}_{slice_name}"
            save_slice_array(img_slices[slice_name], name, array_location)
            save_slice_img(img_slices[slice_name], name, img_location, "png")
            
        for slice_name in lbl_slices:
            name = f"patient{int(1+i/4)}_{slice_name}_label"
            save_slice_array(lbl_slices[slice_name], name, array_location)
            save_slice_img(lbl_slices[slice_name], name, img_location, "png")


def main():
    array_location = os.path.join(data_dir, "slice_arrays")
    img_location = os.path.join(data_dir, "slice_images")
    save_all_slices_array(array_location, img_location)


if __name__ == '__main__':
    main()   