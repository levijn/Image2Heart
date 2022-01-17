"""
Preprocesses the data and save the individual slices in folders. It saves them as a png file and as csv file.

- Contains a function to create the indexed dictionary.
"""

from array import array
import os
import sys
from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json

#add the parent folder to the path so modules can be imported
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

import config

data_dir = config.data_dir
sdata_dir = os.path.join(data_dir, "simpledata")


def save_dict(dict, path, filename):
    p = os.path.join(path, filename)
    with open(f'{p}.json', 'w') as fp:
        json.dump(dict, fp,  indent=4)


def load_dict(path):
    p = path
    if ".json" not in path:
        p += ".json"
    with open(p, 'r') as fp:
        data_dict = json.load(fp)
    return data_dict


def get_filenames(directory) -> list:
    """Returns a list of all the filenames in the directory"""
    for (_, _, filenames) in os.walk(directory):
        return filenames


def plot_slice_with_lbl(slice_array, lbl_array) -> None:
    """Plots a slice of a frame"""
    fig = plt.figure(1)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(slice_array, cmap="gray")
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(lbl_array, cmap="gray")
    plt.show()


def create_four_digit_num_str(number) -> str:
    """Adds zeros in front of the number and returns it as a string"""
    num_str = str(number)
    while len(num_str) < 4:
        num_str = "0" + num_str
    return num_str


def convert_nifti_to_slices(img, label) -> tuple:
    """Returns a tuple of dictionaries. One dictionary for the 2D images and one for the corresponding labels"""
    img_dict = {}
    lbl_dict = {}
    for i in range(img.shape[2]):
        num = create_four_digit_num_str(i+1)
        slice_name = "slice" + num
        img_dict[slice_name] = img[:,:,i]
        lbl_dict[slice_name] = label[:,:,i]
    return (img_dict, lbl_dict)
        

def save_slice_img(slice, name, location, format) -> None:
    """Makes an image from the slice, then saves it as a file at the given location"""
    path = os.path.join(location, name)
    max_value = np.max(slice)
    normalized_slice = np.uint8(slice/max_value * 255)
    im = Image.fromarray(normalized_slice, mode="L")
    im.save(f"{path}.{format}")


def save_slice_array(slice, name, location) -> None:
    """Saves a slice array as a csv file"""
    path = os.path.join(location, name)
    np.savetxt(path, slice, delimiter=",")


def load_slice_array(filename):
    """Loads a slice array csv file and returns it as an array"""
    array = np.loadtxt(filename, dtype=int, delimiter=",")
    return array


def save_all_slices_array(array_location):
    """Saves all the slices as arrays in the slice_arrays folder"""
    print("Creating arrays...")
    nifti_files = sorted(get_filenames(sdata_dir))
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
        
        patient_str = create_four_digit_num_str(int(1+i/4))
        
        for slice_name in img_slices:
            name = f"patient{patient_str}_{slice_name}"
            save_slice_array(img_slices[slice_name], name, array_location)
            
        for slice_name in lbl_slices:
            name = f"patient{patient_str}_{slice_name}_label"
            save_slice_array(lbl_slices[slice_name], name, array_location)
        print(f"Saving arrays patient {patient_str}")
    print("Done")


def save_all_slice_image(img_location):
    """Saves all the slices as images in the slice_images folder"""
    print("Creating images...")
    nifti_files = sorted(get_filenames(sdata_dir))
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
        
        patient_str = create_four_digit_num_str(int(1+i/4))
        
        for slice_name in img_slices:
            name = f"patient{patient_str}_{slice_name}"
            save_slice_img(img_slices[slice_name], name, img_location, "png")
            
        for slice_name in lbl_slices:
            name = f"patient{patient_str}_{slice_name}_label"
            save_slice_img(lbl_slices[slice_name], name, img_location, "png")
        
        print(f"Saving images patient {patient_str}")
    print("Done")


def create_indexed_file_dict(array_dir, max_size=300):
    data_dict = {}
    filenames = sorted(get_filenames(array_dir))
    skippedfiles = 0
    # img_sizesh = []
    # img_sizesw = []
    for i in range(0, len(filenames), 2):
        img_file = filenames[i]
        lbl_file = filenames[i+1]
        
        img = load_slice_array(os.path.join(array_dir, filenames[i]))
        if img.shape[0] > max_size or img.shape[1] > max_size:
            skippedfiles += 2
            continue
        # img_sizesh.append(img.shape[0])
        # img_sizesw.append(img.shape[1])
        slice_dict = {
            "img_data_file": img_file,
            "lbl_data_file": lbl_file
        }
        data_dict[int(i/2)-skippedfiles] = slice_dict
    return data_dict


def get_all_shapes_hw(data_dir, idx_dict):
    """Returns the height and width of all images"""
    widths = []
    heights = []
    for i in range(len(idx_dict)):
        slice = idx_dict[i]
        img_data_file = slice["img_data_file"]
        img = load_slice_array(os.path.join(data_dir, img_data_file))
        img_h, img_w = img.shape[:2]
        heights.append(img_h)
        widths.append(img_w)
    return (heights, widths)


def create_hist_imgsize(heights, widths, plot=False, save=False):
    """Creates a histogram of all the widths and heights of the images.\n
    Args:
        heights: list of all the heights
        widths: list of all the widths
        plot: True if you want to show the plots
        save: True if you want to save the plots in current folder
    """
    fig = plt.figure(1)
    ax1 = fig.add_subplot(1,2,1)
    ax1.hist(heights, bins=[x for x in range(100, 600, 50)])
    ax1.set_xlabel("Height")
    ax1.set_title("Image Height")
    ax1.grid(alpha=0.4, axis="y", linestyle="--")
    
    ax2 = fig.add_subplot(1,2,2)
    ax2.hist(widths, bins=[x for x in range(100, 600, 50)])
    ax2.set_xlabel("Width")
    ax1.set_title("Image Width")
    ax2.grid(alpha=0.4, axis="y", linestyle="--")
    
    if save:
        plt.savefig("Img_size_hist.png")
    if plot:
        plt.show()


def main():
    # Save all slices as arrays in the array_dir
    if len([x for x in config.array_dir.iterdir()]) == 0:
        save_all_slices_array(config.array_dir)
    
    # Optional to save all slices as png-images. Set save_images = True to save.
    save_images = False
    if save_images == True and len([x for x in config.image_dir.iterdir()]) == 0:
        save_all_slice_image(config.image_dir)
    
    # Filter the slices to size. A couple of large images are will not be used for training.
    # The filtered data is saved in a JSON file
    print("Creating filtered dictionary")
    filtered_dict = create_indexed_file_dict(config.array_dir)
    print("Saving dictionary")
    save_dict(filtered_dict, current_dir, "filtered_data")
    print("Saved dictionary successfully.")
    

if __name__ == '__main__':
    main()