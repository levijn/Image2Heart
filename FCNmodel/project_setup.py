"""
Run this file to setup the right directories.

Creates folder structure as follows:
- Data
    - slice_arrays
    - slice_images

After running you have to place the data in the data folder with the name "simpledata"
"""
import os
import sys


def setup():
    root_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(os.path.dirname(root_dir))
    data_dir = os.path.join(parent_dir, "Datas")
    array_dir = os.path.join(data_dir, "slice_arrays")
    images_dir = os.path.join(data_dir, "slice_images")
    
    for directory in [data_dir, array_dir, images_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)


if __name__ == '__main__':
    setup()