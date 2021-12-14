import os, os.path, shutil
from pathlib import Path
import config
import sys
import inspect

def label_data_split(location, data_destination, label_destination):

    parent_path = config.data_dir       #path to the main "Data" folder
    folder_path = os.path.join(parent_path, location)       #path to current location of the files

    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for file in files:
        if "gt" in file:
            folder_name = label_destination
        else:
            folder_name = data_destination

        new_path = os.path.join(parent_path, folder_name)
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        old_image_path = os.path.join(folder_path, file)
        new_image_path = os.path.join(new_path, file)
        shutil.copy(old_image_path, new_image_path)