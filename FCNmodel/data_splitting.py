import os, os.path, shutil
from pathlib import Path
import config
import sys
import inspect

parent_path = config.data_dir

def train_test_split(current_location="simpledata", train_folder="training_data", test_folder="testing_data", train_size=0.8, destination=parent_path):

    parent_path = config.data_dir                                   #path to the main "Data" folder
    folder_path = os.path.join(parent_path, current_location)       #path to current location of the files

    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for i in range(len(files)):
        if i < train_size*len(files):
            folder = train_folder
        else:
            folder = test_folder

        new_path = os.path.join(destination, folder)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        
        old_file_path = os.path.join(folder_path, files[i])
        new_file_path = os.path.join(new_path, files[i])
        shutil.copy(old_file_path, new_file_path)

def label_data_split(current_location, destination, data_folder="data", label_folder="labels"):

    parent_path = config.data_dir                               #path to the main "Data" folder
    folder_path = os.path.join(parent_path, current_location)   #path to current location of the files
    destination_path = os.path.join(parent_path, destination)   #path to destination folder

    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for file in files:
        if "gt" in file:
            folder_name = label_folder
        else:
            folder_name = data_folder

        new_path = os.path.join(destination_path, folder_name)
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        old_file_path = os.path.join(folder_path, file)
        new_file_path = os.path.join(new_path, file)
        shutil.move(old_file_path, new_file_path)

train_test_split()
label_data_split("training_data", "training_data")
label_data_split("testing_data", "testing_data")