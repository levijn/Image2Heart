# Image to Heart 


## Introduction
The goal of this project is to automatically segment the heart of a patient.
This is done by training a fully convolutional network on labeled MRI scans of the heart.  


## Setting up project
Before any setup you have to download the data. This should be provided by one of the group members. To get the project working you have to then run preprocess.py. This will create the neccesary folders and files.


## Running the model
After setting up the project you can run nifti files through our model by using segment_patient.py