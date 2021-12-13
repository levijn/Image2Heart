"""Configuration file for the project.

Data directory:
- To make sure the datafolder can be called you have to name the folder "Data"
- Place this folder as the same directory as you Image2Heart folder

"""

import os
import sys
import inspect


# Saves the location of the root and data directory
root_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
data_dir = os.path.join(os.path.dirname(os.path.dirname(root_dir)), "Data")
preprocessing_dir = os.path.join(root_dir, "Preprocessing")
