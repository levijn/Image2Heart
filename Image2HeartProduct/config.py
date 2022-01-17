"""Configuration file for the project.

Data directory:
- To make sure the datafolder can be called you have to name the folder "Data"
- Place this folder as the same directory as you Image2Heart folder

"""
from pathlib import Path

# Saves the location of the root and data directory
root_dir = Path(__file__).resolve().parent
preprocessing_dir = root_dir / "Preprocessing"
data_dir = root_dir.parent.parent / "Data"
array_dir = data_dir / "slice_arrays"
image_dir = data_dir / "slice_images"
simpledata_dir = data_dir / "simpledata"

for p in [array_dir, image_dir, simpledata_dir]:
    p.mkdir(parents=True, exist_ok=True)
    