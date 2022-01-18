import sys
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


#add the parent folder to the path so modules can be imported
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
sys.path.append(os.path.join(parent_dir, "Preprocessing"))


def create_voxelplot_from_results(images):
    """Takes segmented images and creates a 3D"""
    num_slices = len(images)
    size = images[0].shape
    num_classes = 4
    # prepare some coordinates
    voxelarrays = [np.zeros((size[0], size[1], num_slices), dtype=bool) for x in range(num_classes)]
    for k in range(num_slices):
        for i in range(size[0]):
            for j in range(size[1]):
                a_val = images[k][i,j]
                voxelarrays[a_val][i,j,k] = True
    
    voxelarray = voxelarrays[1] | voxelarrays[2] | voxelarrays[3]
            
    colors = np.empty(voxelarray.shape, dtype=object)
    colors[voxelarrays[1]] = "red"
    colors[voxelarrays[2]] = "blue"
    colors[voxelarrays[3]] = "green"
    
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxelarray, alpha=0.6, facecolors=colors)
    plt.show()