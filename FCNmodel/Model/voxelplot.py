import sys
import os
import inspect
import matplotlib.pyplot as plt
import numpy as np


#add the parent folder to the path so modules can be imported
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
preprocess_dir = os.path.join(parent_dir, "Preprocessing")
sys.path.append(parent_dir)
sys.path.append(preprocess_dir)
import config
data_dir = config.data_dir
sdata_dir = os.path.join(data_dir, "simpledata")

from preprocess import load_slice_array

def example():
    # prepare some coordinates
    x, y, z = np.indices((8, 8, 8))

    # draw cuboids in the top left and bottom right corners, and a link between
    # them
    cube1 = (x < 3) & (y < 3) & (z < 3)
    cube2 = (x >= 5) & (y >= 5) & (z >= 5)
    link = abs(x - y) + abs(y - z) + abs(z - x) <= 2
    
    print(cube1.shape, cube2.shape, link.shape)

    # combine the objects into a single boolean array
    voxelarray = cube1 | cube2 | link

    # set the colors of each object
    colors = np.empty(voxelarray.shape, dtype=object)
    colors[link] = 'red'
    colors[cube1] = 'blue'
    colors[cube2] = 'green'

    # and plot everything
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxelarray, facecolors=colors, edgecolor='k')

    plt.show()

def create_voxelplot_from_results(images):
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


def main():
    patient = "0097"
    array = load_slice_array(os.path.join(data_dir, "slice_arrays", f"patient{patient}_slice0001_label"))
    array2 = load_slice_array(os.path.join(data_dir, "slice_arrays", f"patient{patient}_slice0002_label"))
    array3 = load_slice_array(os.path.join(data_dir, "slice_arrays", f"patient{patient}_slice0003_label"))
    array4 = load_slice_array(os.path.join(data_dir, "slice_arrays", f"patient{patient}_slice0004_label"))
    array5 = load_slice_array(os.path.join(data_dir, "slice_arrays", f"patient{patient}_slice0005_label"))
    array6 = load_slice_array(os.path.join(data_dir, "slice_arrays", f"patient{patient}_slice0006_label"))
    array7 = load_slice_array(os.path.join(data_dir, "slice_arrays", f"patient{patient}_slice0007_label"))
    array8 = load_slice_array(os.path.join(data_dir, "slice_arrays", f"patient{patient}_slice0008_label"))
   
    
    arrays = [array, array2, array3, array4, array5, array6, array7, array8]
    
    num_slices = len(arrays)
    num_classes = 4
    # prepare some coordinates
    voxelarrays = [np.zeros((array.shape[0], array.shape[1], num_slices), dtype=bool) for x in range(num_classes)]
    for k in range(num_slices):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                a_val = arrays[k][i,j]
                voxelarrays[a_val][i,j,k] = True
    
    voxelarray = voxelarrays[1] | voxelarrays[2] | voxelarrays[3]
            
    colors = np.empty(voxelarray.shape, dtype=object)
    colors[voxelarrays[1]] = "red"
    colors[voxelarrays[2]] = "blue"
    colors[voxelarrays[3]] = "green"
    # for i in range(colors.shape[0]):
    #     for j in range(colors.shape[1]):
    #         print(colors[i,j,0], end=" ")
    
    voxelarray = voxelarrays[1] | voxelarrays[2] | voxelarrays[3]
    # set the colors of each object
    # and plot everything
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxelarray, alpha=0.6, facecolors=colors)

    plt.show()

if __name__ == '__main__':
    main()
    #example()