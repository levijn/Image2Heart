from os import walk, path, getcwd
import matplotlib.pyplot as plt
import nibabel as nib
import json
import numpy as np


def get_filenames(root_dir):
    for (_, _, filenames) in walk(root_dir):
        return filenames

def get_slice_count(root, filename):
    nifti_img = nib.load(path.join(root, filename)).get_fdata()
    return nifti_img.shape[2]

def create_slice_histogram(file_dict, show=False, save=False):
    slice_count_list = []
    for f in file_dict:
        if f.find("gt") > -1:
            continue
        slice_count_list.append(file_dict[f]["slice_count"])
    mean = np.mean(slice_count_list)
        
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    ax.hist(slice_count_list, bins=[x for x in range(5, 21)])
    ax.axvline(mean, color="r", linestyle="--")
    _, max_ylim = plt.ylim()
    ax.text(mean-3, max_ylim*0.85, 'Mean: {:.2f}'.format(mean))
    ax.set_xlabel("Slice count")
    ax.set_ylabel("Frames")
    ax.grid(alpha=0.4, axis="y", linestyle="--")
    
    if save:
        fig.savefig("slice_count_plot.png")
    if show:
        plt.show()

def main():
    root_file_dir = "C:\\Users\\levij\\TU Delft\\Engineering with AI\\Capstone\\DataACDC\\simpledata"
    filenames = get_filenames(root_file_dir)
    file_dict = {}
    
    i = 2
    for file in filenames:
        slice_count = get_slice_count(root_file_dir, file)
        file_dict[file] = {"slice_count": slice_count,
                           "patient_num": int(i/2)
                           }
        if file.find("gt")>-1:
            i += 1
    
    create_slice_histogram(file_dict, show=True, save=True)
    
    with open('data.json', 'w') as fp:
        json.dump(file_dict, fp, indent=4, sort_keys=True)
    

if __name__ == '__main__':
    main()