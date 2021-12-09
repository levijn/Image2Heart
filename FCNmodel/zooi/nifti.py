import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage

from sklearn.cluster import KMeans

patientfile = "82"
test_filename = "training\patient0" + patientfile + "\patient0" + patientfile + "_frame01.nii.gz"
label_filename = "training\patient0" + patientfile + "\patient0" + patientfile + "_frame01_gt.nii.gz"

img = nib.load(test_filename)
img_label = nib.load(label_filename)
data = img.get_fdata()
data_label = img_label.get_fdata()

class MriFrame:
    def __init__(self, nifti_img, nifti_lbl_img) -> None:
        self.img_ni = nifti_img
        self.img_ni_lbl = nifti_lbl_img
        
        self.img_data = self.img_ni.get_fdata()
        self.img_data_lbl = self.img_ni_lbl.get_fdata()
    
    def __iter__(self):
        self.slice_index = 0
        return self

    def __next__(self):
        if self.slice_index > self.get_slice_count():
            raise StopIteration
        slice = self.get_slice(self.slice_index)
        self.slice_index += 1
        return slice
    
    def get_slice_count(self):
        return self.img_data.shape[2]
    
    def get_slice(self, index):
        return self.get_slices()[index]
    
    def get_slices(self):
        slices = []
        for i in range(self.get_slice_count()):
            slices.append(self.img_data[:,:,i])
        return slices
    
    def get_labels(self):
        labels = []
        for i in range(self.get_slice_count()):
            labels.append(self.img_data_lbl[:,:,i])
        return labels
    
    def plot(self, slice=1, show=True, save=False, filename=None):
        if slice > self.get_slice_count()-1:
            raise IndexError(f"No slice with index {slice} in the image")
        
        fig = plt.figure(1)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(self.img_data[:,:,slice], cmap="gray")
        
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(self.img_data_lbl[:,:,slice], cmap="gray")
        
        if save and filename:
            plt.img_save(filename)
        if show:
            plt.show()


def get_slice_and_label(data, data_label, slice=5):
    return data[:,:,slice], data_label[:,:,slice]

def k_means_segmenation(img, n_clusters=5, n_init=10):
    img_to_segment = img.copy()
    flat_img = img_to_segment.reshape((-1,1))
    
    kmeans = KMeans(n_clusters=6, n_init=10)
    kmeans.fit(flat_img)

    values = kmeans.cluster_centers_.squeeze()
    labels = kmeans.labels_

    segmented_img = np.choose(labels, values)
    segmented_img = segmented_img.reshape(img.shape)
    
    return segmented_img


def test_kmeans_segment():
    img, img_label = get_slice_and_label(data, data_label)

    segmented_img = k_means_segmenation(img)

    vmin = img.min()
    vmax = img.max()

    fig = plt.figure(1)
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    ax1.set_title("Original")

    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(segmented_img, cmap="gray", vmin=vmin, vmax=vmax)
    ax2.set_title("Segmented K-Means")

    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow(img_label, cmap="gray")
    ax3.set_title("Label")

    plt.show()


# def show_slices(slices):
#     """ Function to display row of image slices """
#     fig, axes = plt.subplots(1, len(slices))
#     for i, s in enumerate(slices):
#         axes[i].imshow(s.T, cmap="gray", origin="lower")
#     fig.suptitle("One slice of the MRI scan with corresponding label")
#     plt.show()
# i = 3
# img_slice1 = data[:,:,i]
# img_slice2 = data[:,:,i+1]
# label_slice1 = data_label[:,:,i]
# label_slice2 = data_label[:,:,i+1]

# #show_slices([img_slice1, label_slice1, img_slice2, label_slice2])

# # -------creates the edges------
# #mask =  ndimage.binary_erosion(label_slice1.tolist())
# #label_slice1[mask] = 0


# img_copy1 = img_slice1.copy()

# for i in range(len(label_slice1)):
#     for j in range(len(label_slice1[1,:])):
#         if label_slice1[i,j] != 0:
#             img_copy1[i,j] = label_slice1[i,j]*50

#show_slices([img_slice1, label_slice1])