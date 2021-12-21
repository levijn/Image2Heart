import os
import sys
import inspect
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils import data
import torchvision.transforms.functional as F

#add the parent folder to the path so modules can be imported
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
preprocess_dir = os.path.join(parent_dir, "Preprocessing")
sys.path.append(parent_dir)
sys.path.append(preprocess_dir)
import config
data_dir = config.data_dir
sdata_dir = os.path.join(data_dir, "simpledata")

from to_image import run_model_rtrn_results, create_segmentated_img
from onepatient_dataset import OnePatientDataset
from slicedataset import (RandomZoom,
                          PadImage,
                          SudoRGB,
                          ToTensor,
                          Normalizer)

def load_patient_nif_to_tensor(file):
    pt_array = nib.load(file).get_fdata().astype("int")
    sudo_images = []
    new_h, new_w = 264, 288

    for i in range(pt_array.shape[2]):
        slice = pt_array[:,:,i]
        rgb_img = np.stack([slice]*3, axis=0)
        tensor_img = torch.from_numpy(rgb_img)
        tensor_img = F.convert_image_dtype(tensor_img, dtype=torch.float)
        norm_img = F.normalize(tensor_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        sudo_images.append(norm_img)

    stacked_tensor = torch.stack(sudo_images)
    return stacked_tensor
    

def create_3d_scatterplot(results):
    segmented_images = []
    for i in range(results.size(dim=0)):
        segmented_images.append(create_segmentated_img(results[i,:,:,:]))
    
    fig = plt.figure(1)
    ax = fig.add_subplot(projection="3d")
    colors = ["r", "b", "g"]
    for k in range(len(segmented_images)):
        img = segmented_images[k]
        for i in range(0, img.shape[0], 5):
            for j in range(0, img.shape[1], 5):
                if img[i,j] != 0:
                    ax.scatter(i, j, k*20, c=colors[int(img[i,j])-1], marker="o")
    plt.show()

def create_3d_scatterplot_labels(labels):
    list_labels = []
    for i in range(labels.size(dim=0)):
        list_labels.append(labels[i,:,:].numpy())
    
    fig = plt.figure(1)
    ax = fig.add_subplot(projection="3d")
    colors = ["r", "b", "g"]
    for k in range(len(list_labels)):
        img = list_labels[k]
        for i in range(0, img.shape[0], 5):
            for j in range(0, img.shape[1], 5):
                if img[i,j] != 0:
                    ax.scatter(i, j, k*20, c=colors[int(img[i,j])-1], marker="o")
    plt.show()


def main():
    patient = "088"
    img_path = os.path.join(config.data_dir, "simpledata", f"patient{patient}_frame01.nii.gz")
    lbl_path = os.path.join(config.data_dir, "simpledata", f"patient{patient}_frame01_gt.nii.gz")
    
    
    
    randomzoom = RandomZoom(10)
    padder = PadImage((264, 288))
    sudorgb_converter = SudoRGB()
    to_tensor = ToTensor()
    normalizer = Normalizer()
    composed_transform = transforms.Compose([randomzoom, padder, sudorgb_converter, to_tensor, normalizer])
    
    dataset = OnePatientDataset(img_path, lbl_path, transform=composed_transform)
    dataloader = data.DataLoader(dataset, batch_size=32)
    
    patient_batch = None
    for batch in dataloader:
        patient_batch = batch
    
    results = run_model_rtrn_results(patient_batch["image"])
    labels = patient_batch["label"]
    
    create_3d_scatterplot(results)
    create_3d_scatterplot_labels(labels)

    
    
    # fig = plt.figure(1)
    # ax1 = fig.add_subplot(1,5,1)
    # ax1.imshow(F.to_pil_image(slice_result[0,:,:]))
    # ax2 = fig.add_subplot(1,5,2)
    # ax2.imshow(F.to_pil_image(slice_result[1,:,:]))
    # ax3 = fig.add_subplot(1,5,3)
    # ax3.imshow(F.to_pil_image(slice_result[2,:,:]))
    # ax4 = fig.add_subplot(1,5,4)
    # ax4.imshow(F.to_pil_image(slice_result[3,:,:]))
    # ax5 = fig.add_subplot(1,5,5)
    # ax5.imshow(F.to_pil_image(slice_input), cmap="gray")
    # plt.show()


if __name__ == '__main__':
    main()