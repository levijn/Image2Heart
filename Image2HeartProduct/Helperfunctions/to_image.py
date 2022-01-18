import os
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.models.segmentation import fcn_resnet50

#adding needed folderpaths
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
sys.path.append(os.path.join(parent_dir, "Preprocessing"))

#importing needed files
import config
from slicedataset import Dataloading
from change_head import change_headsize


def run_model_rtrn_results(image_tensor, model_path):
    """Running the model and testing it on 1 sample
    Args:
        pretrained: True: use the pretrained model, False: use model without pretraining.
        num_classes: number of classes the model has to look for.
    """
    # Creating FCN model
    fcn = fcn_resnet50(pretrained=True)
    fcn = change_headsize(fcn, 4)
    fcn.load_state_dict(torch.load(model_path))

    fcn.eval()
    output = fcn(image_tensor)["out"]
    normalized_masks = torch.nn.functional.softmax(output, dim=1)
    
    return normalized_masks


def create_segmentated_img(result):
    new_img = np.zeros((result.size(dim=1), result.size(dim=2)), dtype=int)
    for i in range(result.size(dim=1)):
        for j in range(result.size(dim=2)):
            p0 = result[0,i,j].item()
            p1 = result[1,i,j].item()
            p2 = result[2,i,j].item()
            p3 = result[3,i,j].item()
            probalities = [p0,p1,p2,p3]
            largest_prob = max(probalities)
            largest_prob_cls = probalities.index(largest_prob)
            new_img[i,j] = largest_prob_cls
    return new_img


def convert_to_segmented_imgs(results):
    images = []
    for i in range(results.size(dim=0)):
        images.append(create_segmentated_img(results[i,:,:,:]))
    return images
    