import torch
import os
import inspect
import numpy as np
import PIL
import matplotlib.pyplot as plt
import sys
from torch.utils.data import dataloader

from torchvision.models.segmentation import fcn_resnet50
import torchvision.transforms.functional as F

#adding needed folderpaths
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
prepdir = os.path.join(parentdir, "Preprocessing")
sys.path.append(prepdir)
sys.path.append(parentdir)

#importing needed files
import config
from slicedataset import Dataloading
from change_head import change_headsize


def run_model_rtrn_results(image_tensor):
    """Running the model and testing it on 1 sample
    Args:
        pretrained: True: use the pretrained model, False: use model without pretraining.
        num_classes: number of classes the model has to look for.
    """
    #creating fcn model
    fcn = fcn_resnet50(pretrained=True)
    #loading the weights from "weights.h5"
    device = "cuda"
    fcn = change_headsize(fcn, 4)
    fcn.load_state_dict(torch.load(os.path.join(currentdir, "weights_lr1.h5")))

    image_float = F.convert_image_dtype(image_tensor, dtype=torch.float)
    fcn.eval()
    output = fcn(image_float)["out"]
    normalized_masks = torch.nn.functional.softmax(output, dim=1)
    
    return normalized_masks

def create_segmentated_img(result, input_img):
    new_img = np.zeros((result.size(dim=1), result.size(dim=2)))
    p0_probabilities = []
    p1_probabilities = []
    p2_probabilities = []
    p3_probabilities = []
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

def main():
    one_batch = None
    dataloading = Dataloading(test_size=0.2, array_path=config.array_dir, batch_size=4, shuffle=True)
    for i_batch, batch in enumerate(dataloading.test_dataloader):
        #remove the padding
        one_batch = batch
        break
    img = one_batch["image"][0,0,:,:]
    for i in range(img.size(dim=0)):
        for j in range(img.size(dim=1)):
            print(img[i,j].item(), end=" ")

    results = run_model_rtrn_results(one_batch["image"])
    create_segmentated_img(results[0,:,:,:], one_batch["image"][0,0,:,:])
    
    
if __name__ == '__main__':
    main()
    