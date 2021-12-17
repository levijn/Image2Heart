import torch
import os
import inspect
import numpy as np
import PIL
import matplotlib.pyplot as plt
import sys

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


def run_model_rtrn_results(pretrained=True, num_classes=4):
    """Running the model and testing it on 1 sample
    Args:
        pretrained: True: use the pretrained model, False: use model without pretraining.
        num_classes: number of classes the model has to look for.
    """
    #creating fcn model
    fcn = fcn_resnet50(pretrained=pretrained)
    #loading the weights from "weights.h5"
    device = "cuda"
    fcn = change_headsize(fcn, 4)
    fcn.load_state_dict(torch.load(os.path.join(currentdir, "weights_lr1.h5")))

    #retrieving 1 batch
    one_batch = None
    dataloading = Dataloading(test_size=0.2, array_path=config.array_dir, batch_size=4, shuffle=True)
    for i_batch, batch in enumerate(dataloading.train_dataloader):
        #remove the padding
        one_batch = batch
        break

    batch_img = one_batch["image"]
    batch_img = F.convert_image_dtype(batch_img, dtype=torch.float)
    fcn.eval()
    output = fcn(batch_img)["out"]
    normalized_masks = torch.nn.functional.softmax(output, dim=1)

    input_images = one_batch["image"]
    
    # image = one_batch["image"][0,0,:,:]
    # img = F.to_pil_image(image)
    # plt.imshow(img)
    # plt.show()
    
    # #Displaying probabilities of the num_classes
    # for i in range(normalized_masks.shape[1]):
    #     img = F.to_pil_image(normalized_masks[0,i,:,:])
    #     plt.imshow(img)
    #     plt.show()
    
    return normalized_masks, input_images

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
    
    fig = plt.figure(1)
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(F.to_pil_image(input_img), cmap="gray")
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(new_img, cmap="gray")
    plt.show()

def main():
    results, input_images = run_model_rtrn_results()
    create_segmentated_img(results[0,:,:,:], input_images[0,0,:,:])
    
    
if __name__ == '__main__':
    main()
    