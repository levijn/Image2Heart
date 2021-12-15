from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50
import torch
import os
import inspect
import numpy as np
import PIL
import matplotlib.pyplot as plt
import sys

import torchvision.transforms.functional as F
from torchvision.transforms.functional import convert_image_dtype, resize
from torchvision.transforms import Resize
        
from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
prepdir = os.path.join(parentdir, "Preprocessing")
sys.path.append(prepdir)
sys.path.append(parentdir)

import config
from slicedataset import Dataloading

def main():
    array_path = os.path.join(config.data_dir, "slice_arrays")
    dataloading = Dataloading(0.2, array_path)

    onebatch = ""
    for i_batch, sample_batched in enumerate(dataloading.test_dataloader):
        onefile = sample_batched
        break

    samples = Dataloading.remove_padding(onefile)
    image = samples[0]["image"]

    img = F.to_pil_image(image[1,:,:])
    plt.imshow(img)
    plt.show()


    plt.rcParams["savefig.bbox"] = 'tight'

    # dog_int = read_image(os.path.join(currentdir, 'dog.jpg'))
    # dog2_int = read_image(os.path.join(currentdir, 'dog2.png'))
    # boat_int = read_image(os.path.join(currentdir, 'boat.jpg'))
    # vinc_int = read_image(os.path.join(currentdir, 'Vincent.png'))

    # vinc_rgba = PIL.Image.open(os.path.join(currentdir, 'Vincent.png'))
    # vinc_rgb = vinc_rgba.convert('RGB')
    # vinc_new = convert_tensor(vinc_rgb)

    # print(dog2_int.shape)


    #Used to plot the images:
    # dog = dog_int.detach()
    # dog2 = dog2_int.detach()
    # dog = F.to_pil_image(dog)
    # dog_array = np.asarray(dog)
    # dog2 = F.to_pil_image(dog2)
    # dog2_array = np.asarray(dog2)

    batch_int = torch.stack([image])
    batch = convert_image_dtype(batch_int, dtype=torch.float)

    fcn = fcn_resnet50(pretrained=True)
    # print(fcn)
    fcn = fcn.eval()

    print(batch)
    normalized_batch = F.normalize(batch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    print(normalized_batch.shape)

    output = fcn(normalized_batch)["out"]

    normalized_masks = torch.nn.functional.softmax(output, dim=1)

    for i in range(normalized_masks.shape[1]):
        img = F.to_pil_image(normalized_masks[0,i,:,:])
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    main()