from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50
import torch
import os
import inspect
import numpy as np
import PIL
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
from torchvision.transforms.functional import convert_image_dtype, resize
from torchvision.transforms import Resize
        
from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

convert_tensor = transforms.ToTensor()

plt.rcParams["savefig.bbox"] = 'tight'

dog_int = read_image(os.path.join(currentdir, 'dog.jpg'))
dog2_int = read_image(os.path.join(currentdir, 'dog2.png'))
boat_int = read_image(os.path.join(currentdir, 'boat.jpg'))
vinc_int = read_image(os.path.join(currentdir, 'Vincent.png'))

vinc_rgba = PIL.Image.open(os.path.join(currentdir, 'Vincent.png'))
vinc_rgb = vinc_rgba.convert('RGB')
vinc_new = convert_tensor(vinc_rgb)

print(dog2_int.shape)


#Used to plot the images:
dog = dog_int.detach()
dog2 = dog2_int.detach()
dog = F.to_pil_image(dog)
dog_array = np.asarray(dog)
dog2 = F.to_pil_image(dog2)
dog2_array = np.asarray(dog2)

batch_int = torch.stack([vinc_new])
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

sem_classes = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
sem_class_to_idx = 'person'
#{cls: idx for (idx, cls) in enumerate(sem_classes)}


dog_and_boat_masks = [
    normalized_masks[img_idx, sem_class_to_idx[cls]]
    for img_idx in range(batch.shape[0])
    for cls in ('dog', 'boat')
]

dog_mask = dog_and_boat_masks[1]
dog_mask = dog_mask.detach()
dog_mask = F.to_pil_image(dog_mask)
dog_array = np.asarray(dog_mask)
plt.imshow(dog_array)
plt.show()