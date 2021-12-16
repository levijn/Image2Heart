import torch
import os
import inspect
import numpy as np
import PIL
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50
import torchvision.transforms.functional as F
from torchvision.transforms.functional import convert_image_dtype, resize
from torchvision.transforms import Resize
from torchvision.utils import make_grid
from torchvision.io import read_image

#adding folderpaths
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
prepdir = os.path.join(parentdir, "Preprocessing")
sys.path.append(prepdir)
sys.path.append(parentdir)

import config
from slicedataset import Dataloading

def main():
    array_path = os.path.join(config.data_dir, "slice_arrays")
    dataloading = Dataloading(0.1, array_path, shuffle=True)
    num_epochs = 10
    fcn = fcn_resnet50(pretrained=False, num_classes=4)
    #Retrieving an image from dataloader
    # for epoch in range(num_epochs):
    #     print(f"Epoch: {epoch}")
    #     running_loss = 0.0
    #     for i_batch, sample_batched in enumerate(dataloading.train_dataloader):
    #         onefile = sample_batched
    #         samples = Dataloading.remove_padding(onefile)

    #         for sample in samples:
    #             fcn.train()

    #             device = "cuda"
    #             fcn = fcn.to(device)
    #             criterion = torch.nn.CrossEntropyLoss()
    #             LR = 0.001
    #             optimizer = torch.optim.Adam(fcn.parameters(), lr=LR)

    #             data = convert_image_dtype(torch.stack([sample["image"]]), dtype=torch.float).to(device)
    #             target = torch.stack([sample["label"]]).to(device)
    #             optimizer.zero_grad()
    #             output = fcn(data)
    #             loss = criterion(output["out"], target.long())
    #             loss.backward()
    #             optimizer.step()
                
    #             running_loss += loss.item()
    #             if i_batch % 50 == 49:    # print every 2000 mini-batches
    #                 print('[%d, %5d] loss: %.3f' %
    #                     (epoch + 1, i_batch + 1, running_loss / 50))
    #                 running_loss = 0.0
    
    # torch.save(fcn.state_dict(), os.path.join(currentdir, "weights.h5"))


    plt.rcParams["savefig.bbox"] = 'tight'

    fcn = fcn_resnet50(pretrained=False, num_classes=4)

    fcn.load_state_dict(torch.load(os.path.join(currentdir, "weights.h5")))

    one_sample = None
    for i_batch, sample_batched in enumerate(dataloading.train_dataloader):
        onefile = sample_batched
        samples = Dataloading.remove_padding(onefile)
        one_sample = samples[0]
        break
    
    batch_int = torch.stack([one_sample["image"]])
    batch = convert_image_dtype(batch_int, dtype=torch.float)

    fcn = fcn.eval()

    normalized_batch = F.normalize(batch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    output = fcn(normalized_batch)["out"]
    normalized_masks = torch.nn.functional.softmax(output, dim=1)

    image = one_sample["image"]
        
    # Displaying input image
    img = F.to_pil_image(image[1,:,:])
    plt.imshow(img)
    plt.show()
    
    for i in range(normalized_masks.shape[1]):
        img = F.to_pil_image(normalized_masks[0,i,:,:])
        plt.imshow(img)
        plt.show()
    
# def troep():
    # dog_int = read_image(os.path.join(currentdir, 'dog.jpg'))
    # dog2_int = read_image(os.path.join(currentdir, 'dog2.png'))
    # boat_int = read_image(os.path.join(currentdir, 'boat.jpg'))
    # vinc_int = read_image(os.path.join(currentdir, 'Vincent.png'))
    
    # vinc_rgba = PIL.Image.open(os.path.join(currentdir, 'Vincent.png'))
    # vinc_rgb = vinc_rgba.convert('RGB')
    # to_tensor = transforms.ToTensor()
    # vinc_new = to_tensor(vinc_rgb).to(device)

    # print(dog2_int.shape)

    #Used to plot the images:
    # dog = dog_int.detach()
    # dog2 = dog2_int.detach()
    # dog = F.to_pil_image(dog)
    # dog_array = np.asarray(dog)
    # dog2 = F.to_pil_image(dog2)
    # dog2_array = np.asarray(dog2)

    # batch_int = torch.stack([vinc_new])
    # batch = convert_image_dtype(batch_int, dtype=torch.float)

    # fcn = fcn.eval()

    # # print(batch)
    # normalized_batch = F.normalize(batch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    # # print(normalized_batch.shape)

    # out = fcn(normalized_batch)
    # output = out["out"]
    # print(output)


    # normalized_masks = torch.nn.functional.softmax(output, dim=1)

    # for i in range(normalized_masks.shape[1]):
    #     img = F.to_pil_image(normalized_masks[0,i,:,:])
    #     plt.imshow(img)
    #     plt.show()


if __name__ == '__main__':
    main()