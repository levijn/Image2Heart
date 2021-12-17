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


def training_model(test_size=0.2, num_epochs=10, batch_size=16, learning_rate=0.0001, pretrained=True, shuffle=True, array_path=config.array_dir, num_classes=4):
    """Trains the model using the dataloader
    Args:
        test_size: fraction of data used for testing.
        num_epochs: number of epochs used for training.
        batch_size: size of the batches.
        learning_rate: value of the learning rate.
        pretrained: True: use the pretrained model, False: use model without pretraining.
        shuffle: "True" to enable shuffle, "False" to disable shuffle
        array_path: path to the folder containing the arrayfiles per slice.
        num_classes: number of classes the model has to look for.
    """
    #loading datafiles
    dataloading = Dataloading(test_size=test_size, array_path=array_path, batch_size=batch_size, shuffle=shuffle)
    #creating fcn model
    fcn = fcn_resnet50(pretrained=pretrained)

    fcn.train()
    device = "cuda"
    
    # feezing its parameters
    for param in fcn.parameters():
        param.requires_grad = False
    
    fcn = change_headsize(fcn, 4).to(device)

    
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in fcn.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in fcn.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    #looping through epochs
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        running_loss = 0.0
        
        #looping through batches in each epoch
        for i_batch, batch in enumerate(dataloading.train_dataloader):
            print(f"Batch: {i_batch}")

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(fcn.parameters(), lr=learning_rate)

            data = convert_image_dtype(batch["image"], dtype=torch.float).to(device)
            target = batch["label"].to(device)
            optimizer.zero_grad()
            output = fcn(data)
            loss = criterion(output["out"], target.long())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i_batch % 50 == 49:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i_batch + 1, running_loss / 50))
                running_loss = 0.0
    
    #saving calculated weights to "weights.h5"
    torch.save(fcn.state_dict(), os.path.join(currentdir, "weights.h5"))

def running_model(pretrained=False, num_classes=4):
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
    fcn.load_state_dict(torch.load(os.path.join(currentdir, "weights.h5")))


    plt.rcParams["savefig.bbox"] = 'tight'

    #retrieving 1 image for training
    one_batch = None
    dataloading = Dataloading(test_size=0.2, array_path=config.array_dir, batch_size=4, shuffle=True)
    for i_batch, batch in enumerate(dataloading.train_dataloader):
        #remove the padding
        one_batch = batch
        break

    sample = one_batch["image"]
    sample = convert_image_dtype(sample, dtype=torch.float)

    fcn.eval()

    # normalized_sample = F.normalize(sample, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    output = fcn(sample)["out"]
    normalized_masks = torch.nn.functional.softmax(output, dim=1)

    # Displaying input image
    image = one_batch["image"][0,0,:,:]
    img = F.to_pil_image(image)
    plt.imshow(img)
    plt.show()
    
    #Displaying probabilities of the num_classes
    for i in range(normalized_masks.shape[1]):
        img = F.to_pil_image(normalized_masks[0,i,:,:])
        plt.imshow(img)
        plt.show()

def main():
    #set to True if the model has been trained with the weights stored at "weights.h5", False otherwise
    trained = False

    if trained is False:
        training_model(num_epochs=10, pretrained=True)
        running_model(pretrained=True)
    elif trained is True:
        running_model(pretrained=True)

if __name__ == '__main__':
    main()