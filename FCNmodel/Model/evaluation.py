"""uses model_training.py to train the model for different parameters
assumptions: 
pretrained = false
num_classes = 4
"""
import torch
import os
import inspect
import numpy as np
import PIL
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import math

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
from preprocess import save_slice_array

#import the training model
from model_training import training_model, running_model

#copied the training model from model_training.py, and modified it a bit:
#for each epoch, the loss is added to a list. This list is returned
def training_model(test_size=0.2, num_epochs=1, batch_size=4, learning_rate=0.001, pretrained=False, shuffle=True, array_path=config.array_dir, num_classes=4):
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
    fcn = fcn_resnet50(pretrained=pretrained, num_classes=num_classes)

    #creating an empty list for the losses
    loss_per_epoch = []

    #looping through epochs
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        running_loss = 0.0
        
        #looping through batches in each epoch
        for i_batch, sample_batched in enumerate(dataloading.train_dataloader):
            #remove the padding
            batch = Dataloading.remove_padding(sample_batched)

            #looping through samples in each batch
            for sample in batch:
                fcn.train()
                device = "cuda"
                fcn = fcn.to(device)
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(fcn.parameters(), lr=learning_rate)

                data = convert_image_dtype(torch.stack([sample["image"]]), dtype=torch.float).to(device)
                target = torch.stack([sample["label"]]).to(device)
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
            
            # add loss to loss list if batch is the last of epoch
            if i_batch == (math.ceil(len(dataloading.train_slicedata)/batch_size))-1:
                loss_per_epoch.append(loss)
    return loss_per_epoch
    





def evaluation():
    learning_rates = [0.001]
    num_epochs = 2
    batch_size = 4
    # calculate the loss for each learning rate
    for learning_rate in learning_rates:
        loss_per_epoch = training_model(learning_rate = learning_rate, num_epochs = num_epochs, batch_size = batch_size)

        # save the loss
        save_slice_array(training_model, f"loss per epoch with learning rate {learning_rate}", currentdir)

        #plot the loss
        plt.plot(loss_per_epoch, range(num_epochs))
    plt.xlabel("loss")
    plt.ylabel("epoch")
    plt.show()



# running the model
def main():
    evaluation()

if __name__ == '__main__':
    main()
