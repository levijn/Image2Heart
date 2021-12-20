"""Steps:
1. make k amount of dataloaders
2. loop over range(k) with:
    1 assign i'th dataloader as test datasets
    2 assign the others as training datasets
    3 train on train data
    4 test on test data
    5 calculate loss
3. average all the losses
"""
# import stuff
import torch
import os
import inspect
import numpy as np
import PIL
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from torch.utils.data import dataloader

from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50
import torchvision.transforms.functional as F
from torchvision.transforms.functional import convert_image_dtype, resize
from torchvision.transforms import Resize
from torchvision.utils import make_grid
from torchvision.io import read_image

from sklearn.model_selection import KFold


# directories set-up
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
prepdir = os.path.join(parentdir, "Preprocessing")
modeldir = os.path.join(parentdir, "Model")

sys.path.append(prepdir)
sys.path.append(parentdir)
sys.path.append(modeldir)

# make dataloaders
from slicedataset import Dataloading
from change_head import change_headsize
import config


def kfold_training(number_of_folds = 3,test_size=0.2, num_epochs=2, batch_size=16, learning_rate=0.001, pretrained=True, shuffle=True, array_path=config.array_dir, num_classes=4):
    kfold = KFold(n_splits = number_of_folds)


    # prepare dataset bij concatenating Train/Test part; we split later.
    dataloading = Dataloading(test_size=0.2, array_path=config.array_dir, batch_size=4, shuffle=True)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataloading.dataloaders_combined)):
        print(f"FOLD {fold}")

        # Sample elements randomly from a givnen list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                        dataloading, 
                        batch_size=10, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                        dataloading,
                        batch_size=10, sampler=test_subsampler)

        
        fcn = fcn_resnet50(pretrained=pretrained)

        fcn.train()
        device = "cuda"

        #creating an empty list for the losses
        train_loss_per_epoch = []
        eval_loss_per_epoch = []
        
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
            epoch_train_loss = 0.0
            epoch_eval_loss = 0.0
            
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
                
                epoch_train_loss += loss.item()
                if i_batch % 50 == 49:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i_batch + 1, epoch_train_loss / 50))
                    epoch_train_loss = 0.0

            # calculate validation loss after training
            fcn.eval()
            for i_batch, batch in enumerate(dataloading.test_dataloader):
                criterion = torch.nn.CrossEntropyLoss()
                data = F.convert_image_dtype(batch["image"], dtype=torch.float).to(device)
                target = batch["label"].to(device)
                output = fcn(data)
                loss = criterion(output["out"], target.long())
                epoch_eval_loss += output["out"].shape[0]*loss.item()


            train_loss_per_epoch.append(epoch_train_loss/len(dataloading.train_slicedata))
            eval_loss_per_epoch.append(epoch_eval_loss/len(dataloading.test_slicedata))
        
        #saving calculated weights to "weights.h5"
        torch.save(fcn.state_dict(), os.path.join(currentdir, "weights.h5"))

    return train_loss_per_epoch, eval_loss_per_epoch


if __name__ == "__main__":
    kfold_training()