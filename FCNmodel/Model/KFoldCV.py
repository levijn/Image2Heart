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
from preprocess import save_slice_array
from change_head import change_headsize
import config


def kfold_training(number_of_folds = 2, test_size=0.2, num_epochs=2, batch_size=16, learning_rate=0.001, pretrained=True, shuffle=True, array_path=config.array_dir, num_classes=4):
    kfold = KFold(n_splits = number_of_folds)


    # prepare dataset bij concatenating Train/Test part; we split later.
    dataloading = Dataloading(test_size=test_size, array_path=array_path, batch_size=batch_size, shuffle=shuffle)

    train_loss_per_fold = []
    eval_loss_per_fold = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataloading.dataloaders_combined)):
        # voor nu dummy maken om te kijken of KFold zelf werkt, 
        # daarna training_model() uit model_training toepassen
        for fold in range(number_of_folds):
            train_loss_per_epoch = [i for i in range(num_epochs)]
            epoch_loss_per_epoch = [i/2 for i in range(num_epochs)]
        train_loss_per_fold.append(train_loss_per_epoch)
        eval_loss_per_fold.append(eval_loss_per_fold)

    #calculate the average per epoch and variances

    average_train_loss_per_epoch = []
    average_eval_loss_per_epoch = []
    train_variance_per_epoch = []
    eval_variance_per_epoch = []

    for i in range(len(train_loss_per_fold[0])):
        total_train = 0
        total_eval = 0

        for j in range(len(train_loss_per_fold)):
            total_train += train_loss_per_fold[j][i]
            total_eval += eval_loss_per_fold[j][i]

        average_train = total_train/len(train_loss_per_fold)
        average_eval = total_eval/len(eval_loss_per_fold)
        
        average_train_loss_per_epoch.append(average_train)
        average_eval_loss_per_epoch.append(average_eval)

        train_variance_per_epoch.append(np.var(train_loss_per_fold[:][i]))
        eval_variance_per_epoch.append(np.var(eval_loss_per_fold[:][i]))
        
    return average_train_loss_per_epoch, average_eval_loss_per_epoch, train_variance_per_epoch, eval_variance_per_epoch



def evaluation_train():
    """change the learning rates, epochs and batch size here, not in training_model"""
    learning_rates = [0.001]
    num_epochs = 4
    batch_size = 8
    num_folds = 2
    # calculate the loss for each learning rate
    train_loss_per_lr = []
    eval_loss_per_lr = []
    train_var_per_lr = []
    eval_var_per_lr = []
    for learning_rate in learning_rates:
        train_loss_per_epoch, eval_loss_per_epoch, train_var_per_lr, eval_var_per_lr = kfold_training(number_of_folds=num_folds ,learning_rate = learning_rate, num_epochs = num_epochs, batch_size = batch_size)
        train_loss_per_lr.append(train_loss_per_epoch)
        eval_loss_per_lr.append(eval_loss_per_epoch)
        
        #plot the losses
        plt.plot(range(num_epochs), train_loss_per_epoch, label = f" training loss for learning rate {learning_rate}")
        plt.plot(range(num_epochs), eval_loss_per_epoch, label = f" evaluation loss for learning rate {learning_rate}")
    
    print(f"training loss per learning rate = {train_loss_per_lr}")
    print(f"evaluation loss per learning rate = {eval_loss_per_lr}")
    print(f"training variance per learning rate = {train_var_per_lr}")
    print(f"evaluation variance per learning rate = {eval_var_per_lr}")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.show()


 
    # save_slice_array(train_loss_per_lr, 'training loss per learning rate', currentdir)
    # save_slice_array(eval_loss_per_lr, 'evaluation loss per learning rate', currentdir)



if __name__ == "__main__":
    evaluation_train()