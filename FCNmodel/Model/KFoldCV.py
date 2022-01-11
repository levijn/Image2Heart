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

# import necessary functions
from slicedataset import Dataloading
from preprocess import save_slice_array, save_dict, load_dict
from change_head import change_headsize
from model_training import training_model
import config




def kfold_training(number_of_folds = 2, test_size=0.2, num_epochs=2, batch_size=16, learning_rate=0.001, pretrained=True, shuffle=True, array_path=config.array_dir, num_classes=4):
    """makes use of training model, but makes its own dataloaders, based on KFold Cross Validation"""
    kfold = KFold(n_splits = number_of_folds)


    # prepare dataset bij concatenating Train/Test part; we split later.
    dataloading = Dataloading(test_size=test_size, array_path=array_path, batch_size=batch_size, shuffle=shuffle)

    train_loss_per_fold = []
    eval_loss_per_fold = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataloading.dataloaders_combined)):
        print(f"----- FOLD {fold} -----")

        # use training_model() uit model_training.py
        train_loss_per_epoch, eval_loss_per_epoch = training_model(dataloading=dataloading, num_epochs=num_epochs, batch_size = batch_size, learning_rate=learning_rate, pretrained=pretrained, shuffle=shuffle, array_path=array_path, num_classes=num_classes)

        # add losses to loss_per_fold
        train_loss_per_fold.append(train_loss_per_epoch)
        eval_loss_per_fold.append(eval_loss_per_epoch)

    print(f"all train losses for learning rate {learning_rate}: {train_loss_per_fold}")


    # calculate the average and variance of all the folds
    # put them in lists for each epoch
    avg_train_loss_per_epoch = []
    avg_eval_loss_per_epoch = []
    var_train_loss_per_epoch = []
    var_eval_loss_per_epoch = []

    for i in range(len(train_loss_per_fold[0])):
        #make lists of the losses with folds, for each epoch (i.e. flip the lists)
        flipped_train = [loss[i] for loss in train_loss_per_fold]
        flipped_eval = [loss[i] for loss in eval_loss_per_fold]

        # calculate the averages of the losses
        avg_train_loss_per_epoch.append(sum(flipped_train)/len(flipped_train))
        avg_eval_loss_per_epoch.append(sum(flipped_eval)/len(flipped_eval))

        # calculate the variances
        var_train_loss_per_epoch.append(np.var(flipped_train))
        var_eval_loss_per_epoch.append(np.var(flipped_eval))

    print(f"average train loss for each epoch: {avg_train_loss_per_epoch}")
    print(f"with a variance of: {var_train_loss_per_epoch}")
    print(f"average train loss for each epoch: {avg_train_loss_per_epoch}")
    print(f"with a variance of: {var_train_loss_per_epoch}")


    return avg_train_loss_per_epoch, avg_eval_loss_per_epoch, var_train_loss_per_epoch, var_eval_loss_per_epoch

  

def evaluation_train():
    """change the learning rates, epochs and batch size here, not in training_model"""
    learning_rates = [0.001]
    num_epochs = 10
    batch_size = 12
    num_folds = 3
    # calculate the loss for each learning rate
    train_loss_per_lr = {}
    eval_loss_per_lr = {}
    train_var_per_lr = {}
    eval_var_per_lr = {}
    for lr in learning_rates:
        train_loss_per_epoch, eval_loss_per_epoch, train_var_per_epoch, eval_var_per_epoch = kfold_training(number_of_folds=num_folds ,learning_rate = lr, num_epochs = num_epochs, batch_size = batch_size)
        train_loss_per_lr[lr] = train_loss_per_epoch
        eval_loss_per_lr[lr] = eval_loss_per_epoch
        train_var_per_lr[lr] = train_var_per_epoch
        eval_var_per_lr[lr] = eval_var_per_epoch

    print(f"finished, train_loss_per_lr = {train_loss_per_lr}")
    print(f"with variances of {train_var_per_lr}")

    # save the dictionaries
    save_dict(train_loss_per_lr, currentdir, 'avg_train_loss_per_epoch')
    save_dict(train_var_per_lr, currentdir, 'var_train_loss_per_epoch')
    save_dict(eval_loss_per_lr, currentdir, 'avg_eval_loss_per_epoch')
    save_dict(eval_var_per_lr, currentdir, 'var_eval_loss_per_epoch')


def get_graphs():
    train_loss_per_lr = load_dict(os.path.join(currentdir,"avg_train_loss_per_epoch"))
    eval_loss_per_lr = load_dict(os.path.join(currentdir,'avg_eval_loss_per_epoch'))
    train_var_per_lr = load_dict(os.path.join(currentdir,'var_train_loss_per_epoch'))
    eval_var_per_lr = load_dict(os.path.join(currentdir,'var_eval_loss_per_epoch'))


    # plot the dictionaries
    for lr in train_loss_per_lr:
        plt.plot(range(len(train_loss_per_lr[lr])), train_loss_per_lr[lr], label = f'train loss for lr {lr}')
        plt.plot(range(len(eval_loss_per_lr[lr])), eval_loss_per_lr[lr], label = f'eval loss for lr {lr}')
        plt.plot(range(len(train_var_per_lr[lr])), train_var_per_lr[lr], label = f'train variance for lr {lr}')
        plt.plot(range(len(eval_var_per_lr[lr])), eval_var_per_lr[lr], label = f'eval variance for lr {lr}')

    plt.legend()
    plt.show()





def main():
    """set trained to False the first time, after that you can re-use those values"""
    trained = False
    if trained is False:
        evaluation_train()
        get_graphs()
    else:
        get_graphs()


if __name__ == "__main__":
    main()