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
#from model_training import training_model
import config


def training_model(test_size=0.2, num_epochs=10, batch_size=4, learning_rate=[0.001], pretrained=True, shuffle=True, array_path=config.array_dir, num_classes=4, dataloading = None):
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
    if dataloading is None:
        dataloading = Dataloading(test_size=test_size, array_path=array_path, batch_size=batch_size, shuffle=shuffle)
    #creating fcn model
    fcn = fcn_resnet50(pretrained=pretrained)

    # setting model to trainingmode and set the device
    fcn.train()
    device = "cuda"
    
    # freezing its parameters
    for param in fcn.parameters():
        param.requires_grad = False
    
    # change head to output 4 classes
    fcn = change_headsize(fcn, 4).to(device)

    
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in fcn.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in fcn.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    train_loss_per_epoch = []
    eval_loss_per_epoch = []
    #looping through epochs
    for epoch in range(num_epochs):
        print(f"------ Learning rate: {learning_rate} --> Epoch: {epoch+1} ------")
        epoch_train_loss = 0.0
        epoch_eval_loss = 0.0
        fcn.train()
        # Model training loop
        for i_batch, batch in enumerate(dataloading.train_dataloader):
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(fcn.parameters(), lr=learning_rate)

            data = F.convert_image_dtype(batch["image"], dtype=torch.float).to(device)
            target = batch["label"].to(device)
            optimizer.zero_grad()
            output = fcn(data)
            loss = criterion(output["out"], target.long())
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += output["out"].shape[0]*loss.item()
        
        # Calculate validation loss after training
        fcn.eval()
        for i_batch, batch in enumerate(dataloading.test_dataloader):
            criterion = torch.nn.CrossEntropyLoss()
            data = F.convert_image_dtype(batch["image"], dtype=torch.float).to(device)
            target = batch["label"].to(device)
            output = fcn(data)
            loss = criterion(output["out"], target.long())
            epoch_eval_loss += output["out"].shape[0]*loss.item()
        
        train_loss = epoch_train_loss/len(dataloading.train_slicedata)
        eval_loss = epoch_eval_loss/len(dataloading.test_slicedata)
        train_loss_per_epoch.append(train_loss)
        eval_loss_per_epoch.append(eval_loss)
        print("Training loss:", train_loss)
        print("Evaluation loss:", eval_loss)

        #saving calculated weights
        torch.save(fcn.state_dict(), os.path.join(currentdir, "weights_lr1_e10_pad_norm.h5"))
    
    #plotting learningrates
    return train_loss_per_epoch, eval_loss_per_epoch



def kfold_training(number_of_folds = 2, test_size=0.2, num_epochs=2, batch_size=16, learning_rate=0.001, pretrained=True, shuffle=True, array_path=config.array_dir, num_classes=4):
    """makes use of training model, but makes its own dataloaders, based on KFold Cross Validation"""
    kfold = KFold(n_splits = number_of_folds)


    # prepare dataset bij concatenating Train/Test part; we split later.
    dataloading = Dataloading(test_size=test_size, array_path=array_path, batch_size=batch_size, shuffle=shuffle)

    train_loss_per_fold = []
    eval_loss_per_fold = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataloading.dataloaders_combined)):
        print(f"----- FOLD {fold+1} -----")

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

  


def evaluation_train(learning_rates = [0.001], num_epochs = 5, batch_size = 8, num_folds = 3):
    """uses kfold_training() in order to get the average loss and variance for KFold
    Saves the losses as dictionaries"""

    # calculate the loss for each learning rate
    train_loss_per_lr = {}
    eval_loss_per_lr = {}
    train_var_per_lr = {}
    eval_var_per_lr = {}
    for lr in learning_rates:
        print(f"----- LEARNING RATE {lr} -----")
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
    """Uses the dictionaries made in evaluation_train() and plots them"""
    train_loss_per_lr = load_dict(os.path.join(currentdir,"avg_train_loss_per_epoch"))
    eval_loss_per_lr = load_dict(os.path.join(currentdir,'avg_eval_loss_per_epoch'))
    train_var_per_lr = load_dict(os.path.join(currentdir,'var_train_loss_per_epoch'))
    eval_var_per_lr = load_dict(os.path.join(currentdir,'var_eval_loss_per_epoch'))


    # plot the dictionaries
    fig, axs = plt.subplots(2,1)
    for lr in train_loss_per_lr:
        x_axis = range(1, len(train_loss_per_lr[lr]) + 1)
        axs[0].plot(x_axis, train_loss_per_lr[lr], label = f'train loss for lr {lr}')
        axs[0].plot(x_axis, eval_loss_per_lr[lr], label = f'eval loss for lr {lr}')
        axs[1].plot(x_axis, train_var_per_lr[lr], label = f'train variance for lr {lr}')
        axs[1].plot(x_axis, eval_var_per_lr[lr], label = f'eval variance for lr {lr}')

    fig.suptitle('Average losses and variances')

    axs[0].set_title('Average loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()


    axs[1].set_title('Average variance')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(os.path.join(currentdir, "KFold_losses.png"))
    plt.show()





def main():
    """set trained to False the first time, after that you can re-use those values"""
    learning_rates = [0.001]
    num_epochs = 10
    batch_size = 16
    num_folds = 5
    trained = True
    if trained is False:
        evaluation_train(learning_rates = learning_rates, num_epochs=num_epochs, batch_size=batch_size, num_folds=num_folds)
        get_graphs()
    else:
        get_graphs()


if __name__ == "__main__":
    main()

