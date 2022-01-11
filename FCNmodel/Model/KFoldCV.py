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
from preprocess import save_slice_array, save_dict
from change_head import change_headsize
#from model_training import training_model
import config

#copy of training_model, without dataloading
def training_model(dataloading, num_epochs=10, batch_size=16, learning_rate=0.001, pretrained=True, shuffle=True, array_path=config.array_dir, num_classes=4):
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
    #dataloading = Dataloading(test_size=test_size, array_path=array_path, batch_size=batch_size, shuffle=shuffle)
    #creating fcn model
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
            print(f"Batch: {i_batch+1}")

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



def kfold_training(number_of_folds = 2, test_size=0.2, num_epochs=2, batch_size=16, learning_rate=0.001, pretrained=True, shuffle=True, array_path=config.array_dir, num_classes=4):
    kfold = KFold(n_splits = number_of_folds)


    # prepare dataset bij concatenating Train/Test part; we split later.
    dataloading = Dataloading(test_size=test_size, array_path=array_path, batch_size=batch_size, shuffle=shuffle)

    train_loss_per_fold = []
    eval_loss_per_fold = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataloading.dataloaders_combined)):
        print(f"----- FOLD {fold} -----")
        # voor nu dummy maken om te kijken of KFold zelf werkt, 
        # daarna training_model() uit model_training toepassen
        # train_loss_per_epoch = []
        # eval_loss_per_epoch = []
        # for i in range(num_epochs):
        #     train_loss_per_epoch.append(i*fold)
        #     eval_loss_per_epoch.append(i*fold)
        
        # print(f"train_loss: {train_loss_per_epoch}")
        # print(f"eval loss: {eval_loss_per_epoch}")
        # train_loss_per_fold.append(train_loss_per_epoch)
        # eval_loss_per_fold.append(eval_loss_per_epoch)

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

    # save the dictiobaries
    save_dict(train_loss_per_lr, currentdir, 'avg_train_loss_per_epoch')
    save_dict(train_var_per_lr, currentdir, 'var_train_loss_per_epoch')
    save_dict(eval_loss_per_lr, currentdir, 'avg_eval_loss_per_epoch')
    save_dict(eval_var_per_lr, currentdir, 'var_eval_loss_per_epoch')

    # plot the dictionaries
    for lr in train_loss_per_lr:
        plt.plot(range(len(train_loss_per_lr[lr])), train_loss_per_lr[lr], label = f'train loss for lr {lr}')
        plt.plot(range(len(eval_loss_per_lr[lr])), eval_loss_per_lr[lr], label = f'eval loss for lr {lr}')
        plt.plot(range(len(train_var_per_lr[lr])), train_var_per_lr[lr], label = f'train variance for lr {lr}')
        plt.plot(range(len(eval_var_per_lr[lr])), eval_var_per_lr[lr], label = f'eval variance for lr {lr}')

    plt.legend()
    plt.show()






if __name__ == "__main__":
    evaluation_train()