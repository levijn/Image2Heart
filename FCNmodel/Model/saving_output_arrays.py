import torch
import os
import inspect
import numpy as np
import PIL
import matplotlib.pyplot as plt
import sys
import tensorflow as tf

from torchvision.models.segmentation import fcn_resnet50
import torchvision.transforms.functional as F

from torchmetrics import IoU
from sklearn.metrics import jaccard_score
from keras import backend as K

#adding needed folderpaths
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
prepdir = os.path.join(parentdir, "Preprocessing")
sys.path.append(prepdir)
sys.path.append(parentdir)

#importing needed files
import config
from preprocess import save_slice_array, load_slice_array
from slicedataset import Dataloading
from change_head import change_headsize
from to_image import create_segmentated_img
from one_patient import convert_to_segmented_imgs

import numpy as np
np.set_printoptions(threshold=np.inf)


def plot_learningrate(train_loss, eval_loss, learningrates):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    for i, lr in enumerate(learningrates):
        ax.plot(train_loss[i], label=f"train {lr}")
        ax.plot(eval_loss[i], label=f"eval {lr}")
    plt.legend()
    plt.savefig(os.path.join(currentdir, "learningrate.png"))
    plt.show()
    
def plot_results(input, output):
    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 2
    for i in range(1, columns*rows):
        normalized_masks = torch.nn.functional.softmax(output, dim=1)
        img = F.to_pil_image(input)
        fig.add_subplot(rows, columns, 1)
        plt.imshow(img)
        img2 = F.to_pil_image(normalized_masks[0,i,:,:])
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img2)
    plt.show()

def list2string(lst):
    string = ""
    for row in lst:
        row_str = ""
        for pixel in row:
            row_str += f"{str(int(pixel))} "
        string += row_str + "\n"
    return string


def Dice(label_stack, output_stack, smooth=1):
    label_list = []
    for k in range(label_stack.size(dim=0)):
        label_list.append(label_stack[k,:,:])
    output_list = convert_to_segmented_imgs(output_stack)
    f = open("checking_output.txt", "w")

    total_dice = 0
    for i in range(len(label_list)):
        f.write(f"<============================== Image: {i} ===========================>\n")
        f.write("<------------------------- Label List ----------------------------->\n")
        lst = label_list[i].tolist()
        f.write(list2string(lst))
        f.write("<------------------------- Output List ----------------------------->\n")
        f.write(list2string(output_list[i]))
        label_f = K.flatten(tf.cast(label_list[i], dtype=float))
        output_f = K.flatten(tf.cast(output_list[i], dtype=float))
        intersection = K.sum(label_f * output_f)
        dice = (2. * intersection * smooth) / (K.sum(label_f) + K.sum(output_f) + smooth)
        total_dice += dice
        f.write(f"Dice: {dice}\n")
    print(f"Total dice of batch: {total_dice}")
    return total_dice


def training_model(test_size=0.2, num_epochs=10, batch_size=4, learning_rate=[0.001], pretrained=True, shuffle=True, array_path=config.array_dir, num_classes=4, savingfile="weights.h5"):
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
        savingfile: File name for saving the data.
    """
    #loading datafiles
    dataloading = Dataloading(test_size=test_size, array_path=array_path, batch_size=batch_size, shuffle=shuffle)
    #creating fcn model
    fcn = fcn_resnet50(pretrained=pretrained)

    f = open("checking_output.txt", "w")


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

    # initialize way to save loss
    train_loss_per_lr = []
    eval_loss_per_lr = []
    
    for LR in learning_rate:
        train_loss_per_epoch = []
        eval_loss_per_epoch = []
        #looping through epochs
        counte = 0
        for epoch in range(num_epochs):
            if counte == 1:
                    break
            print(f"------ Learning rate: {LR} --> Epoch: {epoch+1} ------")
            epoch_train_dice = 0.0
            epoch_eval_dice = 0.0
            fcn.train()
            # Model training loop
            f.write("........................................................Going through training data......................................................\n")
            
            count = 0
            for i_batch, batch in enumerate(dataloading.train_dataloader):
                if count == 2:
                    break
                f.write(f"||||||||||||||||||||||||||||||| Batch: {i_batch} |||||||||||||||||||||||||||||||\n")
                print(f"Batch: {i_batch}")
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(fcn.parameters(), lr=LR)

                data = F.convert_image_dtype(batch["image"], dtype=torch.float).to(device)
                target = batch["label"].to(device)
                optimizer.zero_grad()
                output = fcn(data)
                loss = criterion(output["out"], target.long())
                loss.backward()
                optimizer.step()
                
                epoch_train_dice += Dice(batch["label"], output["out"].detach())

                count += 1
                
            f.write("........................................................Going through testing data......................................................\n")

            # Calculate validation loss after training
            fcn.eval()
            countt = 0
            for i_batch, batch in enumerate(dataloading.test_dataloader):
                if countt == 2:
                    break
                criterion = torch.nn.CrossEntropyLoss()
                data = F.convert_image_dtype(batch["image"], dtype=torch.float).to(device)
                target = batch["label"].to(device)
                output = fcn(data)

                epoch_eval_dice += Dice(batch["label"], output["out"].detach())
                countt += 1
            

            train_dice = epoch_train_dice/len(dataloading.train_slicedata)
            eval_dice = epoch_eval_dice/len(dataloading.test_slicedata)
            print("Training Dice:", train_dice)
            print("Evaluation Dice:", eval_dice)

            counte += 1

        train_loss_per_lr.append(train_loss_per_epoch)
        eval_loss_per_lr.append(eval_loss_per_epoch)
        
        #saving calculated weights
        torch.save(fcn.state_dict(), os.path.join(currentdir, savingfile))
    
    # #plotting learningrates
    # plot_learningrate(train_loss_per_lr, eval_loss_per_lr, learning_rate)

def running_model(pretrained=False, num_classes=4, savingfile="weights.h5"):
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
    fcn.load_state_dict(torch.load(os.path.join(currentdir, savingfile)))


    plt.rcParams["savefig.bbox"] = 'tight'

    #retrieving 1 image for training
    one_batch = None
    dataloading = Dataloading(test_size=0.2, array_path=config.array_dir, batch_size=4, shuffle=True)
    for i_batch, batch in enumerate(dataloading.test_dataloader):
        #remove the padding
        one_batch = batch
        break

    sample = one_batch["image"]
    sample = F.convert_image_dtype(sample, dtype=torch.float)

    fcn.eval()

    # normalized_sample = F.normalize(sample, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    output = fcn(sample)["out"]
    normalized_masks = torch.nn.functional.softmax(output, dim=1)
    
    # # Displaying input image
    # image = one_batch["image"][0,0,:,:]
    # img = F.to_pil_image(image)
    # plt.imshow(img)
    # plt.show()
    
    # # Displaying probabilities of the num_classes
    # for i in range(normalized_masks.shape[1]):
    #     img = F.to_pil_image(normalized_masks[0,i,:,:])
    #     plt.imshow(img)
    #     plt.show()


def main():
    #set to True if the model has been trained with the weights stored at "weights.h5", False otherwise:
    trained = False
    #Define the name of the weights file for saving or loading:
    weightsfile = "weights_lr1_e10_z10_pad_norm.h5"
    
    print("Transforms: Zoom, Padding, RGB, Tensor, Normalize, RemovePadding")
    if trained is False:
        learningrates = [0.001]
        training_model(pretrained=True, learning_rate=learningrates, batch_size=8, num_epochs=2, test_size=0.2, savingfile=weightsfile)
        running_model(pretrained=True, savingfile=weightsfile)
    elif trained is True:
        running_model(pretrained=True, savingfile=weightsfile)
    

if __name__ == '__main__':
    import timeit
    
    start = timeit.default_timer()

    main()

    stop = timeit.default_timer()

    print('Time: ', stop - start) 