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

def Intersect(label_array, output_array, num_classes=4):
    intersect_per_class = []
    label_occurence_per_class = []
    output_occurence_per_class = []
    for c in range(num_classes):
        intersect = 0
        label_occurence = 0
        output_occurence = 0

        for i in range(len(label_array)):
            if label_array[i] == c and output_array[i] == c:
                intersect += 1
                label_occurence += 1
                output_occurence += 1
            elif output_array[i] == c:
                output_occurence += 1
            elif label_array[i] == c:
                label_occurence += 1
        
        intersect_per_class.append(intersect)
        label_occurence_per_class.append(label_occurence)
        output_occurence_per_class.append(output_occurence)
    return (intersect_per_class, label_occurence_per_class, output_occurence_per_class)

def Dice(label_stack, output_stack, num_classes=4, bg_weight=0.05, smooth=1):
    weights = [bg_weight, (1-bg_weight)/3, (1-bg_weight)/3, (1-bg_weight)/3]

    label_list = []
    for k in range(label_stack.size(dim=0)):
        label_list.append(label_stack[k,:,:])
    output_list = convert_to_segmented_imgs(output_stack)

    total_dice = 0
    for i in range(len(label_list)):
        print(f"<-- Sample: {i} -->")
        label_f = K.flatten(tf.cast(label_list[i], dtype=float)).numpy()
        output_f = K.flatten(tf.cast(output_list[i], dtype=float)).numpy()

        print(f"Total number of parameters: {len(label_f)}")

        weighted_dice = 0
        intersect_per_class, label_occurence_per_class, output_occurence_per_class = Intersect(label_f, output_f)

        for c in range(num_classes):
            if label_occurence_per_class[c] == 0 and output_occurence_per_class[c] == 0:
                dice_class = 1
            else:
                dice_class = 2 * intersect_per_class[c] / (label_occurence_per_class[c] + output_occurence_per_class[c])
            # weight = (len(label_f)-label_occurence_per_class[c]) / len(label_f)
            weighted_dice += dice_class * weights[c]

            print(f"- Class: {c} || Intersect: {intersect_per_class[c]} | Label_occ: {label_occurence_per_class[c]} | Output_occ: {output_occurence_per_class[c]}")
            print(f"              Dice_class: {dice_class} | Weight: {weights[c]}")

        total_dice += weighted_dice
        print(f"Weighted Dice: {weighted_dice}\n")
    # print(f"Total dice of batch: {total_dice}")
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

    f = open("Dice_scores.txt", "w")


    #loading datafiles
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

    # initialize way to save loss
    train_loss_per_lr = []
    eval_loss_per_lr = []
    
    for LR in learning_rate:
        train_loss_per_epoch = []
        eval_loss_per_epoch = []
        #looping through epochs
        for epoch in range(num_epochs):
            print(f"<---------------------- Learning rate: {LR} --> Epoch: {epoch+1} ----------------------->\n")
            epoch_train_dice = 0.0
            epoch_eval_dice = 0.0
            fcn.train()
            # Model training loop
            print("....Going through training data....\n")
            
            for i_batch, batch in enumerate(dataloading.train_dataloader):
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
                
            print("....Going through testing data....\n")

            # Calculate validation loss after training
            fcn.eval()
            for i_batch, batch in enumerate(dataloading.test_dataloader):
                criterion = torch.nn.CrossEntropyLoss()
                data = F.convert_image_dtype(batch["image"], dtype=torch.float).to(device)
                target = batch["label"].to(device)
                output = fcn(data)

                epoch_eval_dice += Dice(batch["label"], output["out"].detach())
            

            train_dice = epoch_train_dice/len(dataloading.train_slicedata)
            eval_dice = epoch_eval_dice/len(dataloading.test_slicedata)
            f.write(f"____________ Epoch: {epoch+1} ____________\n")
            print("Training Dice:", train_dice)
            f.write(f"Training Dice: {train_dice} \n")
            print("Evaluation Dice:", eval_dice)
            f.write(f"Evaluation Dice: {eval_dice} \n")
            

        train_loss_per_lr.append(train_loss_per_epoch)
        eval_loss_per_lr.append(eval_loss_per_epoch)
        
        #saving calculated weights
        torch.save(fcn.state_dict(), os.path.join(currentdir, savingfile))
    
    # #plotting learningrates
    # plot_learningrate(train_loss_per_lr, eval_loss_per_lr, learning_rate)

def running_model(pretrained=False, num_classes=4, loadingfile="weights.h5"):
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
    fcn.load_state_dict(torch.load(os.path.join(currentdir, loadingfile)))

    total_dice = 0


    plt.rcParams["savefig.bbox"] = 'tight'

    #retrieving 1 image for training
    one_batch = None
    dataloading = Dataloading(test_size=0.2, array_path=config.array_dir, batch_size=4, shuffle=True)
    for i_batch, batch in enumerate(dataloading.test_dataloader):
        #remove the padding        
        print(f"Batch: {i_batch}")
        sample = batch["image"]
        sample = F.convert_image_dtype(sample, dtype=torch.float)

        fcn.eval()

        # normalized_sample = F.normalize(sample, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        output = fcn(sample)["out"]
        total_dice += Dice(batch["label"], output.detach())

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
    trained = True
    #Define the name of the weights file for saving or loading:
    savingfile = "weights_lr1_e10_z10_pad_norm.h5"
    loadingfile = "weights_lr1_e10_z10_pad_norm.h5"
    
    print("Transforms: Zoom, Padding, RGB, Tensor, Normalize, RemovePadding")
    if trained is False:
        learningrates = [0.001]
        training_model(pretrained=True, learning_rate=learningrates, batch_size=8, num_epochs=1, test_size=0.2, savingfile=savingfile)
        running_model(pretrained=True, loadingfile=loadingfile)
    elif trained is True:
        running_model(pretrained=True, loadingfile=loadingfile)
    

if __name__ == '__main__':
    import timeit
    
    start = timeit.default_timer()

    main()

    stop = timeit.default_timer()

    print('Time: ', stop - start) 