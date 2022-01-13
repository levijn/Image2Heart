import torch
import os
import inspect
import numpy as np
import PIL
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import torch

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

# def IOU(label_stack, output_stack, n_classes=4):
#     label_list = []
#     for k in range(label_stack.size(dim=0)):
#         label_list.append(label_stack[k,:,:])
#     output_list = convert_to_segmented_imgs(output_stack)

#     total_iou = 0
#     for i in range(len(label_list)):
#         # m = tf.keras.metrics.MeanIoU(num_classes=num_classes)
#         # m.update_state(label_list[i], output_list[i])
#         # total_iou += m.result().numpy()

        
#         jac = jaccard_score(label_list[i], output_list[i])
#         print(jac)
#     return total_iou

def IOU(label_stack, output_stack):
    label_list = []
    for k in range(label_stack.size(dim=0)):
        label_list.append(label_stack[k,:,:])
    output_list = convert_to_segmented_imgs(output_stack)

    total_iou = 0
    for i in range(len(label_list)):
        metric = tf.keras.metrics.MeanIoU(num_classes=4)
        metric.update_state(label_list[i], output_list[i])
        iou = metric.result().numpy()
        total_iou += iou
        print(f"iou: {iou}")
    print(f"Total iou of batch: {total_iou}")
    return total_iou

# werkt niet
# def jaccard(label_stack, output_stack):
#     jac = IoU(num_classes=4)
#     seg_output = convert_to_segmented_imgs(output_stack)
#     seg_output_ts = []
#     for img in seg_output:
#         seg_output_ts.append(torch.tensor(img, dtype=torch.int32))
#     pred = torch.stack(seg_output_ts, dim=0)
#     target = label_stack.int()
#     print(pred.dtype)
#     print(target.dtype)
#     jacindex = jac(output_stack, label_stack)
#     return jacindex.item()
    

# SMOOTH = 1e-6

# def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
#     # You can comment out this line if you are passing tensors of equal shape
#     # But if you are passing output from UNet or something it will most probably
#     # be with the BATCH x 1 x H x W shape
#     seg_output = convert_to_segmented_imgs(outputs)
#     seg_output_ts = []
#     for img in seg_output:
#         seg_output_ts.append(torch.tensor(img, dtype=torch.int32))
#     outputs = torch.stack(seg_output_ts)
#     print(outputs.size())
#     print(labels.size())
    
#     outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
#     intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
#     union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
#     iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
#     thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
#     return thresholded  # Or thresholded.mean() if you are interested in average across the batch


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
            print(f"------ Learning rate: {LR} --> Epoch: {epoch+1} ------")
            epoch_train_iou = 0.0
            epoch_eval_iou = 0.0
            epoch_train_loss = 0.0
            epoch_eval_loss = 0.0
            fcn.train()
            # Model training loop
            print("Going through training data....")
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
                
                epoch_train_loss += output["out"].shape[0]*loss.item()
                # epoch_train_iou += IOU(batch["label"], output["out"].detach())
                print(iou_pytorch(output["out"], batch["label"]))
            print("Going through testing data....")

            # Calculate validation loss after training
            fcn.eval()
            for i_batch, batch in enumerate(dataloading.test_dataloader):
                criterion = torch.nn.CrossEntropyLoss()
                data = F.convert_image_dtype(batch["image"], dtype=torch.float).to(device)
                target = batch["label"].to(device)
                output = fcn(data)
                loss = criterion(output["out"], target.long())
                
                epoch_eval_loss += output["out"].shape[0]*loss.item()
                # epoch_eval_iou += IOU(batch["label"], output["out"].detach())
            

            train_iou = epoch_train_iou/len(dataloading.train_slicedata)
            eval_iou = epoch_eval_iou/len(dataloading.test_slicedata)
            print("Training IoU:", train_iou)
            print("Evaluation IoU:", eval_iou)
            
            train_loss = epoch_train_loss/len(dataloading.train_slicedata)
            eval_loss = epoch_eval_loss/len(dataloading.test_slicedata)
            print("Training loss:", train_loss)
            print("Evaluation loss:", eval_loss)

        # train_loss_per_lr.append(train_loss_per_epoch)
        # eval_loss_per_lr.append(eval_loss_per_epoch)
        
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
    weightsfile = "weights_lr1_e5_z10_pad_norm.h5"
    
    print("Transforms: Zoom, Padding, RGB, Tensor, Normalize, RemovePadding")
    if trained is False:
        learningrates = [0.001]
        training_model(pretrained=True, learning_rate=learningrates, batch_size=8, num_epochs=5, test_size=0.2, savingfile=weightsfile)
        running_model(pretrained=True, savingfile=weightsfile)
    elif trained is True:
        running_model(pretrained=True, savingfile=weightsfile)
    

if __name__ == '__main__':
    import timeit
    
    start = timeit.default_timer()

    main()

    stop = timeit.default_timer()

    print('Time: ', stop - start) 