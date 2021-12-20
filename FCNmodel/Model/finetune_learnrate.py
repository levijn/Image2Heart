import torch
import os
import inspect
import numpy as np
import PIL
import matplotlib.pyplot as plt
import sys

from torchvision.models.segmentation import fcn_resnet50
import torchvision.transforms.functional as F

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


def training_model(test_size=0.2, num_epochs=10, batch_size=4, learning_rate=[0.001], pretrained=True, shuffle=True, array_path=config.array_dir, num_classes=4):
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
            epoch_train_loss = 0.0
            epoch_eval_loss = 0.0
            fcn.train()
            # Model training loop
            for i_batch, batch in enumerate(dataloading.train_dataloader):
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

        train_loss_per_lr.append(train_loss_per_epoch)
        eval_loss_per_lr.append(eval_loss_per_epoch)
        
        #saving calculated weights
        torch.save(fcn.state_dict(), os.path.join(currentdir, f"weights_lr{int(LR*1000)}.h5"))
    
    #plotting learningrates
    plot_learningrate(train_loss_per_lr, eval_loss_per_lr, learning_rate)

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
    fcn.load_state_dict(torch.load(os.path.join(currentdir, "weights_lr1.h5")))


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
    
    print("Transforms: Resizing, RGB, Tensor, Normalize")
    if trained is False:
        learningrates = [0.001]
        training_model(pretrained=True, learning_rate=learningrates, batch_size=16, num_epochs=10, test_size=0.2)
        running_model(pretrained=True)
    elif trained is True:
        running_model(pretrained=True)
    

if __name__ == '__main__':
    import timeit

    start = timeit.default_timer()

    main()

    stop = timeit.default_timer()

    print('Time: ', stop - start) 