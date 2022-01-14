import os
import sys
import inspect

#torch imports
import torch
import torchvision.transforms.functional as F
from torchvision.models.segmentation import fcn_resnet50

#tensorflow/keras imports
import tensorflow

#adding needed folderpaths
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
prepdir = os.path.join(parentdir, "Preprocessing")
sys.path.append(prepdir)
sys.path.append(parentdir)

# custom imports
import config
from slicedataset import Dataloading
from change_head import change_headsize
from one_patient import convert_to_segmented_imgs


def IOU(label_stack, output_stack):
    """IoU fuction: uses tensorflow keras to calculate iou loss"""
    label_list = []
    for k in range(label_stack.size(dim=0)):
        label_list.append(label_stack[k,:,:])
    output_list = convert_to_segmented_imgs(output_stack)

    total_iou = 0
    for i in range(len(label_list)):
        metric = tensorflow.keras.metrics.MeanIoU(num_classes=4)
        metric.update_state(label_list[i], output_list[i])
        iou = metric.result().numpy()
        total_iou += iou
        # print(f"iou: {iou}")
    # print(f"Total iou of batch: {total_iou}")
    return total_iou


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
    #creating fcn model and loss function
    fcn = fcn_resnet50(pretrained=pretrained)
    criterion = torch.nn.CrossEntropyLoss()

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
    train_loss_iou_per_lr = []
    train_loss_cel_per_lr = []
    eval_loss_iou_per_lr = []
    eval_loss_cel_per_lr = []
    
    for LR in learning_rate:
        #create optimizer
        optimizer = torch.optim.Adam(fcn.parameters(), lr=LR)
        
        train_loss_iou_per_epoch = []
        train_loss_cel_per_epoch = []
        eval_loss_iou_per_epoch = []
        eval_loss_cel_per_epoch = []
        #looping through epochs
        for epoch in range(num_epochs):
            print(f"------ Learning rate: {LR} --> Epoch: {epoch+1} ------")
            epoch_train_iou = 0.0
            epoch_train_cel = 0.0
            epoch_eval_iou = 0.0
            epoch_eval_cel = 0.0
            
            fcn.train()
            # Model training loop
            print("Going through training data....")
            for i_batch, batch in enumerate(dataloading.train_dataloader):
                print(f"Batch: {i_batch}")
                data = F.convert_image_dtype(batch["image"], dtype=torch.float).to(device)
                target = batch["label"].to(device)
                optimizer.zero_grad()
                output = fcn(data)
                loss = criterion(output["out"], target.long())
                loss.backward()
                optimizer.step()
                
                epoch_train_cel += output["out"].shape[0]*loss.item()
                if i_batch%16 == 0:
                    # epoch_train_iou += IOU(batch["label"], output["out"].detach())
                    pass


            print("Going through testing data....")
            # Calculate validation loss after training
            fcn.eval()
            for i_batch, batch in enumerate(dataloading.test_dataloader):
                data = F.convert_image_dtype(batch["image"], dtype=torch.float).to(device)
                target = batch["label"].to(device)
                output = fcn(data)
                loss = criterion(output["out"], target.long())
                
                epoch_eval_cel += output["out"].shape[0]*loss.item()
                if i_batch%16 == 0:
                    # epoch_eval_iou += IOU(batch["label"], output["out"].detach())
                    pass
            

            train_iou = epoch_train_iou/(len(dataloading.train_slicedata)/16)
            train_cel = epoch_train_cel/len(dataloading.train_slicedata)
            eval_iou = epoch_eval_iou/(len(dataloading.test_slicedata)/16)
            eval_cel = epoch_eval_cel/len(dataloading.test_slicedata)
            print("Training Cross Entropy loss:", train_cel)
            print("Evaluation Cross Entropy loss:", eval_cel)
            print("Training IoU:", train_iou)
            print("Evaluation IoU:", eval_iou)

            train_loss_cel_per_epoch.append(train_cel)
            train_loss_iou_per_epoch.append(train_iou)
            eval_loss_cel_per_epoch.append(eval_cel)
            eval_loss_iou_per_epoch.append(eval_iou)

            
        train_loss_iou_per_lr.append(train_loss_iou_per_epoch)
        train_loss_cel_per_lr.append(train_loss_cel_per_epoch)
        eval_loss_iou_per_lr.append(eval_loss_iou_per_epoch)
        eval_loss_cel_per_lr.append(eval_loss_cel_per_epoch)
        
        #saving calculated weights
        #torch.save(fcn.state_dict(), os.path.join(currentdir, savingfile))
    
    print(train_loss_cel_per_lr, train_loss_iou_per_lr, eval_loss_cel_per_lr, eval_loss_iou_per_lr)
    loss_list = [train_loss_cel_per_lr, train_loss_iou_per_lr, eval_loss_cel_per_lr, eval_loss_iou_per_lr]
    loss_name_list = ["train cross entropy", "train iou", "evaluation cross entropy", "evaluation iou"]
    
    with open(os.path.join(currentdir, "results.txt"), "w") as f:
        for i, loss_type in enumerate(loss_list):
            f.write(f"{loss_name_list[i]}\n")
            for j, lr in enumerate(loss_type):
                f.write(str(learning_rate[j]) + " " + str(lr) + ",\n")



def main():
    #Define the name of the weights file for saving or loading:
    weightsfile = "weights_lr1_e3_z10_pad_norm.h5"
    
    print("Transforms: Zoom, Padding, RGB, Tensor, Normalize, RemovePadding")
    learningrates = [0.0001, 0.001, 0.01, 0.1]
    training_model(pretrained=True, learning_rate=learningrates, batch_size=16, num_epochs=40, test_size=0.3, savingfile=weightsfile)
    

if __name__ == '__main__':
    import timeit
    
    start = timeit.default_timer()

    main()

    stop = timeit.default_timer()

    print('Time: ', stop - start) 