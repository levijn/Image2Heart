import os
import sys
from pathlib import Path

#torch imports
import torch
import torchvision.transforms.functional as F
from torchvision.models.segmentation import fcn_resnet50

#adding needed folderpaths
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
sys.path.append(os.path.join(parent_dir, "Preprocessing"))
sys.path.append(os.path.join(parent_dir, "Helperfunctions"))

# custom imports
import config
from slicedataset import Dataloading
from change_head import change_headsize


def training_model(test_size=0.2, num_epochs=10, batch_size=4, learning_rate=[0.001], pretrained=True, shuffle=True, array_path=config.array_dir, num_classes=4):
    """Trains the model with different learning rates. Saves the model per learning rate and saves all loss data per epoch.
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
    # Loading datafiles
    dataloading = Dataloading(test_size=test_size, array_path=array_path, batch_size=batch_size, shuffle=shuffle)
    # Creating fcn model and loss function
    fcn = fcn_resnet50(pretrained=pretrained)

    # Setting model to trainingmode and set the device
    fcn.train()
    device = "cuda"
    
    # Can set the background weight different from other classes. The other weights are all equal to each other.
    bg_w = 0.25
    fg_w = (1-bg_w)/3
    weights = torch.Tensor([bg_w, fg_w, fg_w, fg_w]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    
    # Freezing its parameters
    for param in fcn.parameters():
        param.requires_grad = False
    
    # Change head to output 4 classes
    fcn = change_headsize(fcn, num_classes).to(device)
    
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in fcn.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in fcn.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    # Initialize way to save loss.
    train_loss_cel_per_lr = []
    eval_loss_cel_per_lr = []
    
    for LR in learning_rate:
        # Create optimizer
        optimizer = torch.optim.Adam(fcn.parameters(), lr=LR)
        
        train_loss_cel_per_epoch = []
        eval_loss_cel_per_epoch = []
        
        # Looping through epochs
        for epoch in range(num_epochs):
            print(f"------ Learning rate: {LR} --> Epoch: {epoch+1} ------")
            
            epoch_train_cel = 0.0
            epoch_eval_cel = 0.0
            
            fcn.train()
            # Model training loop
            print("Going through training data....")
            for i_batch, batch in enumerate(dataloading.train_dataloader):
                data = F.convert_image_dtype(batch["image"], dtype=torch.float).to(device)
                target = batch["label"].to(device)
                optimizer.zero_grad()
                output = fcn(data)
                loss = criterion(output["out"], target.long())
                loss.backward()
                optimizer.step()
                
                epoch_train_cel += output["out"].shape[0]*loss.item()
                
            print("Going through testing data....")
            # Calculate validation loss after training
            fcn.eval()
            for i_batch, batch in enumerate(dataloading.test_dataloader):
                data = F.convert_image_dtype(batch["image"], dtype=torch.float).to(device)
                target = batch["label"].to(device)
                output = fcn(data)
                loss = criterion(output["out"], target.long())
                
                epoch_eval_cel += output["out"].shape[0]*loss.item()

            # Adding epoch loss to lists. This is the average loss per epoch -> total_loss/amount_of_data
            train_cel = epoch_train_cel/len(dataloading.train_slicedata)
            eval_cel = epoch_eval_cel/len(dataloading.test_slicedata)
            
            print("Training Cross Entropy loss:", train_cel)
            print("Evaluation Cross Entropy loss:", eval_cel)

            train_loss_cel_per_epoch.append(train_cel)
            eval_loss_cel_per_epoch.append(eval_cel)

        
        train_loss_cel_per_lr.append(train_loss_cel_per_epoch)
        eval_loss_cel_per_lr.append(eval_loss_cel_per_epoch)
        
        # Saving calculated weights. Name is the learning rate*10000 and the number of epochs.
        torch.save(fcn.state_dict(), os.path.join(current_dir, f"weights_lr{str(int(LR*10000))}_e{str(num_epochs)}.h5"))
    
    # Saving the evaluation data to a file.
    loss_list = [train_loss_cel_per_lr, eval_loss_cel_per_lr]
    loss_name_list = ["Train cross entropy", "Evaluation cross entropy"]
    
    with open(os.path.join(current_dir, "results.txt"), "w") as f:
        for i, loss_type in enumerate(loss_list):
            f.write(f"{loss_name_list[i]}\n")
            for j, lr in enumerate(loss_type):
                f.write(str(learning_rate[j]) + " = " + str(lr) + "\n")



def main():
    print("Transforms: Zoom, Padding, RGB, Tensor, Normalize")
    learningrates = [0.001]
    training_model(pretrained=True, 
                   learning_rate=learningrates, 
                   batch_size=16, 
                   num_epochs=15, 
                   test_size=0.3
                   )
    

if __name__ == '__main__':
    import timeit
    
    start = timeit.default_timer()

    main()

    stop = timeit.default_timer()

    print('Time: ', stop - start) 