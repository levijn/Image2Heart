import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
import os
import sys
import inspect
from collections import OrderedDict
import PIL

from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50
import torch
import os
import inspect
import numpy as np
import PIL
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
from torchvision.transforms.functional import convert_image_dtype, resize
from torchvision.transforms import Resize
        
from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path


root_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
data_dir = os.path.join(os.path.dirname(os.path.dirname(root_dir)), "Data")
slice_array = os.path.join(data_dir, "slice_arrays")
slice_images = os.path.join(data_dir, "slice_images")
input_path = os.path.join(data_dir, "data")
model_path = os.path.join(input_path, "model")


def run():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    data_transforms = {
        'train':
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'validation':
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize
        ]),
    }

    image_datasets = {
        'train': 
        datasets.ImageFolder(os.path.join(input_path, "train"), data_transforms['train']),
        'validation': 
        datasets.ImageFolder(os.path.join(input_path, "validation"), data_transforms['validation'])
    }

    dataloaders = {
        'train':
        torch.utils.data.DataLoader(image_datasets['train'],
                                    batch_size=32,
                                    shuffle=True,
                                    num_workers=8),  # for Kaggle
        'validation':
        torch.utils.data.DataLoader(image_datasets['validation'],
                                    batch_size=32,
                                    shuffle=False,
                                    num_workers=8)  # for Kaggle
    }


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = models.segmentation.fcn_resnet50(pretrained = True, progress = True, num_classes = 4).to(device)
    model = models.resnet50(pretrained=True).to(device)

    for param in model.parameters():
        param.requires_grad = False   
        
    model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters())

    # def train_model(model, criterion, optimizer, num_epochs=3):
    #     for epoch in range(num_epochs):
    #         print('Epoch {}/{}'.format(epoch+1, num_epochs))
    #         print('-' * 10)

    #         for phase in ['train', 'validation']:
    #             if phase == 'train':
    #                 model.train()
    #             else:
    #                 model.eval()

    #             running_loss = 0.0
    #             running_corrects = 0

    #             for inputs, labels in dataloaders[phase]:
    #                 inputs = inputs.to(device)
    #                 labels = labels.to(device)

    #                 outputs = model(inputs)
    #                 loss = criterion(outputs, labels)

    #                 if phase == 'train':
    #                     optimizer.zero_grad()
    #                     loss.backward()
    #                     optimizer.step()

    #                 _, preds = torch.max(outputs, 1)
    #                 running_loss += loss.item() * inputs.size(0)
    #                 running_corrects += torch.sum(preds == labels.data)

    #             epoch_loss = running_loss / len(image_datasets[phase])
    #             epoch_acc = running_corrects.double() / len(image_datasets[phase])

    #             print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
    #                                                         epoch_loss,
    #                                                         epoch_acc))
    #     return model
        
    # model_trained = train_model(model, criterion, optimizer, num_epochs=3)
    
    # torch.save(model_trained.state_dict(), os.path.join(model_path, "weights.h5"))


    model = models.resnet50(pretrained=False).to(device)

    model.fc = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2)).to(device)
    #model = model()
    #model.load_state_dict(torch.load('model_state.pth')
    #model = torch.load(os.path.join(model_path, "weights.h5"))
    model.load_state_dict(torch.load(os.path.join(model_path, "weights.h5")))

    validation_path = os.path.join(input_path, "validation")
    alien_path = os.path.join(validation_path, "alien")
    predator_path = os.path.join(validation_path, "predator")

    validation_img_paths = [os.path.join(alien_path, "11.jpg"),
                            os.path.join(alien_path, "22.jpg")]
    img_list = [Image.open(os.path.join(input_path,  img_path)) for img_path in validation_img_paths]

    validation_batch = torch.stack([data_transforms['validation'](img).to(device)
                                    for img in img_list])

    pred_logits_tensor = model(validation_batch)
    pred_probs = torch.nn.functional.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()

    fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
    for i, img in enumerate(img_list):
        ax = axs[i]
        ax.axis('off')
        ax.set_title("{:.0f}% Alien, {:.0f}% Predator".format(100*pred_probs[i,0],
                                                                100*pred_probs[i,1]))
        ax.imshow(img)    
    # plt.show()
    
    img = F.to_pil_image(pred_probs[:,:])
    plt.imshow(img)
    plt.show()



if __name__ == '__main__':
    run()    
