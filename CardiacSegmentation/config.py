#!/usr/bin/env python
import os
import inspect

# Directory of the UK Biobank data
data_root = '/data/asclepios/share/private/ukbiobank-hearts-296-converted'

# Directory of the CardiacSegmentationPropagation project
# code_root = 'C:\Users\daans\OneDrive\Documenten\Werktuigbouwkunde\Jaar 3\Minor AI\Capstone -Image2Heart\Image2Heart\CardiacSegmentation'

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

code_root = parentdir

# ROI-net
roi_net_initial_lr = 1e-4
roi_net_decay_rate = 1.0
roi_net_batch_size = 16
roi_net_imput_img_size = 128
roi_net_epochs = 50



# LVRV-net
lvrv_net_initial_lr = 1e-4
lvrv_net_decay_rate = 1.0
lvrv_net_batch_size = 16
lvrv_net_imput_img_size = 192
lvrv_net_epochs = 80


# LV-net
lv_net_initial_lr = 1e-4
lv_net_decay_rate = 1.0
lv_net_batch_size = 16
lv_net_imput_img_size = 192
lv_net_epochs = 80

