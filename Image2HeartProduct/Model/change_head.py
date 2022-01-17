from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.segmentation.fcn import FCNHead


def change_headsize(model, num_classes):
    """Replaces the 'head' with one that outputs 4 classes and returns the model"""
    new_classifier = FCNHead(2048, num_classes)
    new_aux_classifier = FCNHead(1024, num_classes)
    model.classifier = new_classifier
    model.aux_classifier = new_aux_classifier
    return model


def main():
    fcn = fcn_resnet50(pretrained=True)
    change_headsize(fcn, 4)

if __name__ == '__main__':
    main()