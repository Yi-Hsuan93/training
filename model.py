import timm
import torch
import torch.nn as nn

def build_model(model_name, num_classes):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model

if __name__ == "__main__":
    model = build_model('resnet50.tv_in1k', num_classes=10)