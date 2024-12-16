import torch
import torch.nn as nn
from models.resnet import ResNet50
from models.noisy_resnet import noisy_ResNet50

class ModelWrapper(nn.Module):
    def __init__(self, name, model_name, num_classes, num_channels=3):
        super(ModelWrapper, self).__init__()
        self.model_name = model_name
        print("this si the value fo the model_name: ", model_name)
        self.name = name
        self.len = 0 
        self.loss = 0
        self.num_classes = num_classes

        if self.model_name == "noisy_resnet50":
            self.model = noisy_ResNet50(self.num_classes)
        elif self.model_name == "resnet50":
            self.model = ResNet50(self.num_classes)
        else:
            raise ValueError("Model not implemented")

    
