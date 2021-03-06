from localize_pets.architecture.efficientnet import EfficientNet
from localize_pets.architecture.vgg19 import VGG19
from localize_pets.architecture.resnet import Resnet50
from localize_pets.architecture.simplenet import SimpleNet


class ArchitectureFactory:
    def __init__(self, name):
        self.name = name

    def factory(self):
        if self.name == "EfficientNet":
            return EfficientNet
        elif self.name == "VGG19":
            return VGG19
        elif self.name == "SimpleNet":
            return SimpleNet
        elif self.name == "Resnet50":
            return Resnet50
        else:
            raise ValueError("Architecture name not implemented %s" % (self.name))
