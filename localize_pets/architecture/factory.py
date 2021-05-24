from localize_pets.architecture.efficientnet import EfficientNet
from localize_pets.architecture.vgg19 import VGG19
from localize_pets.architecture.simplenet import SimpleNet


class ArchitectureFactory():

    def __init__(self, name):
        self.name = name

    def factory(self):
        if self.name == 'EfficientNet':
            return EfficientNet
        elif self.name == 'VGG':
            return VGG19
        elif self.name == 'SimpleNet':
            return SimpleNet
        else:
            raise ValueError('Architecture name not implemented %s' % (self.name))
