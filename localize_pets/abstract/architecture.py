from abc import ABC, abstractmethod


class Architecture(ABC):
    def __init__(self,
                 backbone,
                 feature_extraction,
                 image_width,
                 image_height):
        self.backbone = backbone
        self.feature_extraction = feature_extraction
        self.image_width = image_width
        self.image_height = image_height

    @property
    def backbone(self):
        pass

    @backbone.setter
    def backbone(self, backbone):
        self._backbone = backbone

    @backbone.getter
    def backbone(self):
        return self._backbone

    @property
    def feature_extraction(self):
        pass

    @feature_extraction.setter
    def feature_extraction(self, feature_extraction):
        self._feature_extraction = feature_extraction

    @feature_extraction.getter
    def feature_extraction(self):
        return self._feature_extraction

    @property
    def image_width(self):
        pass

    @image_width.setter
    def image_width(self, image_width):
        self._image_width = image_width

    @image_width.getter
    def image_width(self):
        return self._image_width

    @property
    def image_height(self):
        pass

    @image_height.setter
    def image_height(self, image_height):
        self._image_height = image_height

    @image_height.getter
    def image_height(self):
        return self._image_height

    @abstractmethod
    def model(self):
        pass
