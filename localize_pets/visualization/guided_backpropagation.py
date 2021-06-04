import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from localize_pets.visualization.utils import deprocess_image, save_image, to_rgb
from localize_pets.loss_metric.iou import IOU
from localize_pets.transforms.transforms import process_bbox_image


@tf.custom_gradient
def guided_relu(x):
    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy

    return tf.nn.relu(x), grad


class GuidedBackpropagation:
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name
        if self.layer_name == None:
            self.layer_name = self.find_target_layer()
        self.gbModel = self.build_guided_model()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError(
            "Could not find 4D layer. Cannot apply guided backpropagation."
        )

    def build_guided_model(self):
        gbModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output],
        )
        layer_dict = [
            layer for layer in gbModel.layers[1:] if hasattr(layer, "activation")
        ]
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guided_relu
        return gbModel

    def guided_backpropagation(self, images, upsample_size):
        """Guided backpropagation method for visualizing input saliency."""
        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            outputs = self.gbModel(inputs)
        grads = tape.gradient(outputs, inputs)[0]
        saliency = cv2.resize(np.asarray(grads), upsample_size)
        return saliency


description = "Inference script for object localization task on pets dataset"
parser = argparse.ArgumentParser(description=description)
parser.add_argument(
    "-i",
    "--image_path",
    default="/home/deepan/Downloads/sample/images/Abyssinian_12.jpg",
    type=str,
    help="Image path",
)
parser.add_argument(
    "-m",
    "--model_path",
    default="save_checkpoint_vgg_2/pets_model/",
    type=str,
    help="Model path",
)
parser.add_argument(
    "-l", "--layer_name", default="dense_1", type=str, help="Layer to visualize"
)
parser.add_argument(
    "-iw", "--image_width", default=224, type=int, help="Input image width"
)
parser.add_argument(
    "-ih", "--image_height", default=224, type=int, help="Input image height"
)
parser.add_argument(
    "-n",
    "--normalize",
    default="vgg19",
    type=str,
    help="Normalization strategy. "
    "Available options: max, same, vgg19. "
    "Max for SimpleNet, VGG19 and same_scale for EfficientNet",
)
parser.add_argument(
    "--resize",
    default=True,
    type=bool,
    help="Whether to resize the image",
)

args = parser.parse_args()
config = vars(args)
model_path = config["model_path"]
layer_name = config["layer_name"]
image_path = config["image_path"]
image_height = config["image_height"]
image_width = config["image_width"]

transforms = dict()
if config["resize"]:
    transforms["resize"] = [config["image_height"], config["image_width"]]
elif config["normalize"]:
    transforms["normalize"] = config["normalize"]

image = cv2.imread(image_path)
image, _ = process_bbox_image(image, None, transforms)
image = image[np.newaxis]

assert os.path.exists(model_path), "Model path does not exist."
model = tf.keras.models.load_model(model_path, custom_objects={"IOU": IOU(name="iou")})
print(model.summary())

gbp = GuidedBackpropagation(model, layer_name)
gbp_image = gbp.guided_backpropagation(image, (image_height, image_width))
gbp_image = deprocess_image(gbp_image)
gbp_image = to_rgb(gbp_image)
save_image("./gb.jpg", gbp_image)
