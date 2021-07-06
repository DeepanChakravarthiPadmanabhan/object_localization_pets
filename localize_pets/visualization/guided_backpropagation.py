import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from localize_pets.visualization.utils import plot_inference_and_visualization, deprocess_image, to_rgb
from localize_pets.loss_metric.iou import IOU
from localize_pets.transforms.transforms import process_bbox_image
from localize_pets.utils.misc import CLASS_MAPPING
from localize_pets.architecture.factory import ArchitectureFactory


@tf.custom_gradient
def guided_relu(x):
    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy
    return tf.nn.relu(x), grad


class GuidedBackpropagation:
    def __init__(self, model, layer_name=None, visualize_idx=None):
        self.model = model
        self.layer_name = layer_name
        self.visualize_idx = visualize_idx
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
        if self.visualize_idx:
            gbModel = Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(self.layer_name).output[:, self.visualize_idx - 1], self.model.output],
            )
        else:
            gbModel = Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(self.layer_name).output, self.model.output],
            )

        base_model_layers = [act_layer for act_layer in gbModel.layers[1].layers if hasattr(act_layer, 'activation')]
        head_layers = [
            layer for layer in gbModel.layers[1:] if hasattr(layer, "activation")
        ]
        all_layers = base_model_layers + head_layers
        for layer in all_layers:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guided_relu

        if 'class' in self.layer_name:
            gbModel.get_layer(self.layer_name).activation = None
        return gbModel

    def guided_backpropagation(self, image, upsample_size):
        """Guided backpropagation method for visualizing input saliency."""
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            tape.watch(inputs)
            conv_outs, preds = self.gbModel(inputs)

        print('Conv outs shape: ', conv_outs.shape)
        grads = tape.gradient(conv_outs, inputs)[0]
        saliency = cv2.resize(np.asarray(grads), upsample_size)
        pet_class = CLASS_MAPPING[np.argmax(preds[0].numpy())]
        pet_coord = preds[1].numpy()[0]
        return saliency, pet_coord, pet_class


description = "Inference script for object localization task on pets dataset"
parser = argparse.ArgumentParser(description=description)
parser.add_argument(
    "-i",
    "--image_path",
    default="/media/deepan/externaldrive1/datasets_project_repos/pets_data/images/basset_hound_163.jpg",
    type=str,
    help="Image path",
)
parser.add_argument(
    "-m",
    "--model_path",
    default="save_checkpoint_resnet/pets_model/",
    type=str,
    help="Model path",
)
parser.add_argument(
    "-l", "--layer_name", default="class_out", type=str, help="Layer to visualize"
)
parser.add_argument(
    "--visualize_idx", default=None, type=int, help="Index to visualize. Corresponds to the class."
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
    default="resnet50",
    type=str,
    help="Normalization strategy. "
    "Available options: max, same, vgg19, resnet50. "
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
visualize_idx = config["visualize_idx"]
transforms = dict()
if config["resize"]:
    transforms["resize"] = [config["image_height"], config["image_width"]]
if config["normalize"]:
    transforms["normalize"] = config["normalize"]

image = cv2.imread(image_path)
image, _ = process_bbox_image(image, None, transforms)
image = image[np.newaxis]

assert os.path.exists(model_path), "Model path does not exist."
model = tf.keras.models.load_model(model_path, custom_objects={"IOU": IOU(name="iou")})
print(model.summary())

gbp = GuidedBackpropagation(model, layer_name, visualize_idx)
gbp_image, pet_bbox, pet_class = gbp.guided_backpropagation(image, (image_height, image_width))
gbp_image = deprocess_image(gbp_image)
gbp_image = to_rgb(gbp_image)
plot_inference_and_visualization(image=image[0],
                                 pet_bbox=pet_bbox,
                                 pet_class=pet_class,
                                 saliency=gbp_image,
                                 visualization='gbp')
