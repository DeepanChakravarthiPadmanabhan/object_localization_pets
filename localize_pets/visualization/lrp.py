import os
import argparse
import cv2
import innvestigate
import numpy as np
import tensorflow as tf
from localize_pets.loss_metric.iou import IOU
from localize_pets.transforms.transforms import process_bbox_image
import matplotlib.pyplot as plt


CLASS_MAPPING = {0: "Cat", 1: "Dog", 2: "XMIN", 3: "YMIN", 4: "XMAX", 5: "YMAX"}
description = "Inference script for object localization task on pets dataset"
parser = argparse.ArgumentParser(description=description)
parser.add_argument(
    "-dd",
    "--data_dir",
    default="/media/deepan/externaldrive1/datasets_project_repos/pets_data/",
    type=str,
    help="Data directory for training",
)
parser.add_argument(
    "--train_samples", default=2800, type=int, help="Sample set size for training data"
)
parser.add_argument(
    "--test_samples", default=800, type=int, help="Sample set size for testing data"
)
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
    "-l", "--layer_name", default="conv2d_1", type=str, help="Layer to visualize"
)
parser.add_argument(
    "--visualize_head", default="detection", type=str, help="Head to visualize"
)
parser.add_argument(
    "--visualize_idx", default=1, type=int, help="Index to visualize. Corresponds to the class."
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
dataset_path = config["data_dir"]
model_path = config["model_path"]
layer_name = config["layer_name"]
image_path = config["image_path"]
image_height = config["image_height"]
image_width = config["image_width"]
visualize_head = config["visualize_head"]
visualize_idx = config["visualize_idx"]

transform_resize = dict()
transform_normalize = dict()
if config["resize"]:
    transform_resize["resize"] = [config["image_height"], config["image_width"]]
if config["normalize"]:
    transform_normalize["normalize"] = config["normalize"]

assert os.path.exists(model_path), "Model path does not exist."
model = tf.keras.models.load_model(model_path, custom_objects={"IOU": IOU(name="iou")})
input_layer = model.layers[0].input
outs = model.output
concatenated = tf.concat(outs, axis=1)
model = tf.keras.Model(inputs=input_layer, outputs=concatenated)
print(model.summary())

raw_image = cv2.imread(image_path)
raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
input_image, _ = process_bbox_image(raw_image, None, transform_resize)
input_image, _ = process_bbox_image(input_image, None, transform_normalize)
input_image = input_image[np.newaxis]
input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
det = model(input_image).numpy()
print(det)
plt.imsave('raw_image.jpg', raw_image)

from innvestigate.analyzer.base import AnalyzerNetworkBase, ReverseAnalyzerBase, AnalyzerBase
import innvestigate.layers as ilayers
import numpy as np
import os

import keras
import keras.backend
import keras.models

import innvestigate
import innvestigate.applications.imagenet
import innvestigate.utils as iutils

class Example1(ReverseAnalyzerBase):
    def _default_reverse_mapping(self, Xs, Ys, reversed_Ys, reverse_state):
        # Create dense layer that applies the transpose of the forward pass
        layer = reverse_state["layer"]
        weight = layer.get_weights()[0]
        dense_transposed = keras.layers.Dense(weight.shape[0])
        dense_transposed.set_weights([weight.T])
        # Apply the layer
        reversed_Xs = dense_transposed(reversed_Ys)
        return reversed_Xs

    def _head_mapping(self, X):
        # Initialize the mapping with ones.
        return ilayers.OnesLike()(X)

class Example2(ReverseAnalyzerBase):
    def _default_reverse_mapping(self, Xs, Ys, reversed_Ys, reverse_state):
        # Create a layer that takes len(Xs) inputs and creates backward mapping
        # by applying the chain rule
        gradient = ilayers.GradientWRT(len(Xs))
        # Apply the gradient
        # (passing all tensors in one list to stay conform with keras interface)
        reversed_Xs = gradient(Xs + Ys + reversed_Ys)
        return reversed_Xs

    def _head_mapping(self, X):
        # Initialize the mapping with ones.
        return ilayers.OnesLike()(X)
tmp = getattr(innvestigate.applications.imagenet, os.environ.get("NETWORKNAME", "vgg16"))
net = tmp(load_weights=True, load_patterns='relu')
# Build the model.
model = keras.models.Model(inputs=net["in"], outputs=net["sm_out"])
model.compile(optimizer="adam", loss="categorical_crossentropy")
print(model.summary())

# analyzer = Example2(model)
# a = analyzer.analyze(input_image)
# print(a.shape)

analyzer = innvestigate.create_analyzer("lrp.epsilon", model)
analysis = analyzer.analyze(input_image)
print(analysis)
# Aggregate along color channels and normalize to [-1, 1]
a = analysis.sum(axis=np.argmax(np.asarray(analysis.shape) == 3))
a /= np.max(np.abs(a))
# Plot
plt.imshow(a[0], cmap="seismic", clim=(-1, 1))