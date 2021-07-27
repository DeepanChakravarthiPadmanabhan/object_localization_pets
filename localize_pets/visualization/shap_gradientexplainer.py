import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
import random
from localize_pets.loss_metric.iou import IOU
from localize_pets.transforms.transforms import process_bbox_image
from localize_pets.dataset.pets_detection import Pets_Detection
from tensorflow.compat.v1.keras.backend import get_session
import shap
import matplotlib.pyplot as plt
tf.compat.v1.disable_v2_behavior()

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
input_layer = model.layers[0].input
outs = model.output
concatenated = tf.concat(outs, axis=1)
model = tf.keras.Model(inputs=input_layer, outputs=concatenated)
print(model.summary())

trainval_data = Pets_Detection(config["data_dir"], "trainval")
trainval_dataset = trainval_data.load_data()
print("Total dataset items read: ", len(trainval_dataset))
random.shuffle(trainval_dataset)
train_dataset = trainval_dataset[: config["train_samples"]]
test_dataset = trainval_dataset[
    config["train_samples"] : config["train_samples"] + config["test_samples"]
]
print(
    "Samples in train and test dataset are %d and %d, respectively. "
    % (len(train_dataset), len(test_dataset))
)

num_background_images = 1
num_explain_images = 2

background_images_idx = np.random.choice(len(train_dataset),
                                         num_background_images,
                                         replace=False)
background_images = []
for i in background_images_idx:
    bg_image = cv2.imread(train_dataset[i]['image_path'])
    bg_image, _ = process_bbox_image(bg_image,
                                     None,
                                     transforms)
    background_images.append(bg_image)
background_images = np.stack(background_images,
                             axis=0)


explain_images_idx = np.random.choice(len(test_dataset),
                                      num_explain_images,
                                      replace=False)
explain_images = []
for i in explain_images_idx:
    ex_image = cv2.imread(test_dataset[i]['image_path'])
    ex_image, _ = process_bbox_image(ex_image,
                                     None,
                                     transforms)
    explain_images.append(ex_image)
explain_images = np.stack(explain_images, axis=0)

print(background_images.shape, explain_images.shape)

# explain how the input to the 7th layer of the model explains the top two classes
# TODO: Modify to visualize impact on the box and class outputs
def map2layer(x, layer):
    feed_dict = dict(zip([model.layers[0].input], [x.copy()]))
    return get_session().run(model.layers[layer].input, feed_dict)

layer = 3 # Layer out shape should be similar to input shape
print(model.layers[layer].name)
# shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
e = shap.GradientExplainer(
    (model.layers[layer].input, model.layers[-1].output),
    map2layer(explain_images, layer),
    local_smoothing=0)
shap_values, indexes = e.shap_values(map2layer(explain_images, layer), ranked_outputs=6)
index_names = np.vectorize(lambda x: CLASS_MAPPING[x])(indexes)
shap.image_plot(shap_values,
                explain_images,
                index_names,
                )
plt.savefig('image_plot.jpg')
