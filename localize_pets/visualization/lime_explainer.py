import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from localize_pets.loss_metric.iou import IOU
from localize_pets.transforms.transforms import process_bbox_image
from lime import lime_image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

class CallFunction:
    def __init__(self, model, transforms):
        self.model = model
        self.transforms = transforms
    def batch_predict(self, images):
        images_processed = []
        for i in images:
            ex_image, _ = process_bbox_image(i,
                                             None,
                                             self.transforms)
            images_processed.append(ex_image)
        explain_images = np.stack(images_processed, axis=0)
        return self.model(explain_images)

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

raw_image = cv2.imread(image_path)

assert os.path.exists(model_path), "Model path does not exist."
model = tf.keras.models.load_model(model_path, custom_objects={"IOU": IOU(name="iou")})
input_layer = model.layers[0].input
outs = model.output
concatenated = tf.concat(outs, axis=1)
model = tf.keras.Model(inputs=input_layer, outputs=concatenated)
print(model.summary())

caller = CallFunction(model, transforms)
pred_fn = caller.batch_predict
explainer = lime_image.LimeImageExplainer(verbose=True)
explanation = explainer.explain_instance(raw_image.astype('double'),
                                         pred_fn,
                                         labels=np.arange(0, 6),
                                         top_labels=6,
                                         hide_color=0.0,
                                         num_samples=100,
                                         batch_size=8)
temp, mask = explanation.get_image_and_mask(explanation.top_labels[3],
                                               positive_only=True,
                                               num_features=5,
                                               min_weight=1e-3, hide_rest=False)

img_boundry1 = mark_boundaries(temp/255., mask)
print(img_boundry1.shape, mask.shape)
plt.imsave('lime.jpg', img_boundry1)