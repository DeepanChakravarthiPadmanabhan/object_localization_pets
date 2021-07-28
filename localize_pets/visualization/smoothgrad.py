import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from localize_pets.visualization.utils import to_rgb, \
    plot_inference_and_visualization, get_mpl_colormap, save_image, deprocess_image
from localize_pets.visualization.integrated_gradients import IntegratedGradients
from localize_pets.visualization.guided_backpropagation import GuidedBackpropagation
from localize_pets.loss_metric.iou import IOU
from localize_pets.transforms.transforms import process_bbox_image

def SmoothGrad_IG(model,
                  layer_name,
                  visualize_idx,
                  method,
                  image,
                  stdev_spread=0.15,
                  nsamples=25,
                  magnitude=True):

    stdev = stdev_spread * (np.max(image) - np.min(image))
    total_gradients = np.zeros_like(image, dtype=np.float32)
    for i in range(nsamples):
        print('SAMPLE: ', i)
        noise = np.random.normal(0, stdev, image.shape)
        image_noise = image + noise

        if method == 'IG':
            baseline = tf.zeros(shape=(1, 224, 224, 3))
            m_steps = 50
            ig = IntegratedGradients(model, layer_name, visualize_idx)
            grad = ig.integrated_gradients(baseline=baseline, image=image_noise,
                                           m_steps=m_steps, batch_size=4)
        elif method == 'Guided Backpropagation':
            gbp = GuidedBackpropagation(model, layer_name, visualize_idx)
            gbp_image, pet_bbox, pet_class = gbp.guided_backpropagation(image_noise, (image_height, image_width))
            grad = gbp_image[np.newaxis]

        if magnitude:
            total_gradients += (grad * grad)
        else:
            total_gradients += grad

    return total_gradients / nsamples


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
    "-l", "--layer_name", default="box_out", type=str, help="Layer to visualize"
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

image = cv2.imread(image_path)
image, _ = process_bbox_image(image, None, transforms)
image = image[np.newaxis]

assert os.path.exists(model_path), "Model path does not exist."
model = tf.keras.models.load_model(model_path, custom_objects={"IOU": IOU(name="iou")})
print(model.summary())


save_path = 'attributions_smooth.jpg'

grad = SmoothGrad_IG(model,
                     layer_name,
                     visualize_idx,
                     method='IG',
                     image=image)
save_image(save_path, grad[0])