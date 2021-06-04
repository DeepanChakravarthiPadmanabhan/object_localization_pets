import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from localize_pets.visualization.utils import deprocess_image, save_image, to_rgb
from localize_pets.loss_metric.iou import IOU
from localize_pets.transforms.transforms import process_bbox_image

class GradCAM:
    def __init__(self, model, layer_name=None):
        """
        Model: pre-softmax layer (logit layer)
        :param model:
        :param layer_name:
        """
        self.model = model
        self.layer_name = layer_name

        if self.layer_name == None:
            self.layer_name = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self,
                        image,
                        upsample_size,
                        visualize_idx=0,
                        visualize_head='detection',
                        eps=1e-5):
        grad_model = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (conv_outs, preds) = grad_model(inputs)
            if visualize_head == 'detection':
                loss = preds[1][:, visualize_idx]
            else:
                loss = preds[0][:, visualize_idx]

        # Compute gradients with automatic differentiation
        grads = tape.gradient(loss, conv_outs)
        conv_outs = conv_outs[0]
        grads = grads[0]
        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) * tf.constant(eps))

        # Compute neuron importance
        weights = tf.reduce_mean(norm_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outs), axis=-1)

        # Apply relu
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        cam = cv2.resize(cam, upsample_size, cv2.INTER_LINEAR)
        # Convert to 3d
        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])
        return cam3

def overlay_gradCAM(img, cam3):
    img = img[0]
    cam3 = np.uint8(255 * cam3)
    cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)
    new_image = 0.3 * cam3 + 0.5 * img
    return (new_image * 255 / new_image.max()).astype("uint8")


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
    "-l", "--layer_name", default="conv2d", type=str, help="Layer to visualize"
)
parser.add_argument(
    "--visualize_head", default="classification", type=str, help="Head to visualize"
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
visualize_head = config["visualize_head"]
visualize_idx = config["visualize_idx"]

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

grad_cam = GradCAM(model, layer_name)
cam3 = grad_cam.compute_heatmap(image=image,
                         upsample_size=(image_height, image_width),
                         visualize_idx=visualize_idx,
                         visualize_head=visualize_head)
cam3_overlaid = overlay_gradCAM(image, cam3)
cam3_overlaid = to_rgb(cam3_overlaid)
save_image('./cam3.jpg', cam3_overlaid)
