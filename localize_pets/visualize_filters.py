import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from localize_pets.utils.misc import IOU


def compute_loss(img, filter_index):
    # We avoid border artifacts by only involving non-border pixels in the loss.
    activation = feature_extractor(img)
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)


@tf.function
def gradient_ascent_step(img, filter_index, learning_rate=10):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img


def initialize_image(width, height):
    img = tf.random.uniform((1, width, height, 3))
    return img


def deprocess_image(image):
    image = (image - tf.reduce_min(image)) / tf.reduce_max(
        (image - tf.reduce_min(image))
    )
    # Convert to RGB array
    image *= 255.0
    image = np.clip(image, 0, 255).astype("uint8")
    return image


def visualize_filters(
    filter_index, learning_rate, num_iterations, image_width, image_height
):
    img = initialize_image(image_height, image_width)
    for iteration in range(num_iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate)
    img = deprocess_image(img[0].numpy())
    return loss, img


description = "Inference script for object localization task on pets dataset"
parser = argparse.ArgumentParser(description=description)
parser.add_argument(
    "-m",
    "--model_path",
    default="save_checkpoint/pets_model/",
    type=str,
    help="Model path",
)
parser.add_argument(
    "-l", "--layer_name", default="max_pooling2d_4", type=str, help="Layer to visualize"
)
parser.add_argument(
    "-f",
    "--filter_index",
    default=1,
    type=int,
    help="Filter index of the layer to visualize",
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    default=1e-3,
    type=float,
    help="Learning rate to regenerate input image",
)
parser.add_argument(
    "-it",
    "--num_iterations",
    default=30,
    type=int,
    help="Number of iterations to regenerate input image",
)
parser.add_argument(
    "-iw", "--image_width", default=300, type=int, help="Input image width"
)
parser.add_argument(
    "-ih", "--image_height", default=300, type=int, help="Input image height"
)
args = parser.parse_args()
config = vars(args)
model_path = config["model_path"]
layer_name = config["layer_name"]
filter_idx = config["filter_index"]
learning_rate = config["learning_rate"]
num_iterations = config["num_iterations"]
image_width = config["image_width"]
image_height = config["image_height"]
assert os.path.exists(model_path), "Model path does not exist."
model = tf.keras.models.load_model(model_path, custom_objects={"IOU": IOU(name="iou")})
layer = model.get_layer(name=layer_name)
feature_extractor = tf.keras.models.Model(inputs=model.inputs, outputs=layer.output)
loss, img = visualize_filters(
    filter_idx, learning_rate, num_iterations, image_width, image_height
)
plt.imshow(img)
plt.savefig("visualize_filter_" + layer_name + ".jpg")
