import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from localize_pets.loss_metric.iou import IOU


def compute_loss(img, filter_index):
    # We avoid border artifacts by only
    # involving non-border pixels in the loss.
    activation = feature_extractor(img)
    filter_activation = activation[
        :, :, :, filter_index
    ]  # Included the entire map to avoid 0 loss at center pixel
    loss = tf.reduce_mean(filter_activation)
    return loss


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


def initialize_image(width, height, normalize):
    img = tf.random.uniform((1, width, height, 3)) * 255
    if normalize == "max":
        img = img / 255.0
    elif normalize == "same":
        img = img
    elif normalize == "vgg19":
        img = tf.keras.applications.vgg19.preprocess_input(img)
    elif normalize == "-+":
        a = -1
        b = 1
        numerator = ((img - tf.reduce_min(img)) * (b - a))
        denominator = (tf.reduce_max(img) - tf.reduce_min(img))
        img = a + (numerator / denominator)
    else:
        raise ValueError(
            "Normalization method unsupported %s" % normalize
        )
    print("Initialized image stats: ", tf.reduce_min(img), tf.reduce_max(img))

    return img


def deprocess_image(image):
    image = (image - tf.reduce_min(image)) / tf.reduce_max(
        (image - tf.reduce_min(image))
    )
    # Convert to RGB array
    image *= 255.0
    image = np.clip(image, 0, 255).astype("uint8")
    return image


def visualize_filters(filter_index,
                      learning_rate,
                      num_iterations,
                      image_width,
                      image_height,
                      normalize):
    img = initialize_image(image_height, image_width, normalize)
    for iteration in range(num_iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate)
        print("Loss at iteration %d: %f" % (iteration, loss))
    img = deprocess_image(img[0].numpy())
    return loss, img


description = "Inference script for object localization task on pets dataset"
parser = argparse.ArgumentParser(description=description)
parser.add_argument(
    "-m",
    "--model_path",
    default="save_checkpoint_efficientnet/pets_model/",
    type=str,
    help="Model path",
)
parser.add_argument(
    "-l", "--layer_name", default="conv2d", type=str,
    help="Layer to visualize"
)
parser.add_argument(
    "-f",
    "--filter_index",
    default=45,
    type=int,
    help="Filter index of the layer to visualize",
)
parser.add_argument(
    "--normalize",
    default="max",
    type=str,
    help="Normalization strategy. "
         "Available options: max, same, vgg19. "
         "Max for SimpleNet, VGG19 and same_scale for EfficientNet",
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    default=10,
    type=float,
    help="Learning rate to regenerate input image",
)
parser.add_argument(
    "-it",
    "--num_iterations",
    default=200,
    type=int,
    help="Number of iterations to regenerate input image",
)
parser.add_argument(
    "-iw", "--image_width", default=224, type=int, help="Input image width"
)
parser.add_argument(
    "-ih", "--image_height", default=224, type=int, help="Input image height"
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
normalize = config["normalize"]
assert os.path.exists(model_path), "Model path does not exist."
model = tf.keras.models.load_model(model_path,
                                   custom_objects={"IOU": IOU(name="iou")})
print(model.summary())
layer = model.get_layer(name=layer_name)
feature_extractor = tf.keras.models.Model(inputs=model.inputs,
                                          outputs=layer.output)
loss, img = visualize_filters(filter_idx,
                              learning_rate,
                              num_iterations,
                              image_width,
                              image_height,
                              normalize)
plt.imshow(img)
plt.savefig("visualize_filter_" +
            layer_name + "_fid" +
            str(filter_idx) + ".jpg")
