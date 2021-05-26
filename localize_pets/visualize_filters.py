import argparse
import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from localize_pets.loss_metric.iou import IOU
from localize_pets.objectives.objectives import Loss


def gradient_ascent_step(img, filter_index, learning_rate, objective):
    loss_object = Loss()
    loss_fn = getattr(Loss, objective)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    img = tf.Variable(img)
    with tf.GradientTape() as tape:
        tape.watch(img)
        activation = feature_extractor(img)
        loss = loss_fn(loss_object, activation, filter_index)
        loss = -loss # Gradient ascent
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    optimizer.apply_gradients(zip([grads], [img]))

    # with tf.GradientTape() as tape:
    #     tape.watch(img)
    #     activation = feature_extractor(img)
    #     loss = loss_fn(loss_object, activation, filter_index)
    # grads = tape.gradient(loss, img)
    # grads = tf.math.l2_normalize(grads)
    # img += learning_rate * grads

    return loss, img


def initialize_image(width,
                     height,
                     normalize):
    img = tf.random.uniform((1, width, height, 3)) * 255
    if normalize == "max":
        print('Normalization applied: %s' % normalize)
        img = img / 255.0
    elif normalize == "same":
        print('Normalization applied: %s' % normalize)
        img = img
    elif normalize == "vgg19":
        print('Normalization applied: %s' % normalize)
        img = tf.keras.applications.vgg19.preprocess_input(img)
    elif normalize == "-+":
        print('Normalization applied: %s' % normalize)
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
    mean = tf.math.reduce_mean(image)
    std = tf.math.reduce_std(image)
    image -= mean
    image /= (std + 1e-05)
    image += 0.5
    # Convert to RGB array
    image *= 255.0
    image = np.clip(image, 0, 255).astype("uint8")
    return image


def visualize_filters(filter_index,
                      learning_rate,
                      num_iterations,
                      image_width,
                      image_height,
                      normalize,
                      objective='l2'):
    img = initialize_image(image_height, image_width, normalize)
    for iteration in range(num_iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate, objective)
        print("Loss at iteration %d: %f" % (iteration, loss))
    img = deprocess_image(img[0].numpy())
    return loss, img


def plot_visualized_filters(img_list, filter_idx_list):
    columns = 4
    rows = int(len(img_list) / columns) + 1
    num_images = len(img_list)
    idx = 0
    plt.figure(figsize=(10, 6))
    for cur_row in range(rows):
        for cur_col in range(columns):
            if idx < num_images:
                ax = plt.subplot(rows, columns, idx + 1)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(img_list[idx])
                plt.title(filter_idx_list[idx])
                print('Plotted image %d' % (idx + 1))
                idx += 1
    plt.tight_layout()
    plt.savefig("visualize_filter_" +
                layer_name + "_" +
                objective + ".jpg")
    plt.show()


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
    "-l", "--layer_name", default="conv2d_1", type=str,
    help="Layer to visualize"
)
parser.add_argument(
    "-f",
    "--filter_index",
    default=None,
    type=int,
    help="Filter index of the layer to visualize",
)
parser.add_argument(
    "-n",
    "--normalize",
    default="max",
    type=str,
    help="Normalization strategy. "
         "Available options: max, same, vgg19. "
         "Max for SimpleNet, VGG19 and same_scale for EfficientNet",
)
parser.add_argument(
    "-o",
    "--objective",
    default="custom",
    type=str,
    help="Objective for loss calculation. "
         "Available options: l1, l2, mean, totalvariation.",
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    default=20,
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
    "-nv",
    "--num_visualizations",
    default=10,
    type=int,
    help="Number of filters to visualize",
)
parser.add_argument(
    "--max_filters",
    default=256,
    type=int,
    help="Max number of filters / channels in the layer chosen to visualize",
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
objective = config["objective"]
num_visualizations = config["num_visualizations"]
max_filters = config["max_filters"]

assert os.path.exists(model_path), "Model path does not exist."
model = tf.keras.models.load_model(model_path,
                                   custom_objects={"IOU": IOU(name="iou")})
print(model.summary())

layer = model.get_layer(name=layer_name)
feature_extractor = tf.keras.models.Model(inputs=model.inputs,
                                          outputs=layer.output)
img_list = list()
filter_idx_list = list()
if filter_idx:
    print('Visualizing filter index: ', filter_idx)
    loss, img = visualize_filters(filter_idx,
                                  learning_rate,
                                  num_iterations,
                                  image_width,
                                  image_height,
                                  normalize,
                                  objective)
    filter_idx_list.append(filter_idx)
    img_list.append(img)
else:
    for i in range(num_visualizations):
        filter_idx = random.randint(0, max_filters)
        print('Visualizing filter index: ', filter_idx)
        loss, img = visualize_filters(filter_idx,
                                      learning_rate,
                                      num_iterations,
                                      image_width,
                                      image_height,
                                      normalize,
                                      objective)
        filter_idx_list.append(filter_idx)
        img_list.append(img)

plot_visualized_filters(img_list, filter_idx_list)
