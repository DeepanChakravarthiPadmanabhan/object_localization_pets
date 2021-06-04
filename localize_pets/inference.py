import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from localize_pets.utils.misc import display, CLASS_MAPPING
from localize_pets.transforms.transforms import process_bbox_image
from localize_pets.loss_metric.iou import IOU
from tensorflow.keras.applications.vgg16 import VGG16


def vgg_feature_visualization(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    model = VGG16()
    ixs = [2, 5, 9, 13, 17]
    outputs = [model.layers[i + 1].output for i in ixs]
    model = tf.keras.models.Model(inputs=model.inputs, outputs=outputs)
    model.summary()
    feature_maps = model.predict(image)
    # plot the output from each block
    square = 8
    for n, fmap in enumerate(feature_maps):
        # plot all 64 maps in an 8x8 squares
        print("shape of feature: ", fmap.shape)
        ix = 1
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(fmap[0, :, :, ix - 1], cmap="gray")
                ix += 1
        # show the figure
        plt.savefig("vgg_feature_map" + str(n) + ".jpg")
        plt.show()


def inference_model(
    image_path, model_path, model_name, image_width, image_height, normalize, resize
):
    image = cv2.imread(image_path)
    image = image.astype("float32")

    # Transform
    transforms = dict()
    if resize:
        transforms["resize"] = [image_height, image_width]
    elif normalize:
        transforms["normalize"] = normalize

    image, bbox = process_bbox_image(image, None, transforms)
    image = image[np.newaxis]

    model = tf.keras.models.load_model(
        model_path, custom_objects={"IOU": IOU(name="iou")}
    )
    print(model.summary())
    class_out, det_out = model(image)
    pet_class = CLASS_MAPPING[np.argmax(class_out.numpy())]
    pet_coord = det_out.numpy()[0]
    display((image[0] * 255).astype("uint8"), pet_coord, pet_class, None)


description = "Inference script for object localization task on pets dataset"
parser = argparse.ArgumentParser(description=description)
parser.add_argument(
    "-i",
    "--image_path",
    default="/home/deepan/Downloads/images/Abyssinian_21.jpg",
    type=str,
    help="Inference image path",
)
parser.add_argument(
    "-m",
    "--model_path",
    default="save_checkpoint_efficientnet/pets_model/",
    type=str,
    help="Model path",
)
parser.add_argument(
    "-mn",
    "--model_name",
    default="EfficientNet",
    type=str,
    help="Model name",
)
parser.add_argument(
    "-iw", "--image_width", default=224, type=int, help="Input image width"
)
parser.add_argument(
    "-ih", "--image_height", default=224, type=int, help="Input image height"
)
parser.add_argument(
    "--resize",
    default=True,
    type=bool,
    help="Whether to resize the image",
)
parser.add_argument(
    "--normalize",
    default="same_scale",
    type=str,
    help="Normalization strategy. Available options: max, same_scale",
)
args = parser.parse_args()
config = vars(args)
image_path = config["image_path"]
model_path = config["model_path"]
model_name = config["model_name"]
resize = config["resize"]
normalize = config["normalize"]
image_height = config["image_height"]
image_width = config["image_width"]
assert os.path.exists(model_path), "Model path does not exist."
assert os.path.exists(image_path), "Image path does not exist."
inference_model(
    image_path, model_path, model_name, image_width, image_height, normalize, resize
)
# vgg_feature_visualization(image_path)
