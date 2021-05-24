import cv2
import tensorflow as tf


def process_bbox_image(image, bbox, transforms):
    new_image = image
    new_bbox = bbox

    if "resize" in transforms.keys():
        new_height = transforms["resize"][0]
        new_width = transforms["resize"][1]
        new_image, new_bbox = resize_image(image, bbox, new_width, new_height)

    if "normalize" in transforms.keys():
        if transforms["normalize"] == "max":
            new_image = new_image / 255.0
        elif transforms["normalize"] == "same":
            new_image = new_image
        elif transforms["normalize"] == "vgg19":
            new_image = tf.keras.applications.vgg19.preprocess_input(new_image)
        else:
            raise ValueError(
                "Normalization method unsupported %s" % transforms["normalize"]
            )

    return new_image, new_bbox


def resize_image(image, bbox, new_width, new_height):
    new_bbox = bbox
    width, height = image.shape[1], image.shape[0]
    new_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
    )
    width_factor = new_width / width
    height_factor = new_height / height
    if bbox:
        xmin, ymin, xmax, ymax = bbox
        xmin = xmin * width_factor
        xmax = xmax * width_factor
        ymin = ymin * height_factor
        ymax = ymax * height_factor
        new_bbox = [xmin, ymin, xmax, ymax]

    return new_image, new_bbox
