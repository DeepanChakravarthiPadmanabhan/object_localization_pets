import cv2
import matplotlib.pyplot as plt
import tensorflow as tf


CLASS_MAPPING = {0: "Cat", 1: "Dog"}


def save_fig(fig, filename):
    fig.savefig(filename)


def display(image, bbox, pet_class, mask=None):
    start_point = (int(bbox[0]), int(bbox[1]))
    end_point = (int(bbox[2]), int(bbox[3]))
    image = cv2.rectangle(image, start_point, end_point, (255, 255, 0), 2)
    fig = plt.figure(figsize=(12, 8))
    if mask != None:
        plt.subplot(121)
    plt.imshow(image)
    plt.title(pet_class)
    if mask != None:
        plt.subplot(122)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(mask))
        plt.title("Mask")
    save_fig(fig, "inference.jpg")
    plt.show()


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
        elif transforms["normalize"] == "same_scale":
            new_image = new_image
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
