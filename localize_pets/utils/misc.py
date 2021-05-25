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
    if mask:
        plt.subplot(121)
    plt.imshow(image)
    plt.title(pet_class)
    if mask:
        plt.subplot(122)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(mask))
        plt.title("Mask")
    save_fig(fig, "inference.jpg")
    plt.show()
