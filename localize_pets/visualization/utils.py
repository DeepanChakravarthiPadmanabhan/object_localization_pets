import numpy as np
import cv2
import matplotlib.pyplot as plt


def deprocess_image(image):
    image = image.copy()
    mean = image.mean()
    std = image.std()
    image -= mean
    image /= std + 1e-05
    image *= 0.25
    # clip to [0, 1]
    image += 0.5
    image = np.clip(image, 0, 1)
    # Convert to RGB array
    image *= 255.0
    image = np.clip(image, 0, 255).astype("uint8")
    return image


def to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(save_path_and_name, image):
    plt.imsave(save_path_and_name, image)
