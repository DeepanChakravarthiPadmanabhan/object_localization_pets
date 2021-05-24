import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def lr_schedule(epoch, lr):
    if (epoch + 1) % 8 == 0:
        lr *= 0.2
    return max(lr, 3e-07)


def plot_bounding_box(image, gt_cords, pred_cords, norm=False):
    if norm:
        image *= 255.0
        image = image.astype("uint8")
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = gt_cords
    draw.rectangle((x1, y1, x2, y2), outline="green", width=3)

    x1, y1, x2, y2 = pred_cords
    draw.rectangle((x1, y1, x2, y2), outline="red", width=3)

    return image


def test_model(model, test_datagen):
    sample = random.randint(0, len(test_datagen) - 1)
    example, label = test_datagen[sample]
    x = example["image"]
    y = label["class_out"]
    box = label["box_out"]

    pred_y, pred_box = model.predict(x)
    image = x[0]

    pred_coords = pred_box[0]
    gt_coords = box[0]

    pred_class = np.argmax(pred_y[0])
    gt_class = np.argmax(y[0])

    image = plot_bounding_box(image, gt_coords, pred_coords, True)
    color = "green" if gt_class == pred_class else "red"

    plt.imshow(image)
    plt.xlabel(f"Pred: {pred_class}", color=color)
    plt.ylabel(f"GT: {gt_class}", color=color)
    plt.xticks([])
    plt.yticks([])


def test(model, test_datagen, epoch):
    fig = plt.figure(figsize=(16, 4))
    for i in range(0, 6):
        plt.subplot(1, 6, i + 1)
        test_model(model, test_datagen)
    fig.savefig(str(epoch) + ".jpg")
    plt.show()

class ShowTestImages(tf.keras.callbacks.Callback):
    def __init__(self, test_datagen):
        self.test_datagen = test_datagen

    def on_epoch_end(self, epoch, logs=None):
        test(self.model, self.test_datagen, epoch)
