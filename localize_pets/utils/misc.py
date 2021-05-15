import random

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageDraw

CLASS_MAPPING = {0: 'Cat', 1: 'Dog'}

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
        plt.title('Mask')
    save_fig(fig, 'inference.jpg')
    plt.show()

def plot_bounding_box(image, gt_cords, pred_cords, norm=False):
    if norm:
        image *= 255.
        image = image.astype('uint8')
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = gt_cords
    draw.rectangle((x1, y1, x2, y2), outline='green', width=3)

    x1, y1, x2, y2 = pred_cords
    draw.rectangle((x1, y1, x2, y2), outline='red', width=3)

    return image


def lr_schedule(epoch, lr):
    if (epoch + 1) % 20 == 0:
        lr *= 0.2
    return max(lr, 3e-07)


class IOU(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super(IOU, self).__init__(**kwargs)
        self.iou = self.add_weight(name='iou', initializer='zeros')
        self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
        self.num_ex = self.add_weight(name='num_ex', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        def get_box(y):
            x1, y1, x2, y2 = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
            return x1, y1, x2, y2

        def get_area(x1, y1, x2, y2):
            return tf.math.abs(x2 - x1) * tf.math.abs(y2 - y1)

        gt_x1, gt_y1, gt_x2, gt_y2 = get_box(y_true)
        p_x1, p_y1, p_x2, p_y2 = get_box(y_pred)

        i_x1 = tf.maximum(gt_x1, p_x1)
        i_y1 = tf.maximum(gt_y1, p_y1)
        i_x2 = tf.minimum(gt_x2, p_x2)
        i_y2 = tf.minimum(gt_y2, p_y2)

        i_area = get_area(i_x1, i_y1, i_x2, i_y2)
        u_area = get_area(gt_x1, gt_y1, gt_x2, gt_y2) + get_area(p_x1, p_y1, p_x2, p_y2) - i_area
        iou = tf.math.divide(i_area, u_area)
        self.num_ex.assign_add(1)
        self.total_iou.assign_add(tf.reduce_mean(iou))
        self.iou = tf.math.divide(self.total_iou, self.num_ex)

    def result(self):
        return self.iou

    def reset_state(self):
        # Called at end of each epoch
        self.iou = self.add_weight(name='iou', initializer='zeros')
        self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
        self.num_ex = self.add_weight(name='num_ex', initializer='zeros')

def test_model(model, test_datagen):
    sample = random.randint(0, len(test_datagen) - 1)
    example, label = test_datagen[sample]
    x = example['image']
    y = label['class_out']
    box = label['box_out']

    pred_y, pred_box = model.predict(x)
    image = x[0]

    pred_coords = pred_box[0]
    gt_coords = box[0]

    pred_class = np.argmax(pred_y[0])
    gt_class = np.argmax(y[0])

    image = plot_bounding_box(image, gt_coords, pred_coords, True)
    color = 'green' if gt_class == pred_class else 'red'

    plt.imshow(image)
    plt.xlabel(f'Pred: {pred_class}', color=color)
    plt.ylabel(f'GT: {gt_class}', color=color)
    plt.xticks([])
    plt.yticks([])

def test(model, test_datagen, epoch):
  fig = plt.figure(figsize=(16, 4))
  for i in range(0, 6):
    plt.subplot(1, 6, i + 1)
    test_model(model, test_datagen)
  fig.savefig(str(epoch)+'.jpg')
  plt.show()

class ShowTestImages(tf.keras.callbacks.Callback):
    def __init__(self, test_datagen):
        self.test_datagen = test_datagen
    def on_epoch_end(self, epoch, logs=None):
        test(self.model, self.test_datagen, epoch)
