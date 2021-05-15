import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from localize_pets.utils.misc import IOU, display, CLASS_MAPPING
from tensorflow.keras.applications.vgg16 import VGG16


def vgg_feature_visualization(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path,
                                                  target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.
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
        print('shape of feature: ', fmap.shape)
        ix = 1
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(fmap[0, :, :, ix - 1], cmap='gray')
                ix += 1
        # show the figure
        plt.savefig('vgg_feature_map' + str(n) + '.jpg')
        plt.show()


def inference_model(image_path, model_path):
    image = cv2.imread(image_path)
    image = image.astype('float32')
    image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_NEAREST) / 255.
    image = image[np.newaxis]
    model = tf.keras.models.load_model(model_path, custom_objects={'IOU': IOU(name='iou')})
    print(model.summary())
    class_out, det_out = model(image)
    pet_class = CLASS_MAPPING[np.argmax(class_out.numpy())]
    pet_coord = det_out.numpy()[0]
    display((image[0] * 255).astype('uint8'), pet_coord, pet_class, None)

description = 'Inference script for object localization task on pets dataset'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-i', '--image_path', default='/home/deepan/Downloads/images/Abyssinian_21.jpg', type=str,
                    help='Inference image path')
parser.add_argument('-m', '--model_path', default='save_checkpoint/pets_model/', type=str,
                    help='Model path')
args = parser.parse_args()
config = vars(args)
image_path = config['image_path']
model_path = config['model_path']
assert os.path.exists(model_path), "Model path does not exist."
assert os.path.exists(image_path), "Image path does not exist."
inference_model(image_path, model_path)
# vgg_feature_visualization(image_path)