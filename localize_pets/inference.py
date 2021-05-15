import argparse
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


def compute_loss(img, filter_index):
    # We avoid border artifacts by only involving non-border pixels in the loss.
    activation = feature_extractor(img)
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    # print('filter: ', activation.shape)
    # filter_activation = activation[:, :, :, filter_index]
    return tf.reduce_mean(filter_activation)


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


def initialize_image(width, height):
    img = tf.random.uniform((1, width, height, 3))
    return img


def deprocess_image(image):
    image = (image - tf.reduce_min(image)) / tf.reduce_max((image - tf.reduce_min(image)))
    # Convert to RGB array
    image *= 255.
    image = np.clip(image, 0, 255).astype('uint8')
    return image


def visualize_filters(filter_index=0):
    img = initialize_image(300, 300)
    iterations = 50
    learning_rate = 10
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate)
    img = deprocess_image(img[0].numpy())
    return loss, img


image_path = '/home/deepan/Downloads/images/Abyssinian_21.jpg'
model_path = 'save_checkpoint/pets_model/'
inference_model(image_path, model_path)
# vgg_feature_visualization(image_path)
model = tf.keras.models.load_model(model_path, custom_objects={'IOU': IOU(name='iou')})
layer = model.get_layer(name='max_pooling2d_4')
feature_extractor = tf.keras.models.Model(inputs=model.inputs, outputs=layer.output)
loss, img = visualize_filters(1)
plt.imshow(img)
plt.savefig('here.jpg')
