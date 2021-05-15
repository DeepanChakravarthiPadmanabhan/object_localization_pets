import numpy as np
import cv2
import math
import tensorflow as tf

def process_bbox_image(image, bbox, resize):
    width, height = image.shape[1], image.shape[0]
    new_image = cv2.resize(image, (resize, resize), interpolation=cv2.INTER_NEAREST)
    new_image = new_image / 255.
    width_factor = resize / width
    height_factor = resize / height
    xmin, ymin, xmax, ymax = bbox
    xmin = xmin * width_factor
    xmax = xmax * width_factor
    ymin = ymin * height_factor
    ymax = ymax * height_factor
    new_bbox = [xmin, ymin, xmax, ymax]
    return new_image, new_bbox

class DataGenerator(tf.compat.v2.keras.utils.Sequence):

    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.dataset = dataset
        self.shuffle = shuffle
        self.n = 0
        self.list_IDs = np.arange((len(self.dataset)))
        print('Total examples in the dataset: ', len(self.list_IDs))
        self.on_epoch_end()

    def __len__(self):
        return int(math.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, Y = self._generate_X(list_IDs_temp)
        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        x_batch = np.zeros((self.batch_size, 300, 300, 3))
        y_batch = np.zeros((self.batch_size, 2))
        bbox_batch = np.zeros((self.batch_size, 4))
        for i, ID in enumerate(list_IDs_temp):
            image = cv2.imread(self.dataset[ID]['image_path'])
            class_id = self.dataset[ID]['species']
            bbox = self.dataset[ID]['bbox']
            image, bbox = process_bbox_image(image, bbox, 300)
            x_batch[i] = image
            bbox_batch[i] = np.array(bbox)
            y_batch[i, class_id] = 1.0
        return {'image': x_batch}, {'class_out': y_batch, 'box_out': bbox_batch}