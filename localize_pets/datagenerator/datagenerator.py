import numpy as np
import cv2
import math
import tensorflow as tf
from localize_pets.transforms.transforms import process_bbox_image


class DataGenerator(tf.compat.v2.keras.utils.Sequence):
    def __init__(
        self, dataset, batch_size, image_width, image_height, transforms, shuffle=True
    ):
        self.batch_size = batch_size
        self.dataset = dataset
        self.image_width = image_width
        self.image_height = image_height
        self.transforms = transforms
        self.shuffle = shuffle
        self.n = 0
        self.list_IDs = np.arange((len(self.dataset)))
        print("Total examples in the dataset: ", len(self.list_IDs))
        self.on_epoch_end()

    def __len__(self):
        return int(math.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, Y = self._generate_X(list_IDs_temp)
        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        x_batch = np.zeros(
            (
                self.batch_size,
                self.image_height,
                self.image_width,
                3,
            )
        )
        y_batch = np.zeros((self.batch_size, 2))
        bbox_batch = np.zeros((self.batch_size, 4))
        for i, ID in enumerate(list_IDs_temp):
            image = cv2.imread(self.dataset[ID]["image_path"])
            class_id = self.dataset[ID]["species"]
            bbox = self.dataset[ID]["bbox"]
            image, bbox = process_bbox_image(image, bbox, self.transforms)
            x_batch[i] = image
            bbox_batch[i] = np.array(bbox)
            y_batch[i, class_id] = 1.0
        return {"image": x_batch}, {"class_out": y_batch, "box_out": bbox_batch}
