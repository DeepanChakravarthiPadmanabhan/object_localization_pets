import argparse
import random
import tensorflow as tf

from localize_pets.dataset.pets_detection import Pets_Detection
from localize_pets.datagenerator.datagenerator import DataGenerator
from localize_pets.architecture.simple_model import simple_model
from localize_pets.utils.misc import lr_schedule, IOU, ShowTestImages

print('Code base using TF version: ', tf.__version__)
description = 'Training script for object localization task on pets dataset'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-dd', '--data_dir', default='/home/deepan/Downloads/', type=str,
                    help='Data directory for training')
parser.add_argument('--train_samples', default=2800, type=int,
                    help='Sample set size for training data')
parser.add_argument('--test_samples', default=800, type=int,
                    help='Sample set size for testing data')
parser.add_argument('-bs', '--batch_size', default=64, type=int,
                    help='Batch size for training')
args = parser.parse_args()
config = vars(args)
print('Run config: ', config)
trainval_data = Pets_Detection(config['data_dir'], 'trainval')
trainval_dataset = trainval_data.load_data()
print('Total dataset items read: ', len(trainval_dataset))
random.shuffle(trainval_dataset)
train_dataset = trainval_dataset[:config['train_samples']]
test_dataset = trainval_dataset[config['train_samples']: config['train_samples'] + config['test_samples']]
print('Samples in train and test dataset are %d and %d, respectively. ' % (len(train_dataset), len(test_dataset)))
train_datagen = DataGenerator(train_dataset, config['batch_size'], True)
test_datagen = DataGenerator(test_dataset, 1, True)
model = simple_model()
model.compile(
        loss={'class_out': 'categorical_crossentropy',
              'box_out': 'mse'},
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics={
            'class_out': 'accuracy',
            'box_out': IOU(name='iou')
        }
    )
model.fit(
        train_datagen,
        epochs=100,
        callbacks=[ShowTestImages(test_datagen),
                   tf.keras.callbacks.EarlyStopping(monitor='box_out_iou', patience=3, mode='max'),
                   tf.keras.callbacks.LearningRateScheduler(lr_schedule)]
    )
model.save('save_checkpoint/pets_model')

