import argparse
import random
import tensorflow as tf

from localize_pets.dataset.pets_detection import Pets_Detection
from localize_pets.datagenerator.datagenerator import DataGenerator
from localize_pets.architecture.factory import ArchitectureFactory
from localize_pets.utils.callbacks import lr_schedule, ShowTestImages
from localize_pets.loss_metric.iou import IOU

print("Code base using TF version: ", tf.__version__)
description = "Training script for object localization task on pets dataset"
parser = argparse.ArgumentParser(description=description)
parser.add_argument(
    "-dd",
    "--data_dir",
    default="/home/deepan/Downloads/",
    type=str,
    help="Data directory for training",
)
parser.add_argument(
    "--train_samples", default=2800, type=int, help="Sample set size for training data"
)
parser.add_argument(
    "--test_samples", default=800, type=int, help="Sample set size for testing data"
)
parser.add_argument(
    "-bs", "--batch_size", default=2, type=int, help="Batch size for training"
)
parser.add_argument(
    "-iw", "--image_width", default=224, type=int, help="Input image width"
)
parser.add_argument(
    "-ih", "--image_height", default=224, type=int, help="Input image height"
)
parser.add_argument(
    "--resize",
    default=True,
    type=bool,
    help="Whether to resize the image",
)
parser.add_argument(
    "--backbone",
    default="VGG19",
    type=str,
    help="Architecture name. Available options: SimpleNet, EfficientDet, VGG19",
)
parser.add_argument(
    "-fe",
    "--feature_extraction",
    default=False,
    type=bool,
    help="Train backbone or use as feature extractor",
)
parser.add_argument(
    "--normalize",
    default="max",
    type=str,
    help="Normalization strategy. Available options: max, same_scale. Max for SimpleNet, VGG19 and same_scale for EfficientNet",
)
args = parser.parse_args()
config = vars(args)
print("Run config: ", config)

# Transform
transforms = dict()
if config["resize"]:
    transforms["resize"] = [config["image_height"], config["image_width"]]
elif config["normalize"]:
    transforms["normalize"] = config["normalize"]

# Dataset generation
trainval_data = Pets_Detection(config["data_dir"], "trainval")
trainval_dataset = trainval_data.load_data()
print("Total dataset items read: ", len(trainval_dataset))
random.shuffle(trainval_dataset)
train_dataset = trainval_dataset[: config["train_samples"]]
test_dataset = trainval_dataset[
    config["train_samples"] : config["train_samples"] + config["test_samples"]
]
print(
    "Samples in train and test dataset are %d and %d, respectively. "
    % (len(train_dataset), len(test_dataset))
)
train_datagen = DataGenerator(
    train_dataset,
    config["batch_size"],
    config["image_width"],
    config["image_height"],
    transforms,
    True,
)
test_datagen = DataGenerator(
    test_dataset, 1, config["image_width"], config["image_height"], transforms, True
)

# Model build
architecturefactory = ArchitectureFactory(config['backbone'])
model_class = architecturefactory.factory()
model_class = model_class(
    backbone=config["backbone"],
    feature_extraction=config["feature_extraction"],
    image_width=config["image_width"],
    image_height=config["image_height"],
)
model = model_class.model()


# Model compile
model.compile(
    loss={"class_out": "categorical_crossentropy", "box_out": "mse"},
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics={"class_out": "accuracy", "box_out": IOU(name="iou")},
)

# Training
model.fit(
    train_datagen,
    epochs=100,
    callbacks=[
        ShowTestImages(test_datagen),
        tf.keras.callbacks.EarlyStopping(monitor="box_out_iou", patience=3, mode="max"),
        tf.keras.callbacks.LearningRateScheduler(lr_schedule),
    ],
)

# Save model
model.save("save_checkpoint/pets_model")
