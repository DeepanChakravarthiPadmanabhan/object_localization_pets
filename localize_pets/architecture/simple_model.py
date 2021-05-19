import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Conv2D,
    MaxPool2D,
    BatchNormalization,
    Dropout,
)


def simple_model_v1():
    input_ = Input(shape=(300, 300, 3), name="image")
    x = input_
    for i in range(0, 5):
        n_filters = 2 ** (4 + i)
        x = Conv2D(n_filters, 3, activation="relu")(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(2)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    class_out = Dense(2, activation="softmax", name="class_out")(
        x
    )  # Classification out
    box_out = Dense(4, name="box_out")(x)  # Box out
    model = tf.keras.models.Model(input_, [class_out, box_out])
    model.summary()
    return model


def simple_model():
    input_ = Input(shape=(300, 300, 3), name="image")
    x = input_
    for i in range(0, 5):
        n_filters = 2 ** (4 + i)
        x = Conv2D(n_filters, 3, activation="relu")(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(2)(x)

    class_head = Conv2D(256, 3, activation="relu")(x)
    class_head = Flatten()(class_head)
    class_head = Dense(256, activation="relu")(class_head)
    class_out = Dense(2, activation="softmax", name="class_out")(
        class_head
    )  # Classification out

    box_head = Conv2D(256, 3, activation="relu")(x)
    box_head = Flatten()(box_head)
    box_head = Dense(256, activation="relu")(box_head)
    box_out = Dense(4, name="box_out")(box_head)  # Box out

    model = tf.keras.models.Model(input_, [class_out, box_out])
    model.summary()
    return model
