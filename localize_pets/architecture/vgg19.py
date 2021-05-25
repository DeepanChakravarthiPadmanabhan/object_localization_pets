import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from localize_pets.abstract.architecture import Architecture


class VGG19(Architecture):
    def __init__(self,
                 backbone,
                 feature_extraction,
                 image_width,
                 image_height):
        super(VGG19, self).__init__(backbone,
                                    feature_extraction,
                                    image_width,
                                    image_height
                                    )

    def model(self):
        input_ = Input(shape=(self.image_height,
                              self.image_width,
                              3),
                       name="image")
        x = input_
        base_model = tf.keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_shape=(self.image_height, self.image_width, 3),
        )
        if self.feature_extraction:
            base_model.trainable = False
            x = base_model(x, training=False)
        else:
            base_model.trainable = True
            x = base_model(x)

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
