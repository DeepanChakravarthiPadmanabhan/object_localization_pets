import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense

def dnn(feature_extractor=False,
        image_width=224,
        image_height=224):
    input_ = Input(shape=(image_height, image_width, 3), name='image')
    x = input_
    base_model = tf.keras.applications.EfficientNetB0(include_top=False,
                                                      weights='imagenet',
                                                      input_shape=(image_height, image_width, 3))
    if feature_extractor:
        base_model.trainable = False
        x = base_model(x, training=False)
    else:
        base_model.trainable = True
        x = base_model(x, training=False)

    class_head = Conv2D(256, 3, activation='relu')(x)
    class_head = Flatten()(class_head)
    class_head = Dense(256, activation='relu')(class_head)
    class_out = Dense(2, activation='softmax', name='class_out')(class_head) # Classification out

    box_head = Conv2D(256, 3, activation='relu')(x)
    box_head = Flatten()(box_head)
    box_head = Dense(256, activation='relu')(box_head)
    box_out = Dense(4, name='box_out')(box_head) # Box out

    model = tf.keras.models.Model(input_, [class_out, box_out])
    model.summary()
    return model
