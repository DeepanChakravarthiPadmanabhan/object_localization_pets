import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout

def simple_model():
    input_ = Input(shape=(300, 300, 3), name='image')
    x = input_
    for i in range(0, 5):
        n_filters = 2**(4+i)
        x = Conv2D(n_filters, 3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(2)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    class_out = Dense(2, activation='softmax', name='class_out')(x) # Classification out
    box_out = Dense(4, name='box_out')(x) # Box out
    model = tf.keras.models.Model(input_, [class_out, box_out])
    model.summary()
    return model
