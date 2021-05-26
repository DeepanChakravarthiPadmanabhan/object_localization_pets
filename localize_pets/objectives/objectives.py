import tensorflow as tf


class Loss():
    def __init__(self):
        pass

    def mean(self, activation, filter_index):
        filter_activation = activation[:, :, :, filter_index]
        loss = tf.reduce_mean(filter_activation)
        return loss

    def l1(self, activation, filter_index, constant=0):
        filter_activation = activation[:, :, :, filter_index]
        loss = tf.reduce_sum(tf.abs(filter_activation - constant))
        return loss

    def l2(self, activation, filter_index, constant=0, epsilon=1e-05):
        filter_activation = activation[:, :, :, filter_index]
        loss = tf.sqrt(epsilon + tf.reduce_sum((filter_activation - constant) ** 2))
        return loss

    def totalvariation(self, activation, filter_index):
        filter_activation = activation[:, :, :, filter_index]
        loss = tf.image.total_variation(filter_activation)
        return loss

    def custom(self, activation, filter_index, constant=0, epsilon=1e-5):
        filter_activation = activation[:, :, :, filter_index]
        l1_loss = tf.reduce_sum(tf.abs(filter_activation - constant))
        l2_loss = tf.sqrt(epsilon + tf.reduce_sum((filter_activation - constant) ** 2))
        tv_loss = tf.image.total_variation(filter_activation)
        loss = l1_loss + l2_loss + tv_loss
        return loss

