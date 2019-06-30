
import cv2

import numpy as np
import tensorflow as tf

from Define import *

# x = [1, 2, 3, 4, 5, ...] (784)
# cond = [0, 0, 0, 1, 0, 0] (10)
# concat (x, cond) -> [1, 2, 3, 4, 5, ...., 0, 0, 0, 1, 0, 0]
def Generator(x, reuse = False, name = 'Generator'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        init = tf.contrib.layers.xavier_initializer()

        x = tf.layers.dense(x, 256, kernel_initializer = init)
        x = tf.nn.leaky_relu(x, 0.2)

        x = tf.layers.dense(x, 128, kernel_initializer = init)
        x = tf.nn.leaky_relu(x, 0.2)

        x = tf.layers.dense(x, IMAGE_WIDTH * IMAGE_HEIGHT, kernel_initializer = init)
        x = tf.nn.tanh(x) # -1 ~ 1
        
        return x

def Discriminator(x, reuse = False, name = 'Discriminator'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        init = tf.contrib.layers.xavier_initializer()

        x = tf.layers.dense(x, 256, kernel_initializer = init)
        x = tf.nn.leaky_relu(x, 0.2)

        x = tf.layers.dense(x, 128, kernel_initializer = init)
        x = tf.nn.leaky_relu(x, 0.2)

        logits = tf.layers.dense(x, 1, kernel_initializer = init)
        predictions = tf.nn.sigmoid(logits)

        return logits, predictions
