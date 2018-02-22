import tensorflow as tf
import numpy as np


# parametric leaky relu
def prelu(x):
    alpha = tf.get_variable(
        'alpha',
        shape=x.get_shape()[-1],
        dtype=x.dtype,
        initializer=tf.constant_initializer(0.1))
    return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)


def binarize_fixed(x, threshold):
    with tf.variable_scope("binary-fixed") as scope:

        cond = tf.less(x, threshold)

        out = tf.where(
            cond,
            tf.zeros(tf.shape(x)),
            tf.ones(tf.shape(x)),
            name='thresholding')

    return out


def binarize_random(x, thresh_min=0.495, thresh_max=0.505):
    with tf.variable_scope("binary-random") as scope:

        # for now do random threshold choice between likely values
        voxel_threshold = tf.constant(
            np.random.uniform(thresh_min, thresh_max))

        # voxel_threshold = np.random.uniform(thresh_min, thresh_max)

        cond = tf.less(x, voxel_threshold)

        out = tf.where(
            cond,
            tf.zeros(tf.shape(x)),
            tf.ones(tf.shape(x)),
            name='thresholding')

    return out
