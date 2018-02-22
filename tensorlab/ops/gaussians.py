from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np
# from lib.graphs.constructors.layers import \
#     Conv3dDNNLayer, Full3dFlattenLayer, DeConv3dDNNLayer


# def gaussian_noise(input_tensor, is_training, mu=0, std=0.5):
#   with tf.name_scope('gaussian_noise') as scope:

#     noise = tf.random_normal(shape=tf.shape(
#         input_tensor), mean=mu, stddev=std, dtype=tf.float32)

#     prepped_input = tf.cond(
#         is_training,
#         lambda: input_tensor - noise,
#         lambda: input_tensor)

#   return prepped_input

def gaussian_noise(input_layer, mean=1e-3, std=1e-5, offset=1e-5):
  with tf.name_scope('gaussian-noise') as scope:
    noise = tf.random_normal(shape=tf.shape(
        input_layer), mean=mean, stddev=std, dtype=tf.float32)

  return tf.nn.relu(input_layer + noise) + offset


def sample_gaussian(
        mu,
        log_sigma,
        is_training):
  with tf.variable_scope("sampling") as scope:
    # reparameterization trick
    epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
    # sample = tf.cond(
    #     is_training,
    #     lambda: mu + epsilon * tf.exp(log_sigma),
    #     lambda: mu)

    sample = mu + epsilon * tf.exp(log_sigma)

  return sample
