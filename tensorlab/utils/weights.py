from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers

import numpy as np


def weight_variable(
        name,
        shape,
        initializer=tf.contrib.layers.xavier_initializer(),
        regularizer=None):  # tf.contrib.layers.l2_regularizer(0.1)):
  return tf.get_variable(
      name,
      shape,
      tf.float32,
      initializer=initializer,
      regularizer=regularizer)


def bias_variable(
        name,
        shape,
        initializer=tf.contrib.layers.xavier_initializer()):
  return tf.get_variable(
      # return tf.Variable(
      name,
      shape,
      tf.float32,
      initializer=initializer)


# def binary_activation(x):
#   with tf.variable_scope("binary_activation") as scope:
#     # tf.constant(0.5, shape=x.get_shape()))  # tf.shape(x)))
#     cond = tf.less(x, 0.5)
#     out = tf.select(
#         cond,
#         tf.zeros(tf.shape(x)),
#         tf.ones(tf.shape(x)),
#         name='binarize')

#   return out


# def scale_tensor(input_tensor, min_value, max_value, name="input_tensor"):
#   with tf.variable_scope(name + "_scaling") as scope:
#     current_min = tf.reduce_min(input_tensor)
#     current_max = tf.reduce_max(input_tensor)

#     # scale to [target_min; target_max]
#     scaled_tensor = input_tensor * (max_value - min_value) + min_value
#   return scaled_tensor


# def sample_gaussian(mu, log_sigma):
#   with tf.variable_scope("sampling") as scope:
#     # reparameterization trick
#     epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
#     return mu + epsilon * tf.exp(log_sigma)
