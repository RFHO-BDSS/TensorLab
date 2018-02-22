import tensorflow as tf
import numpy as np
# from tensorflow.contrib import layers

import numpy as np
# from lib.graphs.utils.operations import binary_activation
# from lib.graphs.utils.summaries import tf.summary.scalar
#  binary_activation,


def metrics(output_tensor, target_tensor, epsilon=1e-7):
  with tf.name_scope("measures") as _:

    size = tf.cast(tf.size(output_tensor), dtype=tf.float32)

    zeros = tf.zeros(tf.shape(output_tensor))
    ones = tf.ones(tf.shape(output_tensor))

    true_sum = tf.add(output_tensor, target_tensor)

    not_output = tf.cast(
        tf.logical_not(tf.cast(output_tensor, tf.bool)),
        tf.float32)
    not_sum = tf.add(not_output, target_tensor)

    with tf.name_scope("tp"):
      tp = tf.reduce_sum(tf.where(true_sum > 1, ones, zeros))
      tp = tp / size
      tf.summary.scalar('scalar', tp)

    with tf.name_scope("tn"):
      tn = tf.reduce_sum(tf.where(true_sum < 1, ones, zeros))
      tn = tn / size
      tf.summary.scalar('scalar', tn)

    with tf.name_scope("fp"):
      fp = tf.reduce_sum(tf.where(not_sum < 1, ones, zeros))
      fp = fp / size
      tf.summary.scalar('scalar', fp)

    with tf.name_scope("fn"):
      fn = tf.reduce_sum(tf.where(not_sum > 1, ones, zeros))
      fn = fn / size
      tf.summary.scalar('scalar', fn)

  with tf.name_scope("accuracies"):

    with tf.name_scope("precision") as _:
      precision = tp / (tp + fp + epsilon)
      tf.summary.scalar('scalar', precision)

    with tf.name_scope("recall") as _:
      recall = tp / (tp + fn + epsilon)
      tf.summary.scalar('scalar', recall)

    with tf.name_scope("specificity") as _:
      specificity = tn / (tn + fp + epsilon)
      tf.summary.scalar('scalar', specificity)

    with tf.name_scope("dice") as _:
      dice = 2 * tp / (2 * tp + fp + fn + epsilon)
      tf.summary.scalar('scalar', dice)

  return precision, recall, specificity, dice
