from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers

import numpy as np
# from lib.graphs.utils.summaries import scalar_summary


def get_learning_rate(
        learning_rate,
        global_step,
        decay_steps,
        decay_rate,
        decay_type='Exponential',
        name='learning_rate',
        **kwargs):

  def __piecewise_init(
          global_step,
          boundaries,
          learning_rates,
          name):
    assert not isinstance(learning_rates, float)
    assert not isinstance(boundaries, float)
    assert len(learning_rates) - 1 == len(boundaries)
    return tf.train.piecewise_constant(
        global_step,
        boundaries,
        learning_rates,
        name)

  decay_types = {
      'Polynomial': lambda
      learning_rate,
      global_step,
      decay_steps=2.5e5,
      end_learning_rate=0.0000001,
      power=1.0,
      cycle=False,
      name='polynomial_decay',
      **kwargs:
      tf.train.polynomial_decay(
          learning_rate,
          global_step,
          decay_steps,
          end_learning_rate,
          power,
          cycle,
          name),
      'Exponential': lambda
      learning_rate,
      global_step,
      decay_steps=4e4,
      decay_rate=0.5,
      staircase=True,
      name='exponential_decay':  # ,
      # **kwargs:
      tf.train.exponential_decay(
          learning_rate,
          global_step,
          decay_steps,
          decay_rate,
          staircase,
          name),
      'Inverse_time': lambda
      learning_rate,
      global_step,
      decay_steps=1e5,
      decay_rate=0.95,
      staircase=False,
      name='inverse_time_decay',
      **kwargs:
      tf.train.inverse_time_decay(
          learning_rate,
          global_step,
          decay_steps,
          decay_rate,
          staircase,
          name),
      'Natural_exp': lambda
      learning_rate,
      global_step,
      decay_steps=1e5,
      decay_rate=0.95,
      staircase=False,
      name='natural_exp_decay',
      **kwargs:
      tf.train.natural_exp_decay(
          learning_rate,
          global_step,
          decay_steps,
          decay_rate,
          staircase,
          name),
      'Piecewise': lambda
      learning_rate,
      global_step,
      boundaries=[int(1e4), int(2e4), int(3e4)],
      name='piecewise_constant',
      **kwargs:
      __piecewise_init(
          global_step,
          boundaries, [learning_rate, 1e-5, 1e-6, 1e-7],
          name),
      'Constant': lambda
      learning_rate,
      name='constant',
      **kwargs:
      learning_rate
      # tf.Variable(learning_rate, trainable=False, name=name)
  }

  with tf.name_scope("learning_rate"):
    learning_rate_with_decay = decay_types[
        decay_type](
        learning_rate,
        global_step,
        decay_steps=decay_steps,
        decay_rate=decay_rate)

    # learning_rate_with_decay = learning_rate
    tf.summary.scalar('decay', learning_rate_with_decay)

  return learning_rate_with_decay
