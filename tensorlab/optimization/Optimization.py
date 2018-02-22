import os

from datetime import datetime

import numpy as np
import tensorflow as tf

# from lib.schemas.Schema import Schema

from lib.optimization.constructors.learning_rate import get_learning_rate
from lib.optimization.constructors.optimizer import get_optimizer

# from lib.optimization.generator import Generator


class Optimization(object):

  def __init__(self, loss):
    with tf.variable_scope('Optimization'):

      self.global_step = tf.Variable(
          0,
          trainable=False,
          name='global-step')

      self.learning_rate = get_learning_rate(
          learning_rate=1e-4,
          global_step=self.global_step,
          decay_steps=2e4,
          decay_rate=0.1,
          decay_type='Exponential')

      self.apply_gradients = get_optimizer(
          loss=loss,
          learning_rate=self.learning_rate,
          global_step=self.global_step,
          clip_max=5,
          op='Momentum')

  # @property
  def step(self, sess):
    """
      Train step
    """
    return sess.run(self.global_step)
