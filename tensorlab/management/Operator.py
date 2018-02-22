import os
import copy

import threading
import time

from datetime import datetime

import numpy as np
import tensorflow as tf

from lib.management.Handler import Handler

import lib.io.schema as schema_io

METRICS_COLLECTION = {
    "precision": [],
    "recall": [],
    "specificity": [],
    "dice": []
}

class Operator(Handler):

  def __init__(
          self,
          manual):
    super().__init__(manual['output'])

    self._manual = manual

  # @property
  # def initialize(self):

  #   self._sess.run(tf.global_variables_initializer())
  #   self._sess.run(tf.local_variables_initializer())

  
  # def loss(self, loss):
  #   self._loss = loss

  def metrics(self, metrics):
    self._metrics = metrics

  def reconstruction(self, reconstruction):
    self._reconstruction = reconstruction


  def _sources(self, arrays):

    # sources = []

    # print(arrays[0].shape)

    if len(arrays) > 1:
      sources = np.stack(arrays, axis=-1)

    else:
      sources = np.expand_dims(arrays[0], axis=-1)

    sources = np.expand_dims(sources, axis=0)

    return sources

  
  def array_sparsity(self, input_array, threshold=0.5):


    array_sparsity = np.mean(
        np.where(input_array > threshold, 1, 0))
    # array_sparsity = np.nonzero(input_array)[0].size / \
    #     input_array.size

    return array_sparsity


  def reconstruct(
          self,
          input_array):
    '''External functions for returning a reconstructed tensor

    Args:
        embedding: voxel data as tensor
    '''

    return self._sess.run(
        self._reconstruction,
        {
            self._placeholders[0]: input_array,
            self._placeholders[2]: False
        })



  # def encode(
  #         self,
  #         input_tensor):  # , hidden_size):
  #   '''External functions for returning an encoded tensor


  #   Args:
  #       input_tensor: voxel data as tensor
  #       hidden_size: the size of the embedding dimension
  #   '''
  #   return self.sess.run(
  #       [
  #           self.graph.handles.network.latent.mu,
  #           self.graph.handles.network.latent.log_sigma,
  #           self.graph.handles.network.latent.sample
  #       ],
  #       feed_dict={
  #           self.graph.handles.input.source: input_tensor  # ,
  #           # self.is_testing: True
  #       })

  # def decode(
  #         self,
  #         embedding):
  #   '''External functions for returning a reconstructed tensor

  #   Args:
  #       embedding: voxel data as tensor
  #   '''
  #   # raise NotImplementedError
  #   return self.sess.run(
  #       self.graph.handles.network.binary,
  #       {
  #           self.graph.handles.network.sample: embedding  # ,
  #           # self.is_training: False
  #       })