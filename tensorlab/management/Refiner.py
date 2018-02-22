from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''TensorFlow implementation of '''
import os
# from pydoc import locate
# import csv
# import pickle
import threading
import time

# from collections import deque

# import math
from datetime import datetime

import numpy as np
import tensorflow as tf

from lib.management.Handler import Handler
# from lib.data.queue import DataQueue

import lib.io.schema as schema_io
from lib.schemas.Schema import Schema

import lib.management.preparation as prepare


class Refiner(Handler):

  def __init__(
          self,
          architecture,
          regime,
          data=None):
    Handler.__init__(self, regime.output)

    # self.args = ARGS
    self.regime = regime
    # self.architecture = architecture

    schema_io._save(
        regime,
        regime.output.dirs.logs + '/schemas/regime.json')
    schema_io._save(
        architecture,
        regime.output.dirs.logs + '/schemas/architecture.json')

    self.data = data
    # self.rename_attribute("counter", "current_epoch")
    # self.current_epoch = self.counter
    # this is very hacky
    # due to not being able to acces the num_epochs directly
    # setting hard samples/epoch limit
    # limit = num_examples * augmentation_multiplier
    training_size = int(
        data.train.num_examples *
        regime.augmentation.training.multiplier)

    # set hard limit on batches or use the size of the dataset
    if regime.augmentation.training.num_batches is None:
      self.num_training_batches = int(
          training_size /
          regime.queue.training.batch_size)
    else:
      self.num_training_batches = regime.augmentation.training.num_batches

    validation_size = int(
        data.validation.num_examples *
        regime.augmentation.validation.multiplier)

    if regime.augmentation.validation.num_batches is None:
      self.num_validation_batches = int(
          validation_size /
          regime.queue.validation.batch_size)
    else:
      self.num_validation_batches = regime.augmentation.validation.num_batches

    self.best_stats = None

    # build the graph in the sess
    with self.sess.as_default():

      self.coord = tf.train.Coordinator()

      with tf.device('/cpu:0'):

        self.placeholders = prepare._placeholders(
            self, architecture)

        self.readers = prepare._readers(self, architecture)

      self.graph = prepare._graph(self, architecture)

      self.optimizations = prepare._optimizations(self)

      # self.saver, self.merged = prepare.savers(self)

      # reinitialize the saver for all the new variables
      self.saver = tf.train.Saver(tf.global_variables())

      if self.regime.evaluation.write_summaries:
        self.merged, self.train_writer, self.val_writer = \
            prepare._writers(self, regime)

      tf.global_variables_initializer().run()

    self.restore_session()

  def train(self):
    with self.sess as sess:
      """
        Initial Validation
      """

      print(
          '\n--- Refinement commenced at {0}\n'.format(self.time())
      )

      self.__validate__(sess)

      for epoch in range(self.regime.evaluation.max_epochs):
        """
          Perform Training
        """

        # print("before training")

        if self.coord.should_stop():
          self.coord.clear_stop()

        self.__train__(sess)

        """
          Perform Validation
        """

        # print("after training")

        if self.coord.should_stop():
          self.coord.clear_stop()

        # print("before validation")
        self.__validate__(sess)
        # print("after validation")

    return self.graph

  def __train__(self, sess):

    avg_loss = []
    avg_size = []

    i = self.optimizations.step
    start = self.optimizations.step

    # try:
    threads = self.readers.training.start_threads(sess)

    # print("before while")

    while i - start < self.num_training_batches:

      # print("begin while")

      loss, i, _ = sess.run(
          [
              self.graph.handles.metrics.loss,
              self.optimizations.handles.global_step,
              self.optimizations.handles.optimizer
          ],
          {
              self.placeholders.is_training: True,
              self.placeholders.regularizer_scale:
              self.regime.optimizer.regularizer_scale
          }
      )

      # print("after train op")

      if self.regime.evaluation.queue.training.summaries\
              and i % self.regime.evaluation.queue.training.freq == 0:
        avg_size = np.sum(avg_size) / len(avg_size)
        # self.regime.evaluation.queue.training.freq
        self._print_queue_summary(
            avg_size,
            self.readers.training)

        avg_size = []
      else:
        avg_size.append(sess.run(self.readers.training.size()))

      if i % self.regime.evaluation.eval_freq == 0:
        print(
            "average loss {0:>20f} at {1:>6}, epoch {2:>6}\n".format(
                np.sum(avg_loss)/len(avg_loss),
                i,
                self.step))

        avg_loss = []
      else:
        avg_loss.append(loss)

      # print("mid summaries")

      if np.isnan(loss):
        print(
            "training loss is nan at {0:>6}, epoch {1:>6}".format(
                i,
                self.step))

      # 30 - \
      if i - start == self.num_training_batches - \
              (self.regime.queue.training.n_threads + 1):
        self.coord.request_stop()

      # print("end while")

    # print("before threads join")

    self.coord.join(
        threads,
        stop_grace_period_secs=self.regime.queue.training.grace_period)

    self.readers.training.threads = []

    # print("before increment")

    # counter is a variable for the current epoch
    self.increment_counter()

  def __validate__(self, sess):

    metrics = [
        self.graph.handles.metrics.loss,
        self.graph.handles.metrics.precision,
        self.graph.handles.metrics.recall,
        self.graph.handles.metrics.specificity
    ]

    if self.regime.evaluation.write_summaries:
      metrics += [self.merged]

    ###### collect summaries training ######
    train_measures = [[] for x in range(5)]
    for i in range(self.regime.evaluation.comparison_samples):
      cuts, complete = self.data.train.random_sample(
          self.regime.queue.validation.batch_size)

      values = sess.run(
          metrics,
          {
              self.placeholders.source_tensor: cuts,
              self.placeholders.target_tensor: complete,
              self.placeholders.regularizer_scale:
              self.regime.optimizer.regularizer_scale
          }
      )

      # organise into lists
      for value, measure in zip(values, train_measures):
        measure.append(value)

    # summaries contained in the final element
    if self.regime.evaluation.write_summaries:
      for train_summary in train_measures[4]:
        self.train_writer.add_summary(
            train_summary,
            self.step)

    ###### collect summaries validation ######
    threads = self.readers.validation.start_threads(sess)

    # collect summaries for whole validation set
    val_measures = [[] for x in range(5)]
    for i in range(self.num_validation_batches):  # 20):  #

      if i != 0 and self.regime.evaluation.queue.validation.summaries\
              and i % self.regime.evaluation.queue.validation.freq == 0:
        self._print_queue_summary(
            sess.run(self.readers.validation.size()),
            self.readers.validation)

      values = sess.run(
          metrics,
          {
              self.placeholders.is_validating: True,
              self.placeholders.regularizer_scale:
              self.regime.optimizer.regularizer_scale
          }
      )

      # organise into lists
      for value, measure in zip(values, val_measures):
        measure.append(value)

      if i == self.num_validation_batches - \
              (self.regime.queue.validation.n_threads + 1):
        self.coord.request_stop()

    if self.regime.evaluation.write_summaries:
      for val_summary in val_measures[4]:
        self.val_writer.add_summary(
            val_summary,
            self.step)

    for i, (train, val) in \
            enumerate(zip(train_measures[:4], val_measures[:4])):
      train_measures[i] = np.mean(train)
      val_measures[i] = np.mean(val)

    self._print_validation(
        # global_step.eval(session=self.model.sess),
        self.optimizations.step,
        self.step,
        train_measures,
        val_measures
    )

    if self.best_stats is None or \
            val_measures[1] + val_measures[3] > \
            self.best_stats[1] + self.best_stats[3]:

      self.best_stats = val_measures
      self.save_session()

      print("model saved at epoch {0:>6}, {1:>22}\n".format(
          self.step,
          self.time()))

    self.coord.join(
        threads,
        stop_grace_period_secs=self.regime.queue.validation.grace_period)

    self.readers.validation.threads = []

  def _print_queue_summary(self, queue_size, reader):
    print(
        '{0:>20} size {1:.4f}, {2:>3} threads at {3}'.format(
            reader.name(),
            queue_size,  # sess.run(reader.size()),
            reader.num_threads(),
            self.time())
    )

  def _print_validation(self, step, epoch, train, val):
    print(
        "iter {0:>10}, epoch {1:>4} at {2:>20}\n".format(
            step,
            epoch,
            self.time()
        ))
    print(
        "loss at {0:>6} are {1:>20f} \
        (train) and {2:>20f} (val)".format(
            epoch,
            train[0],
            val[0]))
    print(
        "prec at {0:>6} are {1:>20f} \
        (train) and {2:>20f} (val)".format(
            epoch,
            train[1],
            val[1]))
    print(
        "reca at {0:>6} are {1:>20f} \
        (train) and {2:>20f} (val)".format(
            epoch,
            train[2],
            val[2]))
    print(
        "spec at {0:>6} are {1:>20f} \
        (train) and {2:>20f} (val)".format(
            epoch,
            train[3],
            val[3]),
        "\n")

  def __exit__(self, type, value, tb):

    self.sess.close()

    if self.regime.evaluation.write_summaries:
      self.train_writer.close()
      self.val_writer.close()
