import os

import threading

import datetime
import time
from timeit import default_timer as timer


import tensorflow as tf
# import numpy as np

# from lib.management.Trainer import Trainer
from lib.management import Trainer

# from lib.data.queues.Queue import Queue
# from lib.data.queues.IteratorQueue import IteratorQueue
from lib.data.queues.DataQueue import DataQueue
import lib.data.loader as load_data


import lib.analysis.training.costs as training_costs
import lib.analysis.training.losses as training_losses
import lib.analysis.testing.metrics as testing_metrics

from lib.optimization.Optimization import Optimization

from lib.ops.gaussians import gaussian_noise
from lib.ops.activations import binarize_random


def _new(regime): # architecture, 
  # try:

  with Trainer.Trainer(regime) as trainer:
    # trainer, sess = environment
    with trainer.default as sess:

      output_shape = [32, 32, 32]
      # output_scale = [0.5, 0.5, 0.5]
      batch_size = 75

      with tf.device('/cpu:0'):

        # load the data readers and set the partition names
        datasets = load_data._readers(regime['input'])
        trainer.partitions(datasets.keys())

        # augmenter = CTAAugmenter(output_shape, output_scale)
        # trainer.augmenter(augmenter)

        with tf.name_scope("input"):
          # placeholders
          source = tf.placeholder(
              dtype=tf.float32,
              shape=output_shape + [1],  # [None] +
              name='source-placeholder')

          target = tf.placeholder(
              dtype=tf.float32,
              shape=output_shape + [1],  # [None] +
              name='target-placeholder')

          is_training = tf.placeholder(
              dtype=tf.bool,
              shape=[],
              name='is-training')
          trainer.placeholders(
              [source, target, is_training])

          # coordinator
          coordinator = tf.train.Coordinator()
          trainer.coord(coordinator)

          # queues
          # with tf.container('train-queue-container'):
          training_queue = DataQueue(
              coordinator, [source, target],
              datasets['train'],  # augmenter,
              capacity=3 * batch_size, min_after_dequeue=batch_size,
              dtypes=[tf.float32, tf.float32],
              shapes=[(output_shape + [1]),  # [1] +
                      (output_shape + [1])],  # [1] +
              name='train-queue')

          # queues
          # with tf.container('val-queue-container'):
          validation_queue = DataQueue(
              coordinator, [source, target],
              datasets['val'],  # augmenter,
              capacity=3 * batch_size, min_after_dequeue=batch_size,
              dtypes=[tf.float32, tf.float32],
              shapes=[(output_shape + [1]),  # [1] +
                      (output_shape + [1])],  # [1] +
              name='validation-queue')

          # testing_queue = DataQueue(
          #     coordinator, [source, target],
          #     datasets['test'], augmenter,
          #     capacity=100, dtypes=[tf.float32, tf.float32],
          #     shapes=[(output_shape + [1]),  # [1] +
          #             (output_shape + [2])],  # [1] +
          #     name='test-queue')

        inputs = tf.cond(
            is_training,
            lambda: training_queue.dequeue(batch_size),
            lambda: validation_queue.dequeue(batch_size)
        )

      #   # initialize inputs for graph
      # inputs = training_queue.dequeue(batch_size)

      # graph
      graph = ArtNet(inputs[0], is_training)
      trainer.graph(graph)

      # losses
      logits = graph.logits

      with tf.name_scope('losses'):

        # scale the targets
        targets_scaled = training_costs.scale_tensor(inputs[1])
        # targets_scaled = tf.nn.sigmoid(inputs[1])

        # scale the logits
        logits_scaled = training_costs.scale_tensor(logits)
        # logits_scaled = tf.nn.sigmoid(logits)

        # binarize
        target_binary = binarize_random(targets_scaled)
        # target_binary = tf.Print(
        #     target_binary,
        #     [tf.reduce_mean(target_binary, [1, 2, 3, 4])],
        #     "\nTarget binary sparsity: ")

        logits_binary = binarize_random(logits_scaled)
        # logits_binary = tf.Print(
        # logits_binary,
        # [tf.reduce_mean(logits_binary, [1, 2, 3, 4])],
        # "\nLogits binary sparsity: ")

        with tf.name_scope("sparsity-gamma"):
          gamma = 1 - tf.reduce_mean(target_binary, [1, 2, 3, 4])

        sato_reconstruction = training_costs.reconstruction(
            logits_scaled,
            targets_scaled,
            gamma=gamma)
        loss = sato_reconstruction

        trainer.loss(loss)

      with tf.name_scope("analysis"):
        metrics = testing_metrics.metrics(
            logits_binary,  # binarize_random(logits_scaled),
            target_binary)  # binarize_random(targets_scaled))
        trainer.metrics(metrics)

      # optimizer
      optimizer = Optimization(loss)
      trainer.optimizer(optimizer)

      with tf.name_scope("merged-summaries"):
        merged = tf.summary.merge_all()
        trainer.summaries(merged)

      # initialize it all
      trainer.initialize

      for epoch in range(12):

        print("------> Commencing Epoch {0:>3}\n".format(epoch))

        """
            Perform Training
        """

        # if sess.run(training_queue.is_closed):
        #     sess.reset(
        #         target="input",
        #         containers=['train-queue-container'])

        if coordinator.should_stop():
          coordinator.clear_stop()

        trainer.queue(training_queue)
        trainer.train(sess, is_training)


        """




        """


        # print("threads after training: \n", threading.enumerate())
        # for thread in threading.enumerate():
        #   print(thread.name)



        """
            Perform Validation
        """

        # if sess.run(validation_queue.is_closed):
        #     sess.reset(
        #         target="input",
        #         containers=['val-queue-container'])

        if coordinator.should_stop():
          coordinator.clear_stop()

        trainer.queue(validation_queue)
        trainer.validate(sess, is_training)

        """




        """

        # print("threads after validation: \n", threading.enumerate())

        # trainer.increment_counter()

      trainer.save_session()

  return None

  """




    """
