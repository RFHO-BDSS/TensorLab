import os

import threading

import datetime
import time
from timeit import default_timer as timer

import SimpleITK as sitk

import tensorflow as tf
import numpy as np
from itertools import product

# from lib.management.Trainer import Trainer
from lib.management import Operator

# from lib.data.queues.Queue import Queue
# from lib.data.queues.IteratorQueue import IteratorQueue
# from lib.data.queues.DataQueue import DataQueue
# from lib.data.augmenters.CTAAugmenter import CTAAugmenter
import lib.data.loader as load_data

from lib.architecture.graphs.ArtNet import ArtNet

import lib.analysis.training.costs as training_costs
import lib.analysis.training.losses as training_losses
import lib.analysis.testing.metrics as testing_metrics

# from lib.optimization.Optimization import Optimization

from lib.ops.gaussians import gaussian_noise
from lib.ops.activations import binarize_random

import lib.io.folders as folders_io

def _single(manual):

  with Operator.Operator(manual) as operator:
    # trainer, sess = environment
    with operator.default as sess:

      output_shape = [32, 32, 32]
      # output_scale = [0.5, 0.5, 0.5]
      # batch_size = 75

      with tf.device('/cpu:0'):

        # load the data readers and set the partition names
        datasets = load_data._readers(manual['input'])
        operator.partitions(datasets.keys())

        reader = iter(datasets['test'])

        with tf.name_scope("input"):
          # placeholders
          source = tf.placeholder(
              dtype=tf.float32,
              shape=[1] + output_shape + [1],  # [None] +
              name='source-placeholder')

          target = tf.placeholder(
              dtype=tf.float32,
              shape=[1] + output_shape + [1],  # [None] +
              name='target-placeholder')

          is_training = tf.placeholder(
              dtype=tf.bool,
              shape=[],
              name='is-training')
          operator.placeholders(
              [source, target, is_training])

      # graph
      graph = ArtNet(source, is_training)
      operator.graph(graph)

      # losses
      logits = graph.logits

      with tf.name_scope('losses'):

        # scale the targets
        targets_scaled = training_costs.scale_tensor(target)
        # targets_scaled = tf.nn.sigmoid(inputs[1])

        # scale the logits
        logits_scaled = training_costs.scale_tensor(logits)
        # logits_scaled = tf.nn.sigmoid(logits)
        operator.reconstruction(logits_scaled)

        # binarize
        target_binary = binarize_random(targets_scaled)
        # target_binary = tf.Print(
        #     target_binary,
        #     [tf.reduce_mean(target_binary, [1, 2, 3, 4])],
        #     "\nTarget binary sparsity: ")

        logits_binary = binarize_random(logits_scaled)
        # logits_binary = tf.Print(
        #     logits_binary,
        #     [tf.reduce_mean(logits_binary, [1, 2, 3, 4])],
        #     "\nLogits binary sparsity: ")


      with tf.name_scope("analysis"):
        metrics = testing_metrics.metrics(
            logits_binary,  # binarize_random(logits_scaled),
            target_binary)  # binarize_random(targets_scaled))
        operator.metrics(metrics)

      # optimizer
      # optimizer = Optimization(loss)
      # trainer.optimizer(optimizer)

      with tf.name_scope("merged-summaries"):
        merged = tf.summary.merge_all()
        operator.summaries(merged)

      # initialize it all
      operator.initialize

      checkpoint_dir = '/home/none-exist/Documents/ArterySegmentation/logs/trained/20180219-013535'
      # print(checkpoint_dir)

      checkpoint_path = os.path.join(checkpoint_dir, 'checkpoints/best.ckpt')

      # print(checkpoint_path)

      operator.restore_session(checkpoint_path)


      source_arrays, info = next(reader)

      # print(info[1])
      path, image_id = os.path.split(info[1])
      reconstruction_dir = os.path.join(checkpoint_dir, 'output',image_id )
      # print(reconstruction_dir)

      folders_io._make_dirs(reconstruction_dir)

      # raise StopIteration

      sato_reconstruction = np.zeros(source_arrays[0].shape)

      grid_points = []

      for dim in source_arrays[0].shape:
        grid_points.append(np.arange(0, dim, 32))

      grid = list(product(*grid_points))

      for point in grid:
        slices = []
        for p in point:
          slices.append(slice(p, p + 32))

        samples = []
        for array in source_arrays:
          samples.append(
              array[slices[0], slices[1], slices[2]])

        # print(samples)
        sparsity = operator.array_sparsity(samples[0])


        # if not sparsity > 0.15:
        #   continue

        if any([source_arrays[0].shape[i] - point[i]
                        < 32 for i in range(3)]):
          continue

        # print("  sample sparsity: ", np.mean(
        #     np.where(samples[0] > 0.5, 1, 0)))
        # print(sparsity)


        # if not self._coord.should_stop():
        source_array = operator._sources([samples[0]])
        target_array = operator._sources([samples[1]])

        # self._targets(samples[1:])

        # print("\n   mu sparsity: ",
        #       self.array_sparsity(target[:, :, :, 0]))
        # print("sigma sparsity: ",
        #       self.array_sparsity(target[:, :, :, 1]))

        # print("source min: {0:>5.3f}, max: {1:>5.3f}".format(
        #     np.min(source), np.max(source)))
        # print("mean min: {0:>5.5f}, max: {1:>5.5f}".format(
        #     np.min(target[:, :, :, 0]), np.max(target[:, :, :, 0])))
        # print("var min: {0:>5.3f}, max: {1:>5.3f}".format(
        #     np.min(target[:, :, :, 1]), np.max(target[:, :, :, 1])))


        scaled_target = sess.run(targets_scaled, {
          target: target_array, is_training: False
        })
        reconstruction = operator.reconstruct(source_array)

        # print(np.min(scaled_target), np.max(scaled_target))
        # print(np.min(reconstruction), np.max(reconstruction), "\n")

        # print("target: {0:>.4f} and reconstruction: {1:>.4f} with diff {2:.4f}\n".format(
        #   operator.array_sparsity(scaled_target), 
        #   operator.array_sparsity(reconstruction),
        #   np.abs(operator.array_sparsity(scaled_target) / operator.array_sparsity(reconstruction))
        #   ))

        sato_reconstruction[slices[0], slices[1], slices[2]] = np.squeeze(reconstruction, axis=-1)

      # print(operator.array_sparsity(sato_reconstruction))
      # print(operator.array_sparsity(source_arrays[1]))
    
    
      
      
      original_scan = sitk.ReadImage(info[2])
      reconstructed_arteries = sitk.GetImageFromArray(sato_reconstruction)

      reconstructed_arteries.CopyInformation(original_scan)

      reconstruction_path = os.path.join(reconstruction_dir, 'reconstruction.mha')

      # print(reconstruction_path)

      sitk.WriteImage(reconstructed_arteries, reconstruction_path)
        
      # print(os.path.join(reconstruction_dir, 'reconstruction.mha'))
    
    return None



      

  return None

  """




  """



        # augmenter = CTAAugmenter(output_shape, output_scale)
        # trainer.augmenter(augmenter)

          # coordinator
          # coordinator = tf.train.Coordinator()
          # trainer.coord(coordinator)

          # # queues
          # # with tf.container('train-queue-container'):
          # training_queue = DataQueue(
          #     coordinator, [source, target],
          #     datasets['train'],  # augmenter,
          #     capacity=3 * batch_size, min_after_dequeue=batch_size,
          #     dtypes=[tf.float32, tf.float32],
          #     shapes=[(output_shape + [1]),  # [1] +
          #             (output_shape + [1])],  # [1] +
          #     name='train-queue')

          # # queues
          # # with tf.container('val-queue-container'):
          # validation_queue = DataQueue(
          #     coordinator, [source, target],
          #     datasets['val'],  # augmenter,
          #     capacity=3 * batch_size, min_after_dequeue=batch_size,
          #     dtypes=[tf.float32, tf.float32],
          #     shapes=[(output_shape + [1]),  # [1] +
          #             (output_shape + [1])],  # [1] +
          #     name='validation-queue')

          # testing_queue = DataQueue(
          #     coordinator, [source, target],
          #     datasets['test'], augmenter,
          #     capacity=100, 
          #     dtypes=[tf.float32, tf.float32],
          #     shapes=[(output_shape + [1]),  # [1] +
          #             (output_shape + [1])],  # [1] +
          #     name='test-queue')

        # inputs = tf.cond(
        #     is_training,
        #     lambda: training_queue.dequeue(batch_size),
        #     lambda: validation_queue.dequeue(batch_size)
        # )

      #   # initialize inputs for graph
      # inputs = training_queue.dequeue(batch_size)


        # with tf.name_scope("sparsity-gamma"):
        #   gamma = 1 - tf.reduce_mean(target_binary, [1, 2, 3, 4])

        # sato_reconstruction = training_costs.reconstruction_specific(
        #     logits_scaled,
        #     targets_scaled,
        #     gamma=gamma)
        # loss = sato_reconstruction

        # operator.loss(loss)

        # p_mu, p_sigma = tf.split(logits, 2, axis=-1)
        # p_mu = training_costs.scale_tensor(p_mu)
        # p_sigma = training_costs.scale_tensor(p_sigma)

        # q_mu, q_sigma = tf.split(inputs[1], 2, axis=-1)
        # q_mu = training_costs.scale_tensor(q_mu)
        # q_sigma = training_costs.scale_tensor(q_sigma)

        # output_logits = tf.nn.relu(logits)
        # output_pdfs = gaussian_noise(logits)
        # output_sigmoids = tf.nn.sigmoid(logits)

        # target_logits = tf.nn.relu(inputs[1])
        # target_pdfs = gaussian_noise(inputs[1])
        # target_sigmoids = tf.nn.sigmoid(inputs[1])

        # divergence = training_costs.KLDivergenceGauss(
        #     [p_mu, p_sigma], [q_mu, q_sigma])
        # mu_reconstruction = training_costs.reconstruction(
        #     p_mu, q_mu, name='mu')
        # sigma_reconstruction = training_costs.reconstruction(
        #     p_sigma, q_sigma, name='sigma')

        # mu_reconstruction + sigma_reconstruction  # divergence +

      # trainer.queue(validation_queue)
      # trainer.validate(sess, is_training)

      # for epoch in range(12):

      #   print("-----> Commencing Epoch {0:>3}\n".format(epoch))

      #   """
      #       Perform Training
      #   """

      #   # if sess.run(training_queue.is_closed):
      #   #     sess.reset(
      #   #         target="input",
      #   #         containers=['train-queue-container'])

      #   if coordinator.should_stop():
      #     coordinator.clear_stop()

      #   trainer.queue(training_queue)
      #   trainer.train(sess, is_training)


      #   """




      #   """


      #   # print("threads after training: \n", threading.enumerate())
      #   # for thread in threading.enumerate():
      #   #   print(thread.name)



      #   """
      #       Perform Validation
      #   """

      #   # if sess.run(validation_queue.is_closed):
      #   #     sess.reset(
      #   #         target="input",
      #   #         containers=['val-queue-container'])

      #   if coordinator.should_stop():
      #     coordinator.clear_stop()

      #   trainer.queue(validation_queue)
      #   trainer.validate(sess, is_training)

      #   """




      #   """

      #   # print("threads after validation: \n", threading.enumerate())

      #   # trainer.increment_counter()
