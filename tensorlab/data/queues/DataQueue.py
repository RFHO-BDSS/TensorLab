import threading
import multiprocessing

import time

import numpy as np
import tensorflow as tf

# from lib.data.augmenters.CTAAugmenter import CTAAugmenter

from scipy.ndimage.filters import gaussian_filter
from itertools import product


class DataQueue(object):

  def __init__(
          self, coord, placeholders,
          reader,  # augmenter,
          capacity, min_after_dequeue,
          dtypes, shapes, name):

    self._name = name

    self._workers = []
    self._worker_limit = 12

    self._min = min_after_dequeue
    self._max = capacity

    # self._lock = threading.Lock()

    self._coord = coord
    self._reader = iter(reader)
    # self._augmenter = iter(augmenter)

    self._source = placeholders[0]
    self._target = placeholders[1]

    with tf.name_scope(self._name + '_enqueuing'):
      # self._queue = tf.PaddingFIFOQueue(
      self._queue = tf.RandomShuffleQueue(
          self._max, self._min,
          dtypes,
          shapes=shapes,
          name=name)

      self.enqueue = self._queue.enqueue([
          self._source,
          self._target])

      self.close = self._queue.close(
          cancel_pending_enqueues=True)

  @property
  def size(self):
    return self._queue.size()

  
  @property
  def min(self):
    return self._min


  @property
  def name(self):
    return self._name

  @property
  def num_workers(self):
    return len(self._workers)

  @property
  def is_closed(self):
    return self._queue.is_closed()

  # def load_batch(self, batch_size, augmentation):

  #   batch = {
  #       'jitterslice': lambda num_samples, multiplier:
  #       self.data.next_batch_with_jitterslice(num_samples, multiplier),
  #       'jitter': lambda num_samples, multiplier:
  #       self.data.next_batch_with_jitter(num_samples, multiplier),
  #       'gaussian': lambda num_samples, _:
  #       self.data.next_batch_with_gaussian(num_samples),
  #       'batch': lambda num_samples, _:
  #       self.data.next_batch(num_samples)
  #   }

  #   source, target = batch[augmentation.type](
  #       batch_size, augmentation.multiplier)

  #   # if self.data.train.epochs_completed != self.epoch:
  #   #   self.coord.request_stop()
  #   #   self.epoch = self.data.train.epochs_completed

  #   return source, target

  def dequeue(self, batch_size):

    # with tf.name_scope(self.schema.name + '_dequeuing'):
    source, target = self._queue.dequeue_many(
        batch_size, name=self._name + '_dequeuing')  # num_elements)
    # source, target = self.queue.dequeue_many(num_elements)
    return source, target

  def array_sparsity(self, input_array, threshold=0.5):


    array_sparsity = np.mean(
        np.where(input_array > threshold, 1, 0))
    # array_sparsity = np.nonzero(input_array)[0].size / \
    #     input_array.size

    return array_sparsity

  def _sources(self, arrays):

    # sources = []

    # print(arrays[0].shape)

    if len(arrays) > 1:
      sources = np.stack(arrays, axis=-1)

    else:
      sources = np.expand_dims(arrays[0], axis=-1)

    # sources = np.expand_dims(sources, axis=0)

    return sources

  def _targets(self, arrays):

    targets = []

    for array in arrays:
      targets.append(array)
      # targets.append(gaussian_filter(array, sigma=1))
      targets.append(gaussian_filter(array, sigma=3))
      targets.append(gaussian_filter(array, sigma=5))
      targets.append(gaussian_filter(array, sigma=7))

    targets = np.stack(targets, axis=-1)

    target_avg = np.mean(targets, axis=-1)
    target_std = np.std(targets, axis=-1)

    targets = np.stack([target_avg, target_std],
                       axis=-1)  # np.expand_dims(, 0)

    return targets

  def queue_runner(self, sess): 
    while not self._coord.should_stop():

      arrays, _ = next(self._reader)

      grid_points = []

      for dim in arrays[0].shape:
        grid_points.append(np.arange(0, dim, 32))

      grid = list(product(*grid_points))

      for point in grid:
        slices = []
        for p in point:
          slices.append(slice(p, p + 32))

        samples = []
        for array in arrays:
          samples.append(
              array[slices[0], slices[1], slices[2]])

        # print(samples)
        sparsity = self.array_sparsity(samples[0])


        if not sparsity > 0.05:
          continue

        if any([arrays[0].shape[i] - point[i]
                         < 32 for i in range(3)]):
          continue

        # print("  sample sparsity: ", np.mean(
        #     np.where(samples[0] > 0.5, 1, 0)))
        # print(sparsity)


        if not self._coord.should_stop():
          source = self._sources([samples[0]])
          target = self._sources([samples[1]])
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

          time.sleep(0.01)


          if not self._coord.should_stop():
            # print(self._coord.should_stop())
            try:
              sess.run(
                  self.enqueue,
                  feed_dict={
                      self._source: source,
                      self._target: target})
              time.sleep(0.1)
            except Exception as e:
              print("loader failed at ", e)
              pass
          else:
            break
          
        else:
          break
      # except Exception as e:
      #   print("loader failed at ", e)

    return None

  def start_workers(self, sess): #, coord):  # , augmenter):

    # self._workers = []

    for _ in range(self._worker_limit):
      # worker = multiprocessing.Process(
      worker = threading.Thread(
          target=self.queue_runner,
          args=(sess, )) # coord))

      # worker.setDaemon(True)
      worker.start()
      self._workers.append(worker)

      time.sleep(5)

    return self._workers


  def stop_workers(self):

    # print("workers before join: \n",
    # [worker.is_alive() for worker in self._workers])

    for worker in self._workers:
      # worker._stop()
      worker.join()

    self._workers = []

    # print("workers after join: \n",
    # [worker.is_alive() for worker in self._workers])
    

    # with tf.name_scope('FIFO_' + self.queue.name):

    # train_op_queue = tf.PaddingFIFOQueue(
    #     10,
    #     [
    #         self.graph.handles.metrics.loss.dtype,
    #         self.optimizations.handles.global_step.dtype,
    #         self.optimizations.handles.optimizer.dtype
    #     ],
    #     [
    #         self.graph.handles.metrics.loss.shape,
    #         self.optimizations.handles.global_step.shape,
    #         self.optimizations.handles.optimizer.shape
    #     ]
    # )
    # enqueue_placeholder = [
    #     self.graph.handles.metrics.loss,
    #     self.optimizations.handles.global_step,
    #     self.optimizations.handles.optimizer
    # ]
    # enqueue_op = filename_queue.enqueue(enqueue_placeholder)

    # train_coord = tf.train.Coordinator()
    # train_workers = tf.train.start_queue_runners(coord=train_coord)
    # tf.placeholder(
    #     dtype=source_type,
    #     shape=source_shape,
    #     name=source_name)
    # tf.placeholder(
    #     dtype=target_type,
    #     shape=target_shape,
    #     name=target_name)
