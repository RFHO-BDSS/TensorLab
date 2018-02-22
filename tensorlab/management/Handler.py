import os
import threading

import datetime
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

import lib.io.session as session_io
import lib.io.writers as writers_io


class Handler(object):

  def __init__(self, output):

    config = tf.ConfigProto()
    # inter_op_parallelism_threads=20,
    # intra_op_parallelism_threads=1)
    # config.gpu_options.allow_growth = True
    # self._sess = tf.InteractiveSession(config=config)
    # self._sess = tf.train.MonitoredSession()
    # config=config)  #
    self._sess = tf.Session(config=config)

    self._output = output

    self._partitions = None
    self._placeholders = None
    self._reader = None
    self._augmenter = None
    self._queue = None
    self._coord = None
    self._graph = None

  @property
  def initialize(self):

    # raise Exception('Handler initialization must be overwritten')

    
    self._sess.run(tf.global_variables_initializer())
    self._sess.run(tf.local_variables_initializer())

    # # only initialize after graph def
    # # with tf.variable_scope('counters/'):
    # self.counter = tf.Variable(
    #     0,
    #     name='counters/local_step',
    #     trainable=False,
    #     dtype=tf.int32)
    # tf.summary.scalar('counters/local_step', self.counter)

    # self.writers = writers_io._open(self)

    # self._sess.run(tf.global_variables_initializer())
    # self._sess.run(tf.local_variables_initializer())

  """


  """

  def partitions(self, partitions):
    self._partitions = partitions

  def coord(self, coord):
    self._coord = coord

  def reader(self, reader):
    self._reader = reader

  def augmenter(self, augmenter):
    self._augmenter = augmenter

  def queue(self, queue):
    self._queue = queue

  def placeholders(self, placeholders):
    self._placeholders = placeholders

  def graph(self, graph):
    self._graph = graph

  def has_graph(self):
    return bool(self._graph)

  def summaries(self, summaries):
    self._summaries = summaries

  """


  """

  def save_session(self):
    session_io._save(self)

  def restore_session(self, checkpoint_path):
    session_io._restore(self,checkpoint_path)

  """


  """

  def merge_summaries(self):
    return tf.summary.merge_all().run()

  def write_summaries(self, partition, summaries, step):
    self.writers[partition].add_summary(summaries, step)

  """


  """

  """


  """

  def increment_counter(self, sess):
    sess.run(tf.assign(self.counter, self.counter+1))

  def step(self, sess):
    return sess.run(self.counter)

  @property
  def time(self):
    now = datetime.datetime.now()
    return str(now.strftime('%Y%m%d-%H%M%S'))

  @property
  def default(self):
    return self._sess.as_default()

  """


  """

  def _devices(self):
    devices = self._sess.list_devices()
    for d in devices:
      print(d.name)

  """


  """

  def __enter__(self):

    # set the start time and print commencement
    self._start = timer()
    print("\n----> Session commencing at {0}\n".format(self.time))

    return self

  def __exit__(self, type, value, tb):

    # set the start time and print commencement
    self._finish = timer()
    print("\n----> Session finishing at {0}".format(self.time))
    print("    Time elapsed is {0}\n".format(
        str(datetime.timedelta(seconds=self._finish - self._start))))

    # if self._queue is not None:
    #   queue_close_op = self._queue.close(cancel_pending_enqueues=True)
    #   self._queue.spawner.join()
    #   self._sess.run(queue_close_op)

    # while threading.active_count() > 0:
    #   time.sleep(0.1)

    # for writer in self.writers:
    #   writer.close()

    # self._sess.close()

    # if self.regime.evaluation.write_summaries:
    #   self.train_writer.close()
    #   self.val_writer.close()

    pass
