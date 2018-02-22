import threading
import multiprocessing

import numpy as np
import tensorflow as tf

import datetime
import time

# from lib.utils.timeout import *
from lib.threading.TimeLimitThread import TimeLimitThread


class Queue(tf.PaddingFIFOQueue):

  def __init__(
          self, capacity, dtypes, shapes, name):
    super().__init__(capacity, dtypes,
                     shapes=shapes, name=name)

    self._name = name

    self._threads = []
    self._thread_limit = None

  @property
  def name(self):
    return self._name

  @property
  def num_threads(self):
    return len(self._threads)

  @property
  def threads(self):
    return self._threads

  # @property
  def len(self, sess):
    return sess.run(self.size())

  def summary(self, sess):
    print(
        '{0:>20} size {1:>4}, {2:>3} / {3:>3} threads at {4}'.format(
            self.name,
            # sess.run(reader.size()),
            len([t for t in self._threads if t.is_alive()]),
            self.len(sess),  # sess.run(reader.size()),
            self.num_threads,
            '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
    )

  def main(self):

    raise NotImplementedError

  def start(self):

    cpu_count = multiprocessing.cpu_count()

    print("\n--{0} cores in the current machine --\n".format(cpu_count))

    raise NotImplementedError

  def join(self):
    while any([t.is_alive() for t in self._threads]):
      for t in [thread for thread in self._threads
                if not thread.is_alive()]:
        t.join()

      print(self._threads)

    self._threads = []
