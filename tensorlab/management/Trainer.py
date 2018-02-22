import os
import copy

import threading
import time

from datetime import datetime

import numpy as np
import tensorflow as tf

from lib.management.Handler import Handler

import lib.io.schema as schema_io
import lib.io.writers as writers_io

METRICS_COLLECTION = {
    "precision": [],
    "recall": [],
    "specificity": [],
    "dice": []
}

class Trainer(Handler):

  def __init__(
          self,
          regime):
    super().__init__(regime['output'])

    # self.args = ARGS
    self._regime = regime

    self._loss = None
    self._optimizer = None

    average = None
    self.average_loss = tf.Summary()
    self.average_loss.value.add(
        tag="average-loss",
        simple_value=average)

    schema_io._save(
        regime,
        regime['output']['dirs']['logs'] + '/schemas/regime.json')

  
  @property
  def initialize(self):

    # only initialize after graph def
    # with tf.variable_scope('counters/'):
    self.counter = tf.Variable(
        0,
        name='counters/local_step',
        trainable=False,
        dtype=tf.int32)
    tf.summary.scalar('counters/local_step', self.counter)

    self.writers = writers_io._open(self)

    self._sess.run(tf.global_variables_initializer())
    self._sess.run(tf.local_variables_initializer())


  def loss(self, loss):
    self._loss = loss

  def metrics(self, metrics):
    self._metrics = metrics

  def optimizer(self, optimizer):
    self._optimizer = optimizer


  def train(self, sess, is_training):

    # if self._coord.should_stop():
    #   self._coord.clear_stop()

    print("---> Training\n")

    self.train_collection = copy.deepcopy(METRICS_COLLECTION)

    avg_loss = []
    # avg_size = []

    batch_limit = 5e3 # 1e3 # 1e1 # 
    summ_freq = 1e3 # 2e2 # 2 # 
    avg_freq = summ_freq / 2 # 5e2

    # i = self.step(sess)
    # start = self.step(sess)
    i = sess.run(self._optimizer.global_step, {is_training: True})
    start = i

    # print(i, start, i - start)

    first_eval = True

    # try:
    # workers = 
    self._queue.start_workers(sess) # , self._coord)
    #, self._augmenter)

    # dead_workers = [worker for worker in workers if not worker.is_alive()]
    # alive_workers = [worker for worker in workers if worker.is_alive()]


    # for dead in dead_workers:
    #   dead.join()
    # workers = alive_workers + dead_workers

    # print("before training: \n",
    # [worker.is_alive() for worker in self._queue._workers])


    # print("before while")

    while i - start < batch_limit:

      # print("begin while")

      # if i - start == batch_limit - 1:  # \
      #     # (self._queue.num_workers + 1):
      #   self._coord.request_stop()

      query = [
              self._loss,
              self._optimizer.global_step,
              self._optimizer.apply_gradients
          ]


      # this is really starting to get hacky
      # i+1 because i is incremented after query
      if (i + 1) % avg_freq == 0 or first_eval:
        query += [self._metrics]


      if (i + 1) % summ_freq == 0 or first_eval:
        query += [self._summaries]

      # print(query)

      results = sess.run(
          query,
          {
              is_training: True
          }
      )

      # print(results)

      loss = results[0]
      i = results[1]

      # print("loss after train op at {0:>6}: {1:>20.3f}".format(
      #     i, loss))

      avg_loss.append(loss)

      if i % avg_freq == 0:
        average = np.mean(avg_loss)

        self.average_loss.value[0].simple_value = average
        self.writers['train'].add_summary(
            self.average_loss, str(i))

        # order of results values may change depending
        #  on avg_/summ_freq, leave for now and assume multiple
        train_metrics = results[3] 
        # sess.run(self._metrics, {is_training: True})

        self.train_collection = self._sort_metrics(
            train_metrics, self.train_collection)

        print("--> Average loss: {0:>20.3f} at iter {1:>5}\n".format(
            average, i))

        # if i % 10 == 0:
        avg_loss = []

      if i % summ_freq == 0 or first_eval:
        if first_eval:
          first_eval = False

        print('-----> Writing summaries for {0:>5} at {1}\n'.format(
            str(i), self.time))
        summaries = results[4]
        # sess.run(
        #     self._summaries,
        #     {
        #         is_training: True
        #     })
        self.writers['train'].add_summary(
            summaries, i)

      if i - start == batch_limit - 1:  # \
          # (self._queue.num_workers + 1):
        # print("training queue size at end of epoch: ",
        # sess.run(self._queue.size, {is_training: False}))
      
        # print("in dodgy logic")
        # while sess.run(self._queue.size, {is_training: True}) \
        #         < 2 * self._queue.min:
        #   # print("here")
        #   # print(sess.run(self._queue.size, {is_training: True}))
        #   time.sleep(1)

        while sess.run(self._queue.size) \
            < 2 * self._queue.min:
          # print("here")
          time.sleep(1)

        self._coord.request_stop()

      # print("end while")

    # print("before threads join")

    self._coord.request_stop()
    # sess.run(self._queue.close)

    # print(self._coord.should_stop())

    # print("after training: \n",
    # [worker.is_alive() for worker in self._queue._workers])

    # print("before join")


    self._queue.stop_workers()

    # print("after join")

    # for worker in workers:
    #   try:
    #     worker.join(0.1)
    #   except Exception as e:
    #     print(e)
    #     pass

    # for worker in workers:
    #   worker.join()

    # try:
    #   self._coord.join(
    #       workers,
    #       stop_grace_period_secs=2)
    # except Exception as e:
    #   print(e)

    # self.readers.training.threads = []

    # print("before increment")

    # counter is a variable for the current epoch
    self.increment_counter(sess)


  def validate(self, sess, is_training):

    # if self._coord.should_stop():
    #   self._coord.clear_stop()

    print("---> Validating\n")

    self.validation_collection = copy.deepcopy(METRICS_COLLECTION)

    # pass

    avg_loss = []
    # avg_size = []

    batch_limit = 2e2

    i = sess.run(
      self._optimizer.global_step, 
      {is_training: False})

    # print("i: ", i)

    cnt = 0
    # start = self.step(sess)

    # first_eval = True

    # try:
    # workers = 
    self._queue.start_workers(sess) # , self._coord)
    #, self._augmenter)
    
    # print("before validation: \n",
    # [worker.is_alive() for worker in self._queue._workers])



    # print("before while")

    while cnt < batch_limit:

      loss, summaries, validation_metrics = sess.run(
          [
              self._loss,
              self._summaries,
              self._metrics
          ],
          {
              is_training: False
          }
      )

      self.validation_collection = self._sort_metrics(
          validation_metrics, self.validation_collection)

      # print("validation loss: {0:>20.3f}".format(
      #     loss))

      
      # print("during validation: \n",
      # [worker.is_alive() for worker in workers])



      avg_loss.append(loss)

      self.writers['val'].add_summary(
          summaries, i)

      # if cnt > batch_limit - \
      #         (self._queue.num_workers + 1):
      #     self._coord.request_stop()

      # if cnt == batch_limit - 1:  # \
      #     # (self._queue.num_workers + 1):
      #   self._coord.request_stop()

      cnt += 1

      if cnt == batch_limit - 1:  # \
          # (self._queue.num_workers + 1):
        # print("validation queue size at end of epoch: ", 
        #   sess.run(self._queue.size) )

        while sess.run(self._queue.size) \
            < 2 * self._queue.min:
          # print("here")
          time.sleep(1)
        self._coord.request_stop()


    average = np.mean(avg_loss)

    train_averages = self._average_metrics(
        self.train_collection)
    validation_averages = self._average_metrics(
        self.validation_collection)

    self._print_metrics(
        train_averages, validation_averages)

    print("--> Average validation loss: {0:>20.3f}\n".format(
        average))

    self.average_loss.value[0].simple_value = average
    self.writers['val'].add_summary(
        self.average_loss, i)

    self._coord.request_stop()
    # sess.run(self._queue.close)

    
    # print("after validation: \n",
    # [worker.is_alive() for worker in self._queue._workers])


    # print("before join")

    self._queue.stop_workers()

    # for worker in workers:
    #   worker.join()

    # for worker in workers:
    #   try:
    #     worker.join(0.1)
    #   except Exception as e:
    #     print(e)
    #     pass

    # print("after join")

  """








    """

  def _sort_metrics(self, input_metrics, collection):

    collection['precision'] += [input_metrics[0]]
    collection['recall'] += [input_metrics[1]]
    collection['specificity'] += [input_metrics[2]]
    collection['dice'] += [input_metrics[3]]

    return collection

  def _average_metrics(self, collection):

    averages = []

    averages.append(np.mean(collection['precision']))
    averages.append(np.mean(collection['recall']))
    averages.append(np.mean(collection['specificity']))
    averages.append(np.mean(collection['dice']))

    return averages

  def _print_metrics(self, train, val):  # step,
    # print(
    #     "iter {0:>10}, epoch {1:>4} at {2:>20}\n".format(
    #         step,
    #         epoch,
    #         self.time()
    #     ))
    # print(
    #     "->     losses: {0:>3.3f} \
    # (train), {1:>3.3f} (val)".format(
    #         train[0],
    #         val[0]))
    print(
        "->   precision: {0:>3.3f} (train), {1:>3.3f} (val)".format(
            train[0],
            val[0]))
    print(
        "->      recall: {0:>3.3f} (train), {1:>3.3f} (val)".format(
            train[1],
            val[1]))
    print(
        "-> specificity: {0:>3.3f} (train), {1:>3.3f} (val)".format(
            train[2],
            val[2]))
    print(
        "->        dice: {0:>3.3f} (train), {1:>3.3f} (val)".format(
            train[3],
            val[3]),
        "\n")


"""



















"""

# def train(self, sess):
#   # with self.sess as sess:
#   """
#     Initial Validation
#   """

#   print(
#       '\n--- Training commenced at {0}\n'.format(self.time())
#   )

#   # self.__validate__(sess)

#   for epoch in range(1):
#     # self.regime.evaluation.max_epochs):
#     """
#       Perform Training
#     """

#     # print("before training")

#     if self._coord.should_stop():
#       self._coord.clear_stop()

#     self.__train__(sess)

#     """
#       Perform Validation
#     """

#     # print("after training")

#     # if self.coord.should_stop():
#     #   self.coord.clear_stop()

#     # print("before validation")
#     # self.__validate__(sess)
#     # print("after validation")

# # return self.graph
