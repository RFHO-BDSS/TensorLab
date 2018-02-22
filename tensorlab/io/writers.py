import os
import json

import tensorflow as tf


def _open(handler):

  # init summary dir path
  summary_dir = os.path.join(
      handler._output['dirs']['logs'],
      handler._output['ext']['smrs'])

  # prepare for writers
  writers = {}

  for partition in handler._partitions:
    # one writer per partition
    # print(summary_dir + '/{0}'.format(partition))
    writers[partition] = tf.summary.FileWriter(
        summary_dir + '/{0}'.format(partition))

  # plus an additional for the graph
  writers['graph'] = tf.summary.FileWriter(
      summary_dir, handler._sess.graph)

  return writers
