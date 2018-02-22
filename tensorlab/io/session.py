import os
import json

import tensorflow as tf
# import numpy as np


def _restore(handler, checkpoint_path):
  saver = tf.train.Saver()

  # if separator > 0:
  #   checkpoint_path = handler.output.dirs.checkpoints[:separator] + '/' + \
  #       handler.output.names.ckpt
  # else:
  # checkpoint_path = handler.output.dirs.checkpoints + '/' + \
  #     handler.output.names.ckpt
  # meta_path = checkpoint_path + '.meta'

  # raise NotImplementedError

  try:
    saver.restore(handler._sess, checkpoint_path)
    print("\n---{ Trained model loaded.")
  except:
    handler._sess.close()
    print("Model loading failed, cancelling run")
    raise ValueError


# def _restore_from_meta(handler):
#   checkpoint_path = handler.output.dirs.checkpoints + '/' + \
#       handler.output.names.ckpt
#   meta_path = checkpoint_path + '.meta'

#   meta_graph = tf.train.import_meta_graph(meta_path)
#   handler.graph = meta_graph.restore(handler.sess, checkpoint_path)


def _save(handler):
  save_checkpoint = os.path.join(
      handler._output['dirs']['logs'],
      handler._output['ext']['ckpt'],
      handler._output['names']['ckpt'])

  # print(save_checkpoint)
  saver = tf.train.Saver(tf.global_variables())
  saver.save(handler._sess, save_checkpoint)


def _summary(handler):
  pass
