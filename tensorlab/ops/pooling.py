import tensorflow as tf


def max_pooling_3d(layer_input, ksize, strides, padding='SAME'):
  assert len(ksize) == 5  # [batch, depth, rows, cols, channels]
  assert len(strides) == 5  # [batch, depth, rows, cols, channels]
  assert ksize[0] == ksize[4]
  assert ksize[0] == 1
  assert strides[0] == strides[4]
  assert strides[0] == 1

  return tf.nn.max_pool3d(layer_input, ksize, strides, padding)
