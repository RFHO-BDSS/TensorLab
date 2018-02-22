import tensorflow as tf

from lib.ops.initializers import xavier_uniform_dist_conv3d


def convolution_3d(layer_input, filter, strides, padding='SAME'):
  # [filter_depth, filter_height, filter_width, in_channels, out_channels]
  assert len(filter) == 5
  # must match input dimensions [batch, in_depth, in_height, in_width,
  # in_channels]
  assert len(strides) == 5
  assert padding in ['VALID', 'SAME']

  w = tf.Variable(initial_value=xavier_uniform_dist_conv3d(
      shape=filter), name='weights')
  b = tf.Variable(tf.constant(1.0, shape=[filter[-1]]),
                  name='biases')

  return tf.nn.conv3d(layer_input, w, strides, padding) + b


def deconvolution_3d(
        layer_input, filter, output_shape, strides, padding='SAME'):
  # [depth, height, width, output_channels, in_channels]
  assert len(filter) == 5
  # must match input dimensions [batch, depth, height, width, in_channels]
  assert len(strides) == 5
  assert padding in ['VALID', 'SAME']

  w = tf.Variable(initial_value=xavier_uniform_dist_conv3d(
      shape=filter), name='weights')
  b = tf.Variable(tf.constant(1.0, shape=[filter[-2]]),
                  name='biases')

  return tf.nn.conv3d_transpose(
      layer_input, w, output_shape, strides, padding) + b
