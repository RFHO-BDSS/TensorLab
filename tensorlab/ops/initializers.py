import tensorflow as tf


def xavier_normal_dist_conv3d(shape):
  return tf.truncated_normal(
      shape, mean=0,
      stddev=tf.sqrt(
          3. / (tf.reduce_prod(shape[:3]) * tf.reduce_sum(shape[3:]))))


def xavier_uniform_dist_conv3d(shape):
  with tf.variable_scope('xavier_glorot_initializer'):
    denominator = tf.cast(
        (tf.reduce_prod(shape[:3]) * tf.reduce_sum(shape[3:])), tf.float32)
    lim = tf.sqrt(6. / denominator)
    return tf.random_uniform(shape, minval=-lim, maxval=lim)
