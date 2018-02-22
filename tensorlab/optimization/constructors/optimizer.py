import tensorflow as tf
import numpy as np
# from lib.graphs.utils.summaries import scalar_summary


def get_optimizer(
        loss,
        learning_rate,
        global_step,
        clip_max=1,
        op='Momentum',
        name="Optimizer",
        **kwargs):

  optimizers = {
      'SGD': lambda
      learning_rate,
      use_locking=False,
      name='GradientDescent',
      **kwargs:
      tf.train.GradientDescentOptimizer(
          learning_rate,
          use_locking,
          name),
      'Adadelta': lambda
      learning_rate,
      rho=0.95,
      epsilon=1e-08,
      use_locking=False,
      name='Adadelta',
      **kwargs:
      tf.train.AdadeltaOptimizer(
          learning_rate,
          rho,
          epsilon,
          use_locking,
          name),
      'Adagrad': lambda
      learning_rate,
      initial_accumulator_value=0.1,
      use_locking=False,
      name='Adagrad',
      **kwargs:
      tf.train.AdagradOptimizer(
          learning_rate,
          initial_accumulator_value,
          use_locking,
          name),
      'Momentum': lambda
      learning_rate,
      momentum=0.9,
      use_locking=False,
      name='Momentum',
      use_nesterov=True,
      **kwargs:
      tf.train.MomentumOptimizer(
          learning_rate,
          momentum,
          use_locking,
          name,
          use_nesterov),
      'Adam': lambda
      learning_rate,
      beta1=0.9,
      beta2=0.999,
      epsilon=1e-08,
      use_locking=False,
      name='Adam',
      **kwargs:
      tf.train.AdamOptimizer(
          learning_rate,
          beta1,
          beta2,
          epsilon,
          use_locking,
          name),
      'RMS': lambda
      learning_rate,
      decay=0.9,
      momentum=0.0,
      epsilon=1e-10,
      use_locking=False,
      centered=False,
      name='RMSProp',
      **kwargs:
      tf.train.RMSPropOptimizer(
          learning_rate,
          decay,
          momentum,
          epsilon,
          use_locking,
          centered,
          name)
  }

  # clip_grads=True,
  with tf.name_scope(op + '-' + name) as scope:
    optimizer = optimizers[op](learning_rate, **kwargs)
    tvars = tf.trainable_variables()
    grads_and_vars = optimizer.compute_gradients(
        loss,
        tvars)

    # for g, t in grads_and_vars:
    #   print(g, t, '\n')

    clipped = [(
        tf.clip_by_value(
            grad,
            -clip_max,
            clip_max),
        tvar)  # gradient clipping
        for grad, tvar in grads_and_vars]
    train_op = optimizer.apply_gradients(
        clipped,
        global_step=global_step,
        name="SGD")

  return train_op
