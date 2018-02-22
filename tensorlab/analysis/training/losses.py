import tensorflow as tf
import numpy as np


def mean_loss(losses):
  with tf.name_scope("mean-loss") as scope:
    # mean = None
    # for

    stacked_loss = tf.stack(losses, axis=-1)
    tf.summary.histogram('histogram', stacked_loss)

    loss = tf.reduce_mean(stacked_loss)
    tf.summary.scalar('scalar', loss)

  return loss


# def total_loss(vae_loss, rec_loss):
#   with tf.name_scope("total_loss") as scope:
#     loss = tf.reduce_mean(vae_loss + rec_loss)

#     scalar_summary(loss, name='losses')

#   return loss


# def prior_loss(vae_loss, rec_loss, prior_loss):
#   with tf.name_scope("prior_loss") as scope:
#     loss = tf.reduce_mean(vae_loss + rec_loss + prior_loss)

#     scalar_summary(loss, name='losses')

#   return loss
