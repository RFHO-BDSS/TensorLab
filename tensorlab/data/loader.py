import os
import json

import tensorflow as tf

import lib.io.schema as schema_io
from lib.data.readers.CTAReader import CTAReader
# from lib.data.augmenters.CTAAugmenter import CTAAugmenter
# from lib.data.queues.CTAQueue import CTAQueue


def _readers(input_schema):
  data_dir = input_schema['dirs']['data']
  split_type = input_schema['names']['type']
  splits_name = input_schema['names']['split']

  # split the path for naming
  path, partition = os.path.split(data_dir)

  schema_path = ('/home/none-exist/Documents/ArterySegmentation'
                 '/lib/data/splits/{0}/{1}/{2}'.format(
                     partition, split_type, splits_name))

  # load the image id splits
  splits_schema = schema_io._load(schema_path)

  datasets = {}

  for split in splits_schema.keys():

    # with tf.variable_scope(split, reuse=True):
    reader = CTAReader(data_dir, splits_schema[split], split)

    datasets[split] = reader

  return datasets


# def _augmenter(output_shape, spacing_target):
#   return CTAAugmenter(output_shape, spacing_target)


# def _queue(reader, augmenter, placeholders,
#            capacity, dtypes, shapes,
#            name):

#   return CTAQueue(reader, augmenter, placeholders,
#                   capacity, dtypes, shapes,
#                   name)
