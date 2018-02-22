import os

import signal

import json
import argparse

from datetime import datetime

import tensorflow as tf
import numpy as np


import lib.io.flags as flags_io
import lib.io.schema as schema_io
import lib.io.folders as folders_io

import lib.train as train
import lib.test as test

USE_MODE_DEFAULT = 'train'

TRAIN_SCHEMA_DEFAULT = './lib/schemas/training.json'
GRAPH_SCHEMA_DEFAULT = './lib/schemas/architecture.json'  # gaussnet.json'  #
# REFINE_GRAPH_DEFAULT = ''


def main(FLAGS):

    np.random.seed(42)
    tf.set_random_seed(42)

    # signal.signal(signal.SIGINT, signal.SIG_IGN)

    # add prefix for folder above
    now = datetime.now()
    print("\n----> Main commenced at {0}\n".format(
        str(now.strftime('%Y%m%d-%H%M%S'))))

    # if not FLAGS.use_mode in ['train', 'refine', 'schedule']:
    flags_io._print(FLAGS)

    if FLAGS.use_mode == 'train':
        # load the control training schema
        regime = schema_io._load(FLAGS.train_schema)

        # make folder unique through datetime
        regime['output']['dirs'] = folders_io._datetime(
            regime['output']['dirs'])
        folders_io._initialize(regime['output']['dirs'])

        # optional: print to std for debugging
        schema_io._print(regime)

        # load the control architecture schema
        # architecture = schema_io._load(FLAGS.graph_schema)

        # perform training
        print("\n---{ Training }\n")
        train._new(regime) # architecture, 


    if FLAGS.use_mode == 'test':
        # load the control training schema
        manual = schema_io._load(FLAGS.train_schema)

        # make folder unique through datetime
        # regime['output']['dirs'] = folders_io._datetime(
        #     regime['output']['dirs'])
        # folders_io._initialize(regime['output']['dirs'])

        # optional: print to std for debugging
        schema_io._print(manual)

        # load the control architecture schema
        # architecture = schema_io._load(FLAGS.graph_schema)

        # perform training
        print("\n---{ Training }\n")
        test._single(manual)

    # raise NotImplementedError


if __name__ == '__main__':

    # Command line arguments
    parser = argparse.ArgumentParser()

    # the fewer the better
    parser.add_argument(
        '--use_mode',
        type=str,
        default=USE_MODE_DEFAULT,
        help='Mode for model usage (train, refine, test)')
    parser.add_argument(
        '--train_schema',
        type=str,
        default=TRAIN_SCHEMA_DEFAULT,
        help='Location of the training regime schema')
    parser.add_argument(
        '--graph_schema',
        type=str,
        default=GRAPH_SCHEMA_DEFAULT,
        help='Location of the graph architecture schema')
    # parser.add_argument(
    #     '--refine_graph',
    #     type=str,
    #     default=REFINE_GRAPH_DEFAULT,
    #     help='Location of the graph to be refined')

    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)
