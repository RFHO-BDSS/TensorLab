import os
# import json

from datetime import datetime

# import tensorflow as tf
# import numpy as np

# from lib.schemas.Schema import Schema


def _make_dirs(path):
    """
    Initializes all folders in FLAGS variable.
    """

    if not os.path.exists(path):
        os.makedirs(path)


def _initialize(output):
    """
    Initializes all folders in FLAGS variable.
    """

    for folder in output.keys():
        _make_dirs(output[folder])


def _datetime(output, separator='/'):
    now = datetime.now()

    for folder in output.keys():
        output[folder] += separator + \
            str(now.strftime('%Y%m%d-%H%M%S'))

    return output
