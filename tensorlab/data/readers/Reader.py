"""




"""

import os
import random

import tensorflow as tf


class Reader(object):

    def __init__(
            self, data_dir, data_ids, name=None, shuffle=True):
        # super().__init__()
        # print(split)

        self._name = name
        self._shuffle = shuffle
        self._data_dir = data_dir
        self._ids = data_ids
        self._num_examples = len(data_ids)
        self._epochs_completed = 0
        self._index_in_epoch = -1

    @property
    def ids(self):
        return self._ids

    @property
    def name(self):
        return self._name

    @property
    def epoch(self):
        return self._epochs_completed + 1

    @property
    def size(self):
        return self._num_examples

    @property
    def remaining(self):
        return self._num_examples - self._index_in_epoch - 1

    """




  """

    def _parse(self, image_dir):

        raise NotImplementedError

    def _next_index(self):
        # get old index and iter to new
        # start = self._index_in_epoch
        self._index_in_epoch += 1

        if self._index_in_epoch == self._num_examples:
            # print("here")
            # iter epochs completed and reset index
            self._epochs_completed += 1
            self._index_in_epoch = -1

            # shuffle the data
            if self._shuffle:
                random.shuffle(self._ids)

            # true for next op
            return self._next_index()

        else:

            # print("here")
            # true to continue current epoch
            return True

    def _iter_index(self):
        # get old index and iter to new
        # start = self._index_in_epoch
        self._index_in_epoch += 1

        if self._index_in_epoch == self._num_examples:
            # iter epochs completed and reset index
            self._epochs_completed += 1

            # shuffle the data
            if self._shuffle:
                random.shuffle(self._ids)

            # false at end of epoch
            return False

        else:

            # true to continue current epoch
            return True

    @property
    def reset(self):
        self._index_in_epoch = -1

    """




  """

    def next(self):

        # if current index < num_examples continue iteration
        if self._next_index():
            print("true")
        # get current dir
        print("index", self._index_in_epoch)
        current_dir = os.path.join(
            self._data_dir, self._ids[self._index_in_epoch])

        return self._parse(current_dir)

    def read(self, image_dir):
        return self._parse(image_dir)

    def __len__(self):
        return self._num_examples

    def __iter__(self):
        return self

    def __next__(self):

        # if current index < num_examples continue iteration
        if self._next_index():

            # try:
            image_dir = self._ids[self._index_in_epoch]
            # print(image_dir)
            # get current dir
            current_dir = os.path.join(
                self._data_dir, image_dir)
            return self._parse(current_dir)
            # except:
            #     print("here somehow")

        # else the epoch has finished
        else:
            raise StopIteration
