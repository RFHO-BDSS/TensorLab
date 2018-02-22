import os


class Augmenter(object):

  def __init__(self):
    self._samples = []
    pass

  """




  """

  def augment(self, sample):

    raise NotImplementedError

  def process(self, arrays, source_spacings):

    raise NotImplementedError

  """




  """

  @property
  def samples_remaining(self):
    return len(self._samples)

  @property
  def next_sample(self):
    if self.samples_remaining > 0:
      return self._samples.pop(0)
    else:
      return None

  @property
  def proceed(self):

    if self.samples_remaining > 0:
      return True

    else:
      return False

  """




  """

  def __len__(self):
    return self.samples_remaining

  def __iter__(self):
    return self

  def __next__(self):

    # if current index < num_examples continue iteration
    if self.proceed:
      return self.next_sample
      # sample = self.next_sample
      # return self.augment(sample)

    # else the epoch has finished
    else:
      raise StopIteration
