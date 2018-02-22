import multiprocessing


class LoadProcess(multiprocessing.Process):

  def __init__(self, reader, augmenter, sess):
    super().__init__()
    self.__reader = reader
    self.__augmenter = process
    self.__handled = False
    # self.setDaemon(True)

  def run(self):

    while not self.__handled:

      # while not self.__handled:
      # self.__process(self.__filename)

      # print('\n{0} completed\n'.format(self.__filename))

      self.__handled = True

  # def handled(self):
  #   return self.__handled
