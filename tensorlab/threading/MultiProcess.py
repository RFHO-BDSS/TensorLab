import multiprocessing


class ProcessThread(multiprocessing.Process):

  def __init__(self, filename, process):
    super().__init__()
    self.__filename = filename
    self.__process = process
    self.__handled = False
    # self.setDaemon(True)

  def run(self):
    # while not self.__handled:
    self.__process(self.__filename)

    print('\n{0} completed\n'.format(self.__filename))

    self.__handled = True

  # def handled(self):
  #   return self.__handled
