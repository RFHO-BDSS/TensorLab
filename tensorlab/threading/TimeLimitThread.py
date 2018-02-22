
import threading


class TimeLimitExpired(Exception):
  pass


def TimeLimitThread(target, coord, timeout, args=(), kwargs={}):
  """ Run target with the given timeout. If target didn't finish running
      within the timeout, raise TimeLimitExpired
  """
  class FuncThread(threading.Thread):

    def __init__(self):
      threading.Thread.__init__(self)
      self.result = None

    def run(self):
      self.result = target(*args, **kwargs)

  while not coord.should_stop():
    it = FuncThread()
    it.start()
    it.join(timeout)
    if it.isAlive():
      pass
      # raise TimeLimitExpired()
    else:
      return it.result
