import os


def _print(args):
  """
  Prints all entries in FLAGS variable.
  """
  print("\nFLAGS: ")
  for key, value in vars(args).items():
    # key + ' : ' + str(value))
    print('{0:20s} {1:60s}'.format(key, str(value)))
