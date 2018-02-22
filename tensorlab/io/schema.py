import os
import json

# import tensorflow as tf
# import numpy as np

# from lib.schemas.Schema import Schema


def _load(path):
  with open(path, 'r') as schema_json:
    # , encoding='utf8'
    # print(schema_json)

    # print(schema_json)

    schema = json.load(schema_json)

  return schema


def _print(schema, name=None):
  """
  prints schema as a dictionary
  """
  if name is None:
    name = "Schema"

  # schema_dict = _read(schema)
  schema_json = json.dumps(
      schema,
      indent=2,
      sort_keys=True)

  print("\n{0}: \n{1}".format(name, schema_json))


def _save(schema, path):

  schema_dir = os.path.split(path)[0]

  if not os.path.exists(schema_dir):
    os.makedirs(schema_dir)

  with open(path, 'w') as schema_json:
    # , encoding='utf8'
    json.dump(
        schema,
        schema_json,
        indent=2,
        sort_keys=True)


def _examine(schema):

  # schema_dict = schema.__dict__.copy()
  schema_ex = schema.copy()

  for key in schema_ex.keys():
    if isinstance(schema_ex[key], dict):
      schema_ex[key] = _examine(schema_ex[key])
    else:
      schema_ex[key] = "{0}".format(type(schema_ex[key]))

  return schema_ex


def _brief(schema, name=None):
  """
    display brief of schema
      output: key:type(value)
  """
  if name is None:
    name = "Schema"

  schema_ex = _examine(schema)
  schema_json = json.dumps(
      schema_ex,
      indent=2,
      sort_keys=True)

  print("\n{0}: \n{1}".format(name, schema_json))
