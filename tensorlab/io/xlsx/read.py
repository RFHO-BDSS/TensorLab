import os

import xlrd


def _by_col_values(
        file_path,
        sheet_num,  # = 0
        col_num,  # = 1
        col_values,  # = ['H20f']
        row_parser=None):  # = lambda x : x[0].split('/')[-2]

  # reads an xlsx workbook
  # checks a specific column value for each row
  # file_path: path/to/workbook.xlsx
  # sheet: sheet from workbook to search
  # col_num: sheet column to search
  # col_values: list of values to search for
  # parser: (optional) a function for parsing any rows
  #         with the given column value. If None,
  #         returns the whole row

  results = []

  # open the excel workbook and sheet
  workbook = xlrd.open_workbook(file_path)
  sheet = workbook.sheet_by_index(sheet_num)

  # iterate rows of the sheet
  for row_idx in range(sheet.nrows):
    row = sheet.row_values(row_idx)

    # if any column value matches any given pattern
    if any([row[col_num] == pattern for pattern in col_values]):

      # if there is a row parser supplied
      if row_parser is not None:
        # apply it
        result = row_parser(row)

      else:
        # else return the row
        result = row

      # collect all the results in a handy dandy python list
      results.append(result)

      # print(image_id)

  return results
