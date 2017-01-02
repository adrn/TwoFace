from __future__ import division, print_function

# Standard library
from collections import OrderedDict
import tempfile

# Third-party
import numpy as np
from sqlalchemy import Column, types
from sqlalchemy.dialects import postgresql

__all__ = ['table_to_sql_columns', 'copy_from_table']

numpy_type_map = dict()
numpy_type_map[np.float32] = types.REAL
numpy_type_map[np.float64] = types.Numeric
numpy_type_map[np.int16] = types.SmallInteger
numpy_type_map[np.int32] = types.Integer
numpy_type_map[np.int64] = types.BigInteger
numpy_type_map[np.str_] = types.String

def table_to_sql_columns(table, skip=None):
    """
    Convert an `~astropy.table.Table` to a dictionary of
    `sqlalchemy.Column` objects.
    """

    if skip is None:
        skip = []

    col_map = OrderedDict()
    for name in table.columns:
        if name in skip:
            continue

        dtype,_ = table.dtype.fields[name]
        sql_type = numpy_type_map[table[name].dtype.type]

        if len(dtype.shape) > 0:
            sql_type = postgresql.ARRAY(sql_type)

        col_map[name.lower()] = Column(name.lower(), sql_type)

    return col_map

def copy_from_table(cursor, table, table_name, skip=None):
    """
    """

    if skip is None:
        skip = []

    # automatically figure out which columns are multidimensional
    copy_colnames = []
    for colname in table.colnames:
        if colname in skip:
            continue

        copy_colnames.append(colname)
        dtype,_ = table.dtype.fields[colname]

        if len(dtype.shape) == 0: # skip scalar columns
            continue

        # turn the array into a string, following:
        #   http://stackoverflow.com/questions/11170099/copy-import-data-into-postgresql-array-column
        table[colname] = [str(x.tolist()).replace('[','{').replace(']','}')
                          for x in table[colname]]

    with tempfile.TemporaryFile(mode='r+') as f:
        table[copy_colnames].write(f, format='ascii.fast_no_header', delimiter='\t')
        f.seek(0)
        cursor.copy_from(f, table_name, sep="\t", columns=copy_colnames)

