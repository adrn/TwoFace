from __future__ import division, print_function

# Standard library
import tempfile

__all__ = ['copy_from_table']

def copy_from_table(cursor, table, Table):
    """
    """

    copy_colnames = [str(x).split('.')[1].upper() for x in Table.__table__.columns]

    # automatically figure out which columns are multidimensional
    tbl_colnames = []
    for colname in table.colnames:
        if colname not in copy_colnames:
            continue

        tbl_colnames.append(colname)
        dtype,_ = table.dtype.fields[colname]

        if len(dtype.shape) == 0: # skip scalar columns
            continue

        # turn the array into a string, following:
        #   http://stackoverflow.com/questions/11170099/copy-import-data-into-postgresql-array-column
        table[colname] = [str(x.tolist()).replace('[','{').replace(']','}')
                          for x in table[colname]]

    with tempfile.TemporaryFile(mode='r+') as f:
        table[tbl_colnames].write(f, format='ascii.fast_no_header', delimiter='\t')
        f.seek(0)
        cursor.copy_from(f, Table.__tablename__, sep="\t", columns=tbl_colnames)

