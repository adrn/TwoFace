from __future__ import division, print_function

# Standard library
import os
import sys
import tempfile

# Third-party
from astropy.io import fits
import numpy as np

# Project
# ...

def injest_allVisit(cursor, allVisit_path):
    """

    """

    tbl = fits.getdata(allVisit_path, 1)
    with tempfile.TemporaryFile() as f:
        np.savetxt(f, tbl)
        f.seek(0)

        cursor.copy_from(f, 'allvisit', sep=",")
