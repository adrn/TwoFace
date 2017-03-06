# Standard library
import os
import sys

# Third-party
import astropy.units as u
import numpy as np
from thejoker.sampler import TheJoker, JokerParams

# Project
from ..sample_prior import make_prior_cache

def test_make_prior_cache(tmpdir):

    filename = str(tmpdir / 'prior_samples.h5')
    params = JokerParams(P_min=8*u.day, P_max=8192*u.day)
    joker = TheJoker(params)

    make_prior_cache(filename, joker, N=2**16, max_batch_size=2**14)
