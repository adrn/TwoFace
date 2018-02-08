import os
from os import path
import sys
import time

# Third-party
from astropy.io import fits
from astropy.table import Table, join
from astropy.time import Time
import astropy.units as u
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import schwimmbad

from twobody import KeplerOrbit

from thejoker import RVData, JokerParams, TheJoker, JokerSamples
from thejoker.plot import plot_rv_curves
from thejoker.log import log as logger


def make_data(n_epochs, n_per_epochs=1024, time_sampling='uniform', circ=False):
    """
    time_sampling can be 'uniform' or 'log'
    """

    # Hard-set MAGIC NUMBERs:
    t0 = Time('2013-01-01')
    baseline = 5 * u.yr # similar to APOGEE2
    K = 1 * u.km/u.s
    err = 150 * u.m/u.s

    if time_sampling == 'uniform':
        def t_func(size):
            return np.random.uniform(t0.mjd, (t0 + baseline).mjd, size=size)

    elif time_sampling == 'log':
        def t_func(size):
            t1 = np.log10(baseline.to(u.day).value)
            t = t0 + 10 ** np.random.uniform(0, t1, size=size) * u.day
            return t

    else:
        raise ValueError('invalid time_sampling value')

    # TODO: handle circ == False

    for N in n_epochs:
        orb = KeplerOrbit(P=150.*u.day, e=0., omega=0*u.deg, M0=0*u.deg)

        t = Time(t_func(N), format='mjd')
        rv = K * orb.unscaled_radial_velocity(t)
        data = RVData(t, rv, stddev=np.ones_like(rv.value) * err)



def make_prior_cache(N, circ=False):
    samples, ln_probs = thejoker.sample_prior(N, return_logprobs=True)
    packed_samples, units = pack_prior_samples(samples, u.km/u.s)

    batch_size, K = packed_samples.shape

    with h5py.File(filename, 'r+') as f:
        if 'samples' not in f:
            # make the HDF5 file with placeholder datasets
            f.create_dataset('samples', shape=(N, K), dtype=np.float32)
            f.create_dataset('ln_prior_probs', shape=(N,), dtype=np.float32)
            f.attrs['units'] = np.array([str(x)
                                         for x in units]).astype('|S6')

def main(pool, overwrite=False):
    n_epochs = np.arange(3, 12+1, 1)
    filename = make_data()

    pars = JokerParams(P_min=1*u.day, P_max=1024*u.day)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0,
                          dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0,
                          dest='quietness')

    # multiprocessing
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    # Note: default seed is set!
    parser.add_argument('-s', '--seed', dest='seed', default=42,
                        type=int, help='Random number generator seed.')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        dest='overwrite', default=False,
                        help='Destroy everything.')

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbosity != 0:
        if args.verbosity == 1:
            logger.setLevel(logging.DEBUG)
        else: # anything >= 2
            logger.setLevel(1)

    elif args.quietness != 0:
        if args.quietness == 1:
            logger.setLevel(logging.WARNING)
        else: # anything >= 2
            logger.setLevel(logging.ERROR)

    else: # default
        logger.setLevel(logging.INFO)

    if args.seed is not None:
        np.random.seed(args.seed)

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    main(pool=pool, overwrite=args.overwrite)
