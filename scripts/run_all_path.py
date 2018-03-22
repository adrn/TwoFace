"""
Generate posterior samples over orbital parameters for all data files in a
specified directory.

How to use
==========

TODO: continue with MCMC?

"""

# Standard library
import glob
from os import path
import os
import sys
import time

# Third-party
from astropy.table import QTable
import astropy.units as u
import h5py
import numpy as np
from schwimmbad import choose_pool
from thejoker.log import log as joker_logger
from thejoker import TheJoker, JokerParams, RVData
import yaml

# Project
from twoface.log import log as logger
from twoface.sample_prior import make_prior_cache


def main(data_path, config_file, data_file_ext, pool, seed, overwrite=False):
    # parse config file
    with open(config_file, 'r') as f:
        config = yaml.load(f.read())
        config['config_file'] = config_file

    cache_path = path.join(data_path, 'cache')
    os.makedirs(cache_path, exist_ok=True)

    P_min = u.Quantity(*config['hyperparams']['P_min'].split())
    P_max = u.Quantity(*config['hyperparams']['P_max'].split())
    n_samples_per_star = int(
        config['hyperparams']['requested_samples_per_star'])
    n_prior_samples = config['prior']['num_cache']

    if 'jitter' in config['hyperparams']:
        # jitter is fixed to some quantity, specified in config file
        jitter = u.Quantity(*config['hyperparams']['jitter'].split())
        logger.debug('Jitter is fixed to: {0:.2f}'.format(jitter))
        joker_pars = JokerParams(P_min=P_min, P_max=P_max,
                                 jitter=jitter)

    elif 'jitter_prior_mean' in config['hyperparams']:
        # jitter prior parameters are specified in config file
        jitter_mean = config['hyperparams']['jitter_prior_mean']
        jitter_stddev = config['hyperparams']['jitter_prior_stddev']
        jitter_unit = config['hyperparams']['jitter_prior_unit']
        logger.debug('Sampling in jitter with mean = {0:.2f} (stddev in '
                     'log(var) = {1:.2f}) [{2}]'
                     .format(np.sqrt(np.exp(jitter_mean)),
                             jitter_stddev, jitter_unit))
        joker_pars = JokerParams(P_min=P_min, P_max=P_max,
                                 jitter=(jitter_mean, jitter_stddev),
                                 jitter_unit=u.Unit(jitter_unit))

    else:
        joker_pars = JokerParams(P_min=P_min, P_max=P_max)

    prior_samples_file = path.join(cache_path, 'prior-samples.hdf5')

    # Create TheJoker sampler instance with the specified random seed and pool
    rnd = np.random.RandomState(seed=seed)
    logger.debug("Creating TheJoker instance with {0}, {1}".format(rnd, pool))
    joker = TheJoker(joker_pars, random_state=rnd, pool=pool)

    # Create a cache of prior samples (if it doesn't exist) and store the
    # filename in the database.
    if not os.path.exists(prior_samples_file) or overwrite:
        logger.debug("Prior samples file not found - generating {0} samples..."
                     .format(n_prior_samples))
        make_prior_cache(prior_samples_file, joker,
                         N=n_prior_samples,
                         max_batch_size=2**23) # MAGIC NUMBER
        logger.debug("...done")

    data_files = glob.glob(path.join(data_path, '*.{0}'.format(data_file_ext)))

    for filename in data_files:
        basename = path.splitext(path.basename(filename))[0]
        logger.info('Processing file {0}'.format(basename))

        joker_results_filename = path.join(cache_path,
                                           '{0}-joker.hdf5'.format(basename))

        data_tbl = QTable.read(filename)
        data = RVData(t=data_tbl['t'], rv=data_tbl['rv'],
                      stddev=data_tbl['rv_err'])
        continue

        if not path.exists(joker_results_filename) or overwrite:
            t0 = time.time()
            logger.log(1, "\t visits loaded ({:.2f} seconds)"
                       .format(time.time()-t0))
            try:
                samples = joker.rejection_sample(
                    data=data, prior_cache_file=prior_samples_file,
                    return_logprobs=False)

            except Exception as e:
                logger.warning("\t Failed sampling for star {0} \n Error: {1}"
                               .format(basename, str(e)))
                pool.close()
                sys.exit(1)

            logger.debug("\t done sampling ({:.2f} seconds)"
                         .format(time.time()-t0))

            # Write the samples that pass to the results file
            with h5py.File(joker_results_filename, 'w') as f:
                samples.to_hdf5(f)

            logger.debug("\t saved samples ({:.2f} seconds)"
                         .format(time.time()-t0))
            logger.debug("...done with star {} ({:.2f} seconds)"
                         .format(basename, time.time()-t0))

        # TODO: do MCMC if need be!

    pool.close()


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

    parser.add_argument("-s", "--seed", dest="seed", default=None, type=int,
                        help="Random number seed")

    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False,
                        action='store_true')

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    parser.add_argument("--data-path", dest="data_path", required=True,
                        type=str, help="Path to the data files to run on.")
    parser.add_argument("-c", "--config", dest="config_file", required=True,
                        type=str, help="Path to the config file.")
    parser.add_argument("--ext", dest="data_file_ext", default='csv',
                        type=str, help="Extension of data files.")

    args = parser.parse_args()

    loggers = [joker_logger, logger]

    # Set logger level based on verbose flags
    if args.verbosity != 0:
        if args.verbosity == 1:
            logger.setLevel(logging.DEBUG)
        else: # anything >= 2
            logger.setLevel(1)
            joker_logger.setLevel(1)

    elif args.quietness != 0:
        if args.quietness == 1:
            logger.setLevel(logging.WARNING)
            joker_logger.setLevel(logging.WARNING)
        else: # anything >= 2
            logger.setLevel(logging.ERROR)
            joker_logger.setLevel(logging.ERROR)

    else: # default
        logger.setLevel(logging.INFO)
        joker_logger.setLevel(logging.INFO)

    pool_kwargs = dict(mpi=args.mpi, processes=args.n_procs)
    pool = choose_pool(**pool_kwargs)

    main(data_path=args.data_path, pool=pool, seed=args.seed,
         config_file=args.config_file, overwrite=args.overwrite,
         data_file_ext=args.data_file_ext)
