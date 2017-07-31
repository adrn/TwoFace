"""
Generate posterior samples over orbital parameters for all red clump stars in
the specified database file.

How to use
==========

This script is configured with a YAML configuration file that specifies the
parameters of the processing run. These are mainly hyper-parameters for
`The Joker <thejoker.readthedocs.io>`_, but  also specify things like the name
of the database file to pull data from, the name of the run, and the number of
prior samples to generate and cache. To see an example, check out the YAML file
at ``twoface/run_config/redclump.yml``.

Parallel processing
===================

This script is designed to be used on any machine, and supports parallel
processing on computers from laptop (via multiprocessing) to compute cluster
(via MPI). The type of parallelization is specified using the command-line
flags ``--mpi`` and ``--ncores``. By default (with no flags), all calculations
are done in serial.

TODO
====

- We might want a way to pass in a velocity trend specification, i.e. whether to
  sample over extra linear parameters to account for a long-term velocity trend.
  Right now we assume no long-term trend.
-

"""

# Standard library
from os.path import abspath, expanduser, join
import os
import time

# Third-party
import astropy.units as u
import h5py
import numpy as np
from schwimmbad import choose_pool
from sqlalchemy import func
from sqlalchemy.orm.exc import MultipleResultsFound, NoResultFound
from thejoker.log import log as joker_logger
from thejoker.sampler import JokerParams, TheJoker
from thejoker.utils import quantity_to_hdf5
import yaml

# Project
from twoface.data import APOGEERVData
from twoface.log import log as logger
from twoface.db import db_connect
from twoface.db import (JokerRun, AllStar, AllVisit, StarResult, Status,
                        AllVisitToAllStar, RedClump)
from twoface.config import TWOFACE_CACHE_PATH
from twoface.sample_prior import make_prior_cache

def main(config_file, pool, seed, overwrite=False, _continue=False):
    config_file = abspath(expanduser(config_file))

    # parse config file
    with open(config_file, 'r') as f:
        config = yaml.load(f.read())

    # filename of sqlite database
    if 'database_file' not in config:
        database_file = None

    else:
        database_file = config['database_file']

    db_path = join(TWOFACE_CACHE_PATH, database_file)
    if not os.path.exists(db_path):
        raise IOError("sqlite database not found at '{0}'\n Did you run "
                      "scripts/initdb.py yet for that database?"
                      .format(db_path))

    logger.debug("Connecting to sqlite database at '{0}'".format(db_path))
    Session, engine = db_connect(database_path=db_path,
                                 ensure_db_exists=False)
    session = Session()

    # See if this run (by name) is in the database already, if so, grab that.
    try:
        run = session.query(JokerRun).filter(
            JokerRun.name == config['name']).one()
        logger.info("JokerRun '{0}' already found in database"
                    .format(config['name']))

    except NoResultFound:
        run = None

    except MultipleResultsFound:
        raise MultipleResultsFound("Multiple JokerRun rows found for name '{0}'"
                                   .format(config['name']))

    # If this run doesn't exist in the database yet, create it using the
    # parameters read from the config file.
    if run is None:
        logger.info("JokerRun '{0}' not found in database, creating entry..."
                    .format(config['name']))

        # Create a JokerRun for this run
        run = JokerRun()
        run.config_file = config_file
        run.name = config['name']
        run.P_min = u.Quantity(*config['hyperparams']['P_min'].split())
        run.P_max = u.Quantity(*config['hyperparams']['P_max'].split())
        run.requested_samples_per_star = int(
            config['hyperparams']['requested_samples_per_star'])
        run.max_prior_samples = int(config['prior']['max_samples'])
        run.prior_samples_file = join(TWOFACE_CACHE_PATH,
                                      config['prior']['samples_file'])

        if 'jitter' in config['hyperparams']:
            # jitter is fixed to some quantity, specified in config file
            run.jitter = u.Quantity(*config['hyperparams']['jitter'].split())

        elif 'jitter_prior_mean' in config['hyperparams']:
            # jitter prior parameters are specified in config file
            run.jitter_mean = config['hyperparams']['jitter_prior_mean']
            run.jitter_stddev = config['hyperparams']['jitter_prior_stddev']
            run.jitter_unit = config['hyperparams']['jitter_prior_unit']

        else:
            # no jitter is specified, assume no jitter
            run.jitter = 0. * u.m/u.s

        # Get all stars that are also in the RedClump table with >3 visits
        stars = session.query(AllStar).join(AllVisitToAllStar, AllVisit,
                                            RedClump)\
                                      .group_by(AllStar.apstar_id)\
                                      .having(func.count(AllVisit.id) >= 3)\
                                      .all()

        run.stars = stars
        session.add(run)
        session.commit()

    if run.jitter is None or np.isnan(run.jitter):
        jitter_kwargs = dict(jitter=(float(run.jitter_mean),
                                     float(run.jitter_stddev)),
                             jitter_unit=u.Unit(run.jitter_unit))

    else:
        jitter_kwargs = dict(jitter=run.jitter)

    # Create TheJoker sampler instance with the specified random seed and pool
    rnd = np.random.RandomState(seed=seed)
    logger.debug("Creating TheJoker instance with {0}, {1}".format(rnd, pool))
    params = JokerParams(P_min=run.P_min, P_max=run.P_max, anomaly_tol=1E-11,
                         **jitter_kwargs)
    joker = TheJoker(params, random_state=rnd, pool=pool)

    # Create a cache of prior samples (if it doesn't exist) and store the
    # filename in the database.
    if not os.path.exists(run.prior_samples_file):
        logger.debug("Prior samples file not found - generating...")
        make_prior_cache(run.prior_samples_file, joker,
                         N=config['prior']['num_cache'],
                         max_batch_size=2**22) # MAGIC NUMBER
        logger.debug("...done")

    # Read units from prior samples file
    with h5py.File(run.prior_samples_file, 'r') as f:
        prior_units = [u.Unit(uu) for uu in f.attrs['units']]
        n_cached_samples = f['samples'].shape[0]

    # Query to get all stars associated with this run that need processing:
    # they should have a status id = 0 (needs processing)
    star_query = session.query(AllStar).join(StarResult, JokerRun, Status)\
                                       .filter(JokerRun.name == run.name)\
                                       .filter(Status.id == 0)

    # Base query to get a StarResult for a given Star so we can update the
    # status, etc.
    result_query = session.query(StarResult).join(AllStar, JokerRun)\
                                            .filter(JokerRun.name == run.name)\
                                            .filter(Status.id == 0)

    # Create a file to cache the resulting posterior samples
    results_filename = join(TWOFACE_CACHE_PATH, "{}.hdf5".format(run.name))
    n_stars = star_query.count()
    logger.info("{0} stars left to process for run '{1}'"
                .format(n_stars, run.name))

    # Ensure that the results file exists - this is where we cache samples that
    # pass the rejection sampling step
    if not os.path.exists(results_filename):
        with h5py.File(results_filename, 'w') as f:
            pass

    # --------------------------------------------------------------------------
    # Here is where we do the actual processing of the data for each star. We
    # loop through all stars that still need processing and iteratively
    # rejection sample with larger and larger prior sample batch sizes. We do
    # this for efficiency, but the argument for this is somewhat made up...

    count = 0 # how many stars we've processed in this star batch
    batch_size = 16 # MAGIC NUMBER: how many stars to process before committing
    for star in star_query.all():
        # retrieve existing StarResult from database
        result = result_query.filter(AllStar.apogee_id == star.apogee_id).one()

        logger.log(1, "Starting star {0}".format(star.apogee_id))
        t0 = time.time()

        data = APOGEERVData.from_visits(star.visits)
        logger.log(1, "\t visits loaded ({:.2f} seconds)"
                   .format(time.time()-t0))

        # adaptive scheme for the rejection sampling:
        n_process = 128 * run.requested_samples_per_star # 128 = MAGIC NUMBER
        n_samples = 0 # running total of good samples returned
        start_idx = 0
        all_samples = None
        all_ln_probs = np.array([])
        for n in range(128): # MAGIC NUMBER: but we should never hit this
            logger.debug("\t Iteration {}, using {} prior samples"
                         .format(n, n_process))

            # Process n_process samples from prior cache, pulled from the file
            # starting from the index `start_idx`
            samples, ln_probs = joker._rejection_sample_from_cache(
                data, n_process, run.prior_samples_file, start_idx,
                seed=rnd.randint(100000), return_logprobs=True)

            # Number of samples that survive the rejection sampling step
            n_survive = samples.shape[0]

            # Only store the samples if more than 1 survive
            if n_survive > 1:
                n_samples += n_survive

                if all_samples is None:
                    all_samples = samples
                else:
                    all_samples = np.vstack((all_samples, samples))

                all_ln_probs = np.concatenate((all_ln_probs, ln_probs))

            # End processing for this star if the number of samples we have is
            # larger than or equal to the number requested per star
            if n_samples >= run.requested_samples_per_star:
                break

            # Update the cache index to start pulling samples from
            start_idx += n_process

            # Increase the number of samples to process by a factor of two, so
            # at each iteration the number of samples processed increases. It's
            # not clear the 2 is the right increase - it might be 4, or 8!
            n_process *= 2 # MAGIC NUMBER

            # If the number of samples we would process in the next iteration is
            # larger than the total number of samples, stop. Also, if the number
            # of samples left in the
            if (n_process > run.max_prior_samples or
                    n_process > (n_cached_samples-start_idx)):

                if all_samples is None:
                    all_samples = samples
                    all_ln_probs = np.array(ln_probs)

                logger.debug("\t Hit max prior samples limit!!")
                break

        else:
            # should never get here
            raise RuntimeError("Hit maximum number of iterations!")

        logger.debug("\t done sampling ({:.2f} seconds)".format(time.time()-t0))

        # For now, it's sufficient to write the run results to an HDF5 file
        n = run.requested_samples_per_star
        samples_dict = joker.unpack_full_samples(all_samples[:n],
                                                 data.t_offset,
                                                 prior_units)

        # Write the samples that pass to the results file
        with h5py.File(results_filename, 'r+') as f:
            if star.apogee_id not in f:
                g = f.create_group(star.apogee_id)

            else:
                g = f[star.apogee_id]

            for key in samples_dict.keys():
                if key in g:
                    del g[key]
                quantity_to_hdf5(g, key, samples_dict[key])

            if 'ln_prior_probs' in g:
                del g['ln_prior_probs']
            g.create_dataset('ln_prior_probs', data=all_ln_probs)

        logger.debug("\t saved samples ({:.2f} seconds)".format(time.time()-t0))

        if all_samples.shape[0] >= run.requested_samples_per_star:
            result.status_id = 4 # completed

        elif all_samples.shape[0] == 1:
            # Only one sample was returned - this is probably unimodal, so this
            # star needs MCMC
            result.status_id = 2 # needs mcmc

        else:
            # Multiple samples were returned, but not enough to satisfy the
            # number requested in the config file
            result.status_id = 1 # needs more samples

        logger.debug("...done with star {} ({:.2f} seconds)"
                     .format(star.apogee_id, time.time()-t0))

        if count % batch_size == 0:
            session.commit()

        count += 1

    pool.close()

    session.commit()
    session.close()

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

    oc_group = parser.add_mutually_exclusive_group()
    oc_group.add_argument("--overwrite", dest="overwrite", default=False,
                          action="store_true",
                          help="Overwrite any existing results for this "
                               "JokerRun.")
    oc_group.add_argument("--continue", dest="_continue", default=False,
                          action="store_true",
                          help="Continue the the JokerRun.")

    parser.add_argument("-s", "--seed", dest="seed", default=None, type=int,
                        help="Random number seed")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    #
    parser.add_argument("-c", "--config", dest="config_file", required=True,
                        type=str, help="Path to config file that specifies the "
                                       "parameters for this JokerRun.")

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

    main(config_file=args.config_file, pool=pool, seed=args.seed,
         overwrite=args.overwrite, _continue=args._continue)
