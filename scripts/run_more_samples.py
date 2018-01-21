"""
Given output from the initial run, finish sampling by either further rejection
sampling from the prior cache (needs more samples), or running emcee
(needs mcmc).

How to use
==========

This script must be run *after* run_apogee.py

"""

# Standard library
from os import path
import os
import time

# Third-party
import h5py
import numpy as np
from schwimmbad import choose_pool
from thejoker.log import log as joker_logger
from thejoker.sampler import TheJoker
import yaml

# Project
from twoface import unimodal_P
from twoface.log import log as logger
from twoface.db import db_connect, get_run
from twoface.db import JokerRun, AllStar, StarResult, Status
from twoface.config import TWOFACE_CACHE_PATH
from twoface.sample_prior import make_prior_cache


def main(config_file, pool, seed, overwrite=False):
    config_file = path.abspath(path.expanduser(config_file))

    # parse config file
    with open(config_file, 'r') as f:
        config = yaml.load(f.read())

    # filename of sqlite database
    database_file = config['database_file']

    db_path = path.join(TWOFACE_CACHE_PATH, database_file)
    if not os.path.exists(db_path):
        raise IOError("sqlite database not found at '{0}'\n Did you run "
                      "scripts/initdb.py yet for that database?"
                      .format(db_path))

    logger.debug("Connecting to sqlite database at '{0}'".format(db_path))
    Session, engine = db_connect(database_path=db_path,
                                 ensure_db_exists=False)
    session = Session()

    run = get_run(config, session, overwrite=False)

    # The file with cached posterior samples:
    results_filename = path.join(TWOFACE_CACHE_PATH,
                                 "{0}.hdf5".format(run.name))
    if not path.exists(results_filename):
        raise IOError("Posterior samples result file {0} doesn't exist! Are "
                      "you sure you ran `run_apogee.py`?"
                      .format(results_filename))

    # Create TheJoker sampler instance with the specified random seed and pool
    rnd = np.random.RandomState(seed=seed)
    logger.debug("Creating TheJoker instance with {0}, {1}".format(rnd, pool))
    params = run.get_joker_params()
    joker = TheJoker(params, random_state=rnd, pool=pool,
                     n_batches=8 * pool.size) # HACK: magic number

    # TODO: a temporary hack for the end-of-2017 apogee-jitter run: we need to
    # make sure a 2nd prior cache exists with even more samples!
    _path, ext = path.splitext(run.prior_samples_file)
    new_path = '{0}_moar{1}'.format(_path, ext)
    if not path.exists(new_path):
        make_prior_cache(new_path, joker,
                         N=8 * config['prior']['num_cache'], # OMG: ~100 GB
                         max_batch_size=2**24) # MAGIC NUMBER

    # Get all stars in this JokerRun that need more prior samples
    # TODO HACK: because I suck, some got mis-labeled "needs mcmc" - so re-run
    # on all of those too:
    star_query = session.query(AllStar).join(StarResult, JokerRun, Status)\
                                       .filter(JokerRun.name == run.name)\
                                       .filter(Status.id.in_([1, 2]))
    #                                    .filter(Status.id == 1)

    # Base query to get a StarResult for a given Star so we can update the
    # status, etc.
    result_query = session.query(StarResult).join(AllStar, JokerRun)\
                                            .filter(JokerRun.name == run.name)

    n_stars = star_query.count()
    logger.info("{0} stars left to process for run more samples '{1}'"
                .format(n_stars, run.name))

    # --------------------------------------------------------------------------
    # Here is where we do the actual processing of the data for each star. We
    # loop through all stars that still need processing and continue with
    # rejection sampling.

    count = 0 # how many stars we've processed in this star batch
    batch_size = 16 # MAGIC NUMBER: how many stars to process before committing
    for star in star_query.limit(1).all(): # HACK: REMOVE

        if result_query.filter(AllStar.apogee_id == star.apogee_id).count() < 1:
            logger.debug('Star {0} has no result object!'
                         .format(star.apogee_id))
            continue

        # Retrieve existing StarResult from database. We limit(1) because the
        # APOGEE_ID isn't unique, but we attach all visits for a given star to
        # all rows, so grabbing one of them is fine.
        result = result_query.filter(AllStar.apogee_id == star.apogee_id)\
                             .limit(1).one()

        logger.log(1, "Starting star {0}".format(star.apogee_id))
        logger.log(1, "Current status: {0}".format(str(result.status)))
        t0 = time.time()

        data = star.apogeervdata()
        logger.log(1, "\t visits loaded ({:.2f} seconds)"
                   .format(time.time()-t0))
        try:
            samples, ln_prior = joker.iterative_rejection_sample(
                data=data, n_requested_samples=run.requested_samples_per_star,
                # HACK: prior_cache_file=run.prior_samples_file,
                prior_cache_file=new_path, return_logprobs=True)

        except Exception as e:
            logger.warning("\t Failed sampling for star {0} \n Error: {1}"
                           .format(star.apogee_id, str(e)))
            continue

        logger.debug("\t done sampling ({:.2f} seconds)".format(time.time()-t0))

        # For now, it's sufficient to write the run results to an HDF5 file
        n = run.requested_samples_per_star
        all_ln_probs = ln_prior[:n]
        samples = samples[:n]
        n_actual_samples = len(all_ln_probs)

        # Write the samples that pass to the results file
        with h5py.File(results_filename, 'r+') as f:
            if star.apogee_id in f:
                del f[star.apogee_id]

            g = f.create_group(star.apogee_id)

            samples.to_hdf5(g)

            if 'ln_prior_probs' in g:
                del g['ln_prior_probs']
            g.create_dataset('ln_prior_probs', data=all_ln_probs)

        logger.debug("\t saved samples ({:.2f} seconds)".format(time.time()-t0))

        if n_actual_samples >= run.requested_samples_per_star:
            result.status_id = 4 # completed

        elif n_actual_samples == 1:
            # Only one sample was returned - this is probably unimodal, so this
            # star needs MCMC
            result.status_id = 2 # needs mcmc

        else:

            if unimodal_P(samples, data):
                # Multiple samples were returned, but they look unimodal
                result.status_id = 2 # needs mcmc

            else:
                # Multiple samples were returned, but not enough to satisfy the
                # number requested in the config file
                result.status_id = 1 # needs more samples

        logger.debug("...done with star {} ({:.2f} seconds)"
                     .format(star.apogee_id, time.time()-t0))

        if count % batch_size == 0 and count > 0:
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

    parser.add_argument("-s", "--seed", dest="seed", default=None, type=int,
                        help="Random number seed")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

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

    main(config_file=args.config_file, pool=pool, seed=args.seed)
