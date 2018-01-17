"""
Given output from the initial run, finish sampling by either further rejection
sampling from the prior cache (needs more samples), or running emcee
(needs mcmc).

How to use
==========

This script must be run *after* run_apogee.py

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
from twoface.log import log as logger
from twoface.db import db_connect
from twoface.db import (JokerRun, AllStar, AllVisit, StarResult, Status,
                        AllVisitToAllStar)
from twoface.config import TWOFACE_CACHE_PATH
from twoface.sample_prior import make_prior_cache


def main(config_file, pool, seed, overwrite=False, _continue=False):
    config_file = abspath(expanduser(config_file))

    # parse config file
    with open(config_file, 'r') as f:
        config = yaml.load(f.read())

    # filename of sqlite database
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

    # Get object for this JokerRun
    try:
        run = session.query(JokerRun).filter(
            JokerRun.name == config['name']).one()

    except NoResultFound:
        raise NoResultFound('No JokerRun "{0}" found in database! Did you '
                            'run the `run_apogee.py` script yet?'
                            .format(config['name']))

    # Set up TheJoker based on hyperparams
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

    # run.prior_samples_file

    # Get all stars in this JokerRun that have statuses 1 or 2 (needs more prior
    # samples, needs mcmc respectively)
    done_subq = session.query(AllStar.apogee_id)\
                       .join(StarResult, JokerRun, Status)\
                       .filter(JokerRun.name == config['name'])\
                       .filter(Status.id.in_([1, 2]) > 0).distinct()

    # Query to get all stars associated with this run that need processing:
    # they should have a status id = 0 (needs processing)
    star_query = session.query(AllStar).join(StarResult, JokerRun, Status)\
                                       .filter(JokerRun.name == run.name)\
                                       .filter(AllStar.apogee_id.in_(done_subq))

    # Base query to get a StarResult for a given Star so we can update the
    # status, etc.
    result_query = session.query(StarResult).join(AllStar, JokerRun)\
                                            .filter(JokerRun.name == run.name)

    # The file with cached posterior samples:
    results_filename = join(TWOFACE_CACHE_PATH, "{0}.hdf5".format(run.name))
    n_stars = star_query.count()
    logger.info("{0} stars left to process for run more samples '{1}'"
                .format(n_stars, run.name))

    if not path.exists(results_filename):
        raise IOError("Posterior samples result file {0} doesn't exist! Are "
                      "you sure you ran `run_apogee.py`?"
                      .format(results_filename))

    # --------------------------------------------------------------------------
    # Here is where we do the actual processing of the data for each star. We
    # loop through all stars that still need processing and either continue with
    # iterative rejection sampling (needs more samples) or fire up standard mcmc
    # (needs mcmc)

    # TODO: left off here


    count = 0 # how many stars we've processed in this star batch
    batch_size = 16 # MAGIC NUMBER: how many stars to process before committing
    for star in star_query.all():

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
            samples_dict, ln_prior = joker.iterative_rejection_sample(
                data, run.requested_samples_per_star, run.prior_samples_file,
                return_logprobs=True)

        except Exception as e:
            logger.warning("\t Failed sampling for star {0} \n Error: {1}"
                           .format(star.apogee_id, str(e)))
            continue

        logger.debug("\t done sampling ({:.2f} seconds)".format(time.time()-t0))

        # For now, it's sufficient to write the run results to an HDF5 file
        n = run.requested_samples_per_star
        all_ln_probs = ln_prior[:n]
        samples_dict = dict([(k, v[:n]) for k,v in samples_dict.items()])
        n_actual_samples = len(all_ln_probs)

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

        if n_actual_samples >= run.requested_samples_per_star:
            result.status_id = 4 # completed

        elif n_actual_samples == 1:
            # Only one sample was returned - this is probably unimodal, so this
            # star needs MCMC
            result.status_id = 2 # needs mcmc

        else:
            # Check whether the samples returned are within one mode. If
            # they are, then "needs mcmc" otherwise "needs more samples"
            P_samples = samples_dict['P'].to(u.day).value
            P_med = np.median(P_samples)
            T = np.ptp(data.t.mjd)
            delta = 4*P_med**2 / (2*np.pi*T)

            if np.std(P_samples) < delta:
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
