"""
Check what happens if we rejection sample through the entire prior cache for a
star that got flagged "needs more prior samples"
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


def main(pool):

    db_path = join(TWOFACE_CACHE_PATH, 'apogee.sqlite')
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
            JokerRun.name == 'apogee-jitter').one()

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

    star = session.query(AllStar).filter(AllStar.apogee_id == '2M04171719+4724006').limit(1).one()
    data = star.apogeervdata()
    samples = joker.rejection_sample(data,
                                     prior_cache_file=run.prior_samples_file)

    with h5py.File('dumb_test.hdf5', 'w') as f:
        samples.to_hdf5(f)

    pool.close()
    session.close()
    sys.exit(0)


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
