# Standard library
from os.path import abspath, expanduser
import os
import sys

# Third-party
import astropy.units as u
import h5py
import numpy as np
from schwimmbad import choose_pool
from sqlalchemy import func
from thejoker.data import RVData
from thejoker.sampler import JokerParams, TheJoker, save_prior_samples
import yaml

# Project
from twoface.data import APOGEERVData
from twoface.log import log as logger
from twoface.db import Session, db_connect
from twoface.db import JokerRun, AllStar, AllVisit, StarResult, Status, AllVisitToAllStar
from twoface.config import conf

def main(config_file, pool, seed, overwrite=False, _continue=False):
    config_file = abspath(expanduser(config_file))

    # parse config file
    with open(config_file, 'r') as f:
        config = yaml.load(f.read())

    # look for database credentials in ~/.config/twoface/twoface.cfg - read the [apogee] section
    credentials = dict()
    credentials['host'] = conf['apogee']['host']
    credentials['database'] = conf['apogee']['database']
    credentials['port'] = conf['apogee']['port']
    credentials['user'] = conf['apogee']['user']
    credentials['password'] = conf['apogee']['password']
    logger.debug("Connecting to postgres database '{host}:{port}/{database}' with "
                 "user '{user}'...".format(**credentials))

    engine = db_connect(ensure_db_exists=False, **credentials)
    session = Session()
    logger.debug("...connected!")

    # HACK: for testing
    session.query(StarResult).delete()
    session.query(JokerRun).delete()
    session.commit()

    # see if this run (by name) is in db already, if so, just grab
    _run = session.query(JokerRun).filter(JokerRun.name == config['name']).all()
    if len(_run) == 0:
        logger.info("JokerRun '{}' not found in database - creating new entry..."
                    .format(config['name']))

        # create a JokerRun for this run
        run = JokerRun()
        run.config_file = config_file
        run.name = config['name']
        run.P_min = u.Quantity(*config['hyperparams']['P_min'].split())
        run.P_max = u.Quantity(*config['hyperparams']['P_max'].split())
        run.requested_samples_per_star = int(config['hyperparams']['requested_samples_per_star'])
        run.max_prior_samples = int(config['prior']['max_samples'])
        run.prior_samples_file = abspath(expanduser(config['prior']['cache_path']))

        if 'jitter' in config['hyperparams']:
            run.jitter = u.Quantity(*config['hyperparams']['jitter'].split())

        elif 'jitter_prior_mean' in config['hyperparams']:
            raise NotImplementedError("NOT YET SORRY")

        else:
            # no jitter specified, set to 0
            run.jitter = 0 * u.m/u.s

        # TODO: need a way to pass in velocity trend info

        # TODO: get all stars we're going to process - some filter on number of
        #   observations based on the number of linear parameters
        # TODO: need some way of specifying a constraint on the stars that are to be processed
        stars = session.query(AllStar).join(AllVisitToAllStar, AllVisit)\
                                      .group_by(AllStar)\
                                      .distinct(AllStar.apogee_id)\
                                      .having(func.count(AllVisit.id) >= 3)\
                                      .all()
        run.stars = stars
        session.add(run)
        session.commit()

    elif len(_run) == 1:
        logger.info("JokerRun '{}' already found in database - retrieving".format(config['name']))
        run = _run[0]

    else:
        raise ValueError("Multiple JokerRun rows found for name '{}'".format(config['name']))

    # create TheJoker sampler instance
    rnd = np.random.RandomState(seed=seed)
    params = JokerParams(P_min=run.P_min, P_max=run.P_max, jitter=run.jitter,
                         anomaly_tol=1E-11)
    joker = TheJoker(params, random_state=rnd) # TODO: pool?

    # create prior samples cache, store to file and store filename in DB
    if not os.path.exists(run.prior_samples_file):
        logger.debug("Prior samples file not found - generating now...")
        base_path = os.path.dirname(run.prior_samples_file)
        os.makedirs(base_path, exist_ok=True)

        prior_samples = joker.sample_prior(config['prior']['num_cache'])
        save_prior_samples(run.prior_samples_file, prior_samples, u.km/u.s) # all data in km/s
        del prior_samples
        # TODO: for large prior samples libraries, may want a way to write
        #   without storing all samples in memory

        logger.debug("...done")

    # TODO: grab a batch of targets to process
    star = session.query(AllStar).join(StarResult, JokerRun, Status)\
                  .filter(JokerRun.name == run.name)\
                  .filter(Status.id == 0)\
                  .limit(1)\
                  .one() # get a single untouched star

    print(star.visits)
    data = APOGEERVData.from_visits(star.visits)

    import matplotlib.pyplot as plt
    data.plot()
    plt.show()

    return

    # adaptive scheme for the rejection sampling:
    n_process = 64 * run.requested_samples_per_star
    n_samples = 0 # running total of good samples returned
    for n in range(128): # magic number we should never hit
        # grab n_process samples from prior cache

        # process, see how many remain

        n_survive = ...

        if n_survive > 1:
            n_samples += n_survive
            # save info on the samples that survived

        if n_samples >= run.requested_samples_per_star:
            break

        n_process *= 4

        if n_process > run.max_prior_samples:
            if n_process == run.max_prior_samples:
                break

            else:
                n_process = run.max_prior_samples

    else: # hit maxiter
        # TODO: error, should never get here
        pass

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0, dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0, dest='quietness')

    oc_group = parser.add_mutually_exclusive_group()
    oc_group.add_argument("--overwrite", dest="overwrite", default=False,
                          action="store_true", help="Overwrite any existing results for "
                                                    "this JokerRun.")
    oc_group.add_argument("--continue", dest="_continue", default=False,
                          action="store_true", help="Continue the the JokerRun.")

    parser.add_argument("-s", "--seed", dest="seed", default=None, type=int,
                        help="Random number seed")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    #
    parser.add_argument("-c", "--config", dest="config_file", required=True,
                        type=str, help="Path to config file that specifies the parameters for "
                                       "this JokerRun.")

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

    pool_kwargs = dict(mpi=args.mpi, processes=args.n_procs)
    pool = choose_pool(**pool_kwargs)

    main(config_file=args.config_file, pool=pool, seed=args.seed,
         overwrite=args.overwrite, _continue=args._continue)
