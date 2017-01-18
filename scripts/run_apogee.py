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
from thejoker.log import log as joker_logger
from thejoker.sampler import JokerParams, TheJoker, save_prior_samples
from thejoker.utils import quantity_to_hdf5
import yaml

# Project
from twoface.data import APOGEERVData
from twoface.log import log as logger
from twoface.db import Session, db_connect
from twoface.db import JokerRun, AllStar, AllVisit, StarResult, Status, AllVisitToAllStar
from twoface.config import TWOFACE_CACHE_PATH

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
        raise IOError("sqlite database not found at '{}'\n Did you run scripts/initdb.py yet?"
                      .format(db_path))

    logger.debug("Connecting to sqlite database at '{}'".format(db_path))
    engine = db_connect(database_path=db_path, ensure_db_exists=False)
    session = Session()
    logger.debug("...connected!")

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
        run.prior_samples_file = join(TWOFACE_CACHE_PATH, config['prior']['samples_file'])

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
                                      .group_by(AllStar.apogee_id)\
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
    logger.debug("Creating TheJoker instance with {}, {}".format(rnd, pool))
    params = JokerParams(P_min=run.P_min, P_max=run.P_max, jitter=run.jitter,
                         anomaly_tol=1E-11)
    joker = TheJoker(params, random_state=rnd, pool=pool)

    # create prior samples cache, store to file and store filename in DB
    if not os.path.exists(run.prior_samples_file):
        logger.debug("Prior samples file not found - generating now...")

        # TODO: also need to get value of the prior pdf and save
        prior_samples = joker.sample_prior(config['prior']['num_cache'])
        prior_units = save_prior_samples(run.prior_samples_file, prior_samples, u.km/u.s) # data in km/s
        del prior_samples
        # TODO: for large prior samples libraries, may want a way to write
        #   without storing all samples in memory

        logger.debug("...done")

    else:
        with h5py.File(run.prior_samples_file, 'r') as f:
            prior_units = [u.Unit(uu) for uu in f.attrs['units']]

    # build a query to get all stars associated with this JokerRun that need processing
    star_query = session.query(AllStar).join(StarResult, JokerRun, Status)\
                                       .filter(JokerRun.name == run.name)\
                                       .filter(Status.id == 0)

    result_query = session.query(StarResult).join(AllStar, JokerRun)\
                                            .filter(JokerRun.name == run.name)\
                                            .filter(Status.id == 0)

    # create a file to cache the results
    n_stars = star_query.count()
    logger.info("{} stars left to process for run '{}'".format(n_stars, run.name))

    results_filename = join(TWOFACE_CACHE_PATH, "{}.hdf5".format(run.name))

    # TODO: what should structure be? currently thinking /APOGEE_ID/key, e.g.,
    #       /2M00000222+5625359/P for period, etc.
    if not os.path.exists(results_filename):
        with h5py.File(results_filename, 'w') as f:
            pass

    count = 0
    batch_size = 128
    for star in star_query.limit(5000).all(): # TODO: grab a batch of targets to process
        # retrieve result from database
        result = result_query.filter(AllStar.apogee_id == star.apogee_id).one()

        logger.debug("Starting star {}".format(star.apogee_id))
        t0 = time.time()

        data = APOGEERVData.from_visits(star.visits)
        logger.debug("\t visits loaded ({:.2f} seconds)".format(time.time()-t0))

        # adaptive scheme for the rejection sampling:
        n_process = 64 * run.requested_samples_per_star # MAGIC NUMBER
        n_samples = 0 # running total of good samples returned
        start_idx = 0
        all_samples = None
        for n in range(128): # magic number we should never hit
            logger.debug("\t Iteration {}, using {} prior samples".format(n, n_process))

            # process n_process samples from prior cache
            samples = joker._rejection_sample_from_cache(data, n_process,
                                                         run.prior_samples_file, start_idx)

            n_survive = samples.shape[0]
            if n_survive > 1:
                n_samples += n_survive

                if all_samples is None:
                    all_samples = samples
                else:
                    all_samples = np.vstack((all_samples, samples))

            if n_samples >= run.requested_samples_per_star:
                break

            start_idx += n_process
            n_process *= 2

            if n_process > run.max_prior_samples:
                if all_samples is None:
                    all_samples = samples

                logger.debug("\t Hit max prior samples limit!!")
                break

        else: # hit maxiter
            # TODO: error, should never get here
            pass

        logger.debug("\t done sampling ({:.2f} seconds)".format(time.time()-t0))

        # for now, it's sufficient to write the run results to an HDF5 file
        n = run.requested_samples_per_star
        samples_dict = joker.unpack_full_samples(all_samples[:n], data.t_offset, prior_units)

        with h5py.File(results_filename, 'r+') as f:
            if star.apogee_id not in f:
                g = f.create_group(star.apogee_id)
            else:
                g = f[star.apogee_id]

            for key in samples_dict.keys():
                if key in g:
                    del g[key]
                quantity_to_hdf5(g, key, samples_dict[key])

        logger.debug("\t saved samples ({:.2f} seconds)".format(time.time()-t0))

        if all_samples.shape[0] >= run.requested_samples_per_star:
            result.status_id = 5 # completed

        elif all_samples.shape[0] == 1:
            result.status_id = 3 # needs mcmc

        else:
            result.status_id = 2 # needs more samples

        logger.debug("...done with star {} ({:.2f} seconds)".format(star.apogee_id,
                                                                    time.time()-t0))

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
