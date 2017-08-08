""" Make diagnostic plots for a specified APOGEE ID """

# Standard library
from os import path

# Third-party
import h5py
import matplotlib.pyplot as plt
from sqlalchemy.orm.exc import NoResultFound

# Project
from twoface.log import log as logger
from twoface.db import db_connect
from twoface.db import (JokerRun, AllStar, AllVisit, StarResult, Status,
                        AllVisitToAllStar, RedClump, CaoVelocity)
from twoface.config import TWOFACE_CACHE_PATH
from twoface.io import load_samples
from twoface.plot import plot_data_orbits

def main(database_file, apogee_id, joker_run, cao):

    db_path = path.join(TWOFACE_CACHE_PATH, database_file)
    if not path.exists(db_path):
        raise IOError("sqlite database not found at '{0}'\n Did you run "
                      "scripts/initdb.py yet for that database?"
                      .format(db_path))

    logger.debug("Connecting to sqlite database at '{0}'".format(db_path))
    Session, engine = db_connect(database_path=db_path,
                                 ensure_db_exists=False)
    session = Session()

    # Get The Joker run information
    run = session.query(JokerRun).filter(JokerRun.name == joker_run).one()

    try:
        star = session.query(AllStar).join(StarResult, JokerRun)\
                      .filter(AllStar.apogee_id == apogee_id)\
                      .filter(JokerRun.name == joker_run)\
                      .one()

    except NoResultFound:
        raise NoResultFound("Star {0} has no results in Joker run {1}."
                            .format(apogee_id, joker_run))

    # get the RV data for this star
    data = star.apogeervdata(cao=cao)

    # load posterior samples from The Joker
    samples_dict = load_samples(path.join(TWOFACE_CACHE_PATH,
                                          '{0}.hdf5'.format(run.name)),
                                apogee_id)

    # Plot the data with orbits on top
    fig = plot_data_orbits(data, samples_dict, jitter=run.jitter,
                           xlim_choice='wide', title=star.apogee_id)
    fig.set_tight_layout(True)

    fig = plot_data_orbits(data, samples_dict, jitter=run.jitter,
                           xlim_choice='tight', title=star.apogee_id)
    fig.set_tight_layout(True)

    # TODO:

    session.close()

    plt.show()

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

    # Required:
    parser.add_argument("-a", "--apogeeid", dest="apogee_id",
                        required=True, type=str,
                        help="The APOGEE ID to visualize.")

    parser.add_argument("-j", "--jokerrun", dest="joker_run",
                        required=True, type=str,
                        help="The Joker run name to load results from.")

    # Optional:
    parser.add_argument("-d", "--dbfile", dest="database_file",
                        default="apogee.sqlite", type=str,
                        help="Path to the database file.")

    parser.add_argument("--cao", dest="cao_velocities", default=False,
                        action="store_true",
                        help="Plot the Cao velocities instead of APOGEE "
                             "radial velocities.")

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

    main(apogee_id=args.apogee_id, database_file=args.database_file,
         joker_run=args.joker_run, cao=args.cao_velocities)