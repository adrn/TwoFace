__author__ = "adrn <adrn@astro.princeton.edu>"

# Standard library
import os
from os.path import abspath, expanduser, join, dirname

# Third-party
from astropy.table import Table

# Project
import twoface
from twoface.util import Timer
from twoface.log import log as logger
from twoface.db import Session, db_connect, AllStar, AllVisit, Status
from twoface.db.core import Base

def main(database_path, allVisit_file=None, allStar_file=None, test=False, **kwargs):

    root_path = dirname(abspath(join(abspath(__file__), '..')))
    cache_path = join(root_path, 'cache')

    if not os.path.exists(cache_path):
        os.makedirs(cache_path, exist_ok=True)

    # if running in test mode, get test files
    if test:
        base_path = abspath(dirname(twoface.__file__))
        allVisit_file = os.path.join(base_path, 'db', 'tests', 'test-allVisit.fits')
        allStar_file = os.path.join(base_path, 'db', 'tests', 'test-allStar.fits')

        if database_path is None:
            database_file = join(cache_path, 'test.sqlite')

    else:
        assert allVisit_file is not None
        assert allStar_file is not None

        if database_file is None:
            database_file = join(cache_path, 'apogee.sqlite')

    norm = lambda x: abspath(expanduser(x))
    allvisit_tbl = Table.read(norm(allVisit_file), format='fits', hdu=1)
    allstar_tbl = Table.read(norm(allStar_file), format='fits', hdu=1)

    engine = db_connect(database_file)
    # engine.echo = True
    logger.debug("Connected to database at '{}'".format(database_file))

    # this is the magic that creates the tables based on the definitions in twoface/db/model.py
    Base.metadata.drop_all()
    Base.metadata.create_all()

    # load allVisit and allStar files
    # allvisit_tbl

    session = Session()

    logger.debug("Loading allStar, allVisit tables...")

    allstar_colnames = [str(x).split('.')[1].upper() for x in AllStar.__table__.columns]
    i = allstar_colnames.index('ID')
    allstar_colnames.pop(i)

    allvisit_colnames = [str(x).split('.')[1].upper() for x in AllVisit.__table__.columns]
    i = allvisit_colnames.index('ID')
    allvisit_colnames.pop(i)

    stars = []
    all_visits = []
    with Timer() as t:
        for i,row in enumerate(allstar_tbl):
            row_data = dict([(k.lower(), row[k]) for k in allstar_colnames])
            star = AllStar(**row_data)
            stars.append(star)

            visits = []
            for j,visit_row in enumerate(allvisit_tbl[allvisit_tbl['APOGEE_ID'] == row['APOGEE_ID']]):
                _data = dict([(k.lower(), visit_row[k]) for k in allvisit_colnames])
                visits.append(AllVisit(**_data))
            star.visits = visits

            all_visits += visits

        session.add_all(stars)
        session.add_all(all_visits)
        session.commit()
    logger.debug("tables loaded in {:.2f} seconds".format(t.elapsed()))

    # Load the status table
    logger.debug("Populating Status table...")
    statuses = list()
    statuses.append(Status(id=0, message='untouched'))
    statuses.append(Status(id=1, message='pending'))
    statuses.append(Status(id=2, message='needs more prior samples'))
    statuses.append(Status(id=3, message='needs mcmc'))
    statuses.append(Status(id=4, message='error'))
    statuses.append(Status(id=5, message='completed'))

    session.add_all(statuses)
    session.commit()
    logger.debug("...done")

    session.close()

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="Initialize the TwoFace project database.")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0, dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0, dest='quietness')
    # parser.add_argument('-o', '--overwrite', action='store_true', dest='overwrite',
    #                     default=False, help='Destroy everything.')

    parser.add_argument("-d", "--dbpath", dest="database_path", default=None,
                        type=str, help="Path to create database file.")

    parser.add_argument("--allstar", dest="allStar_file", default=None,
                        type=str, help="Path to APOGEE allStar FITS file.")
    parser.add_argument("--allvisit", dest="allVisit_file", default=None,
                        type=str, help="Path to APOGEE allVisit FITS file.")

    parser.add_argument("--test", dest="test", action="store_true", default=False,
                        help="Setup test database.")

    args = parser.parse_args()

    if not args.test:
        if args.allStar_file is None or args.allVisit_file is None:
            raise ValueError("--allstar and --allvisit are required if not running in "
                             "test mode (--test)!")

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

    main(**vars(args))
