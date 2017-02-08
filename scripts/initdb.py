__author__ = "adrn <adrn@astro.princeton.edu>"

# Standard library
import os
from os.path import abspath, expanduser, join, dirname

# Third-party
from astropy.table import Table
import numpy as np

# Project
import twoface
from twoface.util import Timer
from twoface.log import log as logger
from twoface.db import Session, db_connect, AllStar, AllVisit, Status, RedClump
from twoface.db.core import Base
from twoface.config import TWOFACE_CACHE_PATH

def main(allVisit_file=None, allStar_file=None, rc_file=None, test=False, **kwargs):

    # if running in test mode, get test files
    if test:
        base_path = abspath(dirname(twoface.__file__))
        allVisit_file = os.path.join(base_path, 'db', 'tests', 'test-allVisit.fits')
        allStar_file = os.path.join(base_path, 'db', 'tests', 'test-allStar.fits')
        rc_file = os.path.join(base_path, 'db', 'tests', 'test-rc.fits')

        database_path = join(TWOFACE_CACHE_PATH, 'test.sqlite')

    else:
        assert allVisit_file is not None
        assert allStar_file is not None
        assert rc_file is not None

        database_path = join(TWOFACE_CACHE_PATH, 'apogee.sqlite')

    norm = lambda x: abspath(expanduser(x))
    allvisit_tbl = Table.read(norm(allVisit_file), format='fits', hdu=1)
    allvisit_tbl = allvisit_tbl[np.isfinite(allvisit_tbl['VHELIO'])]

    allstar_tbl = Table.read(norm(allStar_file), format='fits', hdu=1)
    rc_tbl = Table.read(norm(rc_file), format='fits', hdu=1)

    engine = db_connect(database_path)
    # engine.echo = True
    logger.debug("Connected to database at '{}'".format(database_path))

    # this is the magic that creates the tables based on the definitions in twoface/db/model.py
    Base.metadata.drop_all()
    Base.metadata.create_all()

    session = Session()

    logger.debug("Loading allStar, allVisit tables...")

    allstar_colnames = [str(x).split('.')[1].upper() for x in AllStar.__table__.columns]
    i = allstar_colnames.index('ID')
    allstar_colnames.pop(i)

    allvisit_colnames = [str(x).split('.')[1].upper() for x in AllVisit.__table__.columns]
    i = allvisit_colnames.index('ID')
    allvisit_colnames.pop(i)

    rc_colnames = [str(x).split('.')[1].upper() for x in RedClump.__table__.columns]
    for name in ['ID', 'ALLSTAR_ID']:
        i = rc_colnames.index(name)
        rc_colnames.pop(i)

    batch_size = 4000
    stars = []
    all_visits = dict()
    with Timer() as t:
        for i,row in enumerate(allstar_tbl):
            row_data = dict([(k.lower(), row[k]) for k in allstar_colnames])
            star = AllStar(**row_data)
            stars.append(star)

            if star.apogee_id not in all_visits:
                visits = []
                for j,visit_row in enumerate(allvisit_tbl[allvisit_tbl['APOGEE_ID'] == row['APOGEE_ID']]):
                    _data = dict([(k.lower(), visit_row[k]) for k in allvisit_colnames])
                    visits.append(AllVisit(**_data))
                all_visits[star.apogee_id] = visits

            else:
                visits = all_visits[star.apogee_id]

            star.visits = visits

            if i % batch_size == 0 and i > 0:
                session.add_all(stars)
                session.add_all([item for sublist in all_visits.values() for item in sublist])
                session.commit()
                logger.debug("Loaded batch {} ({:.2f} seconds)".format(i*batch_size, t.elapsed()))
                t.reset()

                all_visits = dict()
                stars = []

    if len(stars) > 0:
        session.add_all(stars)
        session.add_all([item for sublist in all_visits.values() for item in sublist])
        session.commit()

    rcstars = []
    with Timer() as t:
        for i,row in enumerate(rc_tbl):
            row_data = dict([(k.lower(), row[k]) for k in rc_colnames])

            rc = RedClump(**row_data)
            try:
                rc.star = session.query(AllStar).filter(AllStar.apstar_id == row_data['apstar_id']).one()
            except:
                continue

            rcstars.append(rc)

            if i % batch_size == 0 and i > 0:
                session.add_all(rcstars)
                session.commit()
                logger.debug("Loaded rc batch {} ({:.2f} seconds)".format(i*batch_size, t.elapsed()))
                t.reset()
                rcstars = []

    if len(rcstars) > 0:
        session.add_all(rcstars)
        session.commit()

    logger.debug("tables loaded in {:.2f} seconds".format(t.elapsed()))

    # Load the status table
    logger.debug("Populating Status table...")
    statuses = list()
    statuses.append(Status(id=0, message='untouched'))
    statuses.append(Status(id=1, message='pending')) # OLD - don't use
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

    parser.add_argument("--allstar", dest="allStar_file", default=None,
                        type=str, help="Path to APOGEE allStar FITS file.")
    parser.add_argument("--allvisit", dest="allVisit_file", default=None,
                        type=str, help="Path to APOGEE allVisit FITS file.")
    parser.add_argument("--redclump", dest="rc_file", default=None,
                        type=str, help="Path to APOGEE Red Clump catalog FITS file.")

    parser.add_argument("--test", dest="test", action="store_true", default=False,
                        help="Setup test database.")

    args = parser.parse_args()

    if not args.test:
        if args.allStar_file is None or args.allVisit_file is None or args.rc_file is None:
            raise ValueError("--allstar, --allvisit, --redclump are required if not running in "
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
