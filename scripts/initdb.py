__author__ = "adrn <adrn@astro.princeton.edu>"

# Standard library
import os

# Third-party
from astropy.table import Table

# Project
import twoface
from twoface.config import conf
from twoface.util import Timer
from twoface.log import log
from twoface.db import Session, db_connect, AllStar, AllVisit, Status
from twoface.db.core import Base

def main(allVisit_file=None, allStar_file=None, test=False, **kwargs):

    # if running in test mode, get test files
    if test:
        base_path = os.path.abspath(os.path.dirname(twoface.__file__))
        allVisit_file = os.path.join(base_path, 'db', 'tests', 'test-allVisit.fits')
        allStar_file = os.path.join(base_path, 'db', 'tests', 'test-allStar.fits')

        # TODO: HACK
        db_path = "/Users/adrian/projects/twoface/cache/test-db.sqlite"

    else:
        raise NotImplementedError()
        assert allVisit_file is not None
        assert allStar_file is not None

        # get credentials from conf
        credentials = dict()
        credentials['host'] = conf['apogee']['host']
        credentials['database'] = conf['apogee']['database']
        credentials['port'] = conf['apogee']['port']
        credentials['user'] = conf['apogee']['user']
        credentials['password'] = conf['apogee']['password']

    norm = lambda x: os.path.abspath(os.path.expanduser(x))
    allvisit_tbl = Table.read(norm(allVisit_file), format='fits', hdu=1)
    allstar_tbl = Table.read(norm(allStar_file), format='fits', hdu=1)

    engine = db_connect(db_path)
    # engine.echo = True
    log.debug("Connected to database at '{}'".format(db_path))

    # this is the magic that creates the tables based on the definitions in twoface/db/model.py
    Base.metadata.drop_all()
    Base.metadata.create_all()

    # load allVisit and allStar files
    # allvisit_tbl

    session = Session()

    log.debug("Loading allStar, allVisit tables...")

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
    log.debug("tables loaded in {:.2f} seconds".format(t.elapsed()))

    # Load the status table
    log.debug("Populating Status table...")
    statuses = list()
    statuses.append(Status(id=0, message='untouched'))
    statuses.append(Status(id=1, message='pending'))
    statuses.append(Status(id=2, message='needs more prior samples'))
    statuses.append(Status(id=3, message='needs mcmc'))
    statuses.append(Status(id=4, message='error'))
    statuses.append(Status(id=5, message='completed'))

    session.add_all(statuses)
    session.commit()
    log.debug("...done")

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
            log.setLevel(logging.DEBUG)
        else: # anything >= 2
            log.setLevel(1)

    elif args.quietness != 0:
        if args.quietness == 1:
            log.setLevel(logging.WARNING)
        else: # anything >= 2
            log.setLevel(logging.ERROR)

    else: # default
        log.setLevel(logging.INFO)

    main(**vars(args))
