from __future__ import division, print_function

__author__ = "adrn <adrn@astro.princeton.edu>"

# TODO: remove this shite
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# Third-party
from astropy.table import Table
import yaml

# Project
from twoface.util import Timer
from twoface.log import log
from twoface.db import Session, db_connect, AllStar, AllVisit, Status
from twoface.db.core import Base
from twoface.db.helper import copy_from_table

def main(allVisit_file, allStar_file, credentials_file, **kwargs):
    norm = lambda x: os.path.abspath(os.path.expanduser(x))
    allvisit_tbl = Table.read(norm(allVisit_file), format='fits', hdu=1)
    allstar_tbl = Table.read(norm(allStar_file), format='fits', hdu=1)

    # connect to database
    with open(credentials_file, 'r') as f:
        credentials = dict(yaml.load(f))
    credentials.setdefault('dbname', 'apogee')
    engine = db_connect(**credentials)
    log.debug("Connected to database '{}'".format(credentials['dbname']))

    # this is the magic that creates the tables based on the definitions in twoface/db/model.py
    Base.metadata.drop_all()
    Base.metadata.create_all()

    # Load data using COPY FROM because we are going to add a lot of rows...
    raw_conn = engine.raw_connection()
    _cursor = raw_conn.cursor()

    log.debug("Copying allStar file into database...")
    with Timer() as t:
        copy_from_table(_cursor, allstar_tbl, AllStar)
    log.debug("...done after {:.2f} seconds".format(t.time))

    log.debug("Copying allVisit file into database...")
    with Timer() as t:
        copy_from_table(_cursor, allvisit_tbl, AllVisit)
    log.debug("...done after {:.2f} seconds".format(t.time))

    _cursor.execute("commit")
    raw_conn.close()
    log.debug("allStar and allVisit committed to database tables")

    # add status table and modify star-visit mapping
    session = Session()

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
    session.flush()
    log.debug("...done")

    # Loop through and match up allstar to allvisit (many-to-many)
    batch_stride = 1000
    n_batches = len(allstar_tbl) // batch_stride

    log.debug("Matching star-visit...")
    with Timer() as t:
        for i,star in enumerate(session.query(AllStar).all()):
            visits = session.query(AllVisit).filter(AllVisit.apogee_id == star.apogee_id).all()
            star.visits = visits

            if i % batch_stride == 0 and i != 0:
                log.debug("Committing batch {}/{} to database [{:.2f} sec]".format(i//batch_stride,
                                                                                   n_batches,
                                                                                   t.elapsed()))
                session.commit()
                t.reset()

    log.debug("Committing final batch to database [{:.2f} sec]".format(t.elapsed()))
    session.commit()

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

    parser.add_argument("--allstar", dest="allStar_file", required=True,
                        type=str, help="Path to APOGEE allStar FITS file.")
    parser.add_argument("--allvisit", dest="allVisit_file", required=True,
                        type=str, help="Path to APOGEE allVisit FITS file.")
    parser.add_argument("--credentials", dest="credentials_file", required=True,
                        type=str, help="Path to YAML file with database credentials.")

    args = parser.parse_args()

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
