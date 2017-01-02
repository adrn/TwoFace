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
from twoface.log import log
from twoface.db import Session, db_connect, table_to_sql_columns, AllStar, AllVisit
from twoface.db.core import Base
from twoface.db.helper import copy_from_table

def main(allVisit_file, allStar_file, credentials_file, **kwargs):
    norm = lambda x: os.path.abspath(os.path.expanduser(x))
    allvisit_tbl = Table.read(norm(allVisit_file), format='fits', hdu=1)
    allstar_tbl = Table.read(norm(allStar_file), format='fits', hdu=1)

    # Columns to skip
    allstar_skip = ['VISITS', 'ALL_VISITS', 'ALL_VISIT_PK', 'VISIT_PK']
    allvisit_skip = []

    # connect to database
    with open(credentials_file, 'r') as f:
        credentials = dict(yaml.load(f))
    credentials.setdefault('dbname', 'apogee')
    engine = db_connect(**credentials)
    log.debug("Connected to database '{}'".format(credentials['dbname']))

    # populate columns of the tables
    for name,col in table_to_sql_columns(allstar_tbl, skip=allstar_skip).items():
        setattr(AllStar, name, col)

    for name,col in table_to_sql_columns(allvisit_tbl, skip=allvisit_skip).items():
        setattr(AllVisit, name, col)

    # this is the magic that creates the tables based on the definitions in twoface/db/model.py
    Base.metadata.drop_all()
    Base.metadata.create_all()

    # Load data using COPY FROM because we are going to add a lot of rows...
    raw_conn = engine.raw_connection()
    _cursor = raw_conn.cursor()

    copy_from_table(_cursor, allstar_tbl, 'allstar', skip=allstar_skip)
    copy_from_table(_cursor, allvisit_tbl, 'allvisit', skip=allvisit_skip)

    _cursor.execute("commit")
    raw_conn.close()

    # Loop through and match up allstar to allvisit (many-to-many)
    session = Session()
    for star in session.query(AllStar).all():
        visits = session.query(AllVisit).filter(AllVisit.apogee_id == star.apogee_id).all()
        star.visits = visits

    session.commit()

    visits = session.query(AllVisit).filter(AllVisit.apogee_id == '2M00000032+5737103').all()
    print(visits[0])
    print(visits[0].stars)

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
