from __future__ import division, print_function

__author__ = "adrn <adrn@astro.princeton.edu>"

# TODO: remove this shite
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# Third-party
from astropy.table import Table

# Project
from twoface.db import Session, db_connect, table_to_sql_columns, AllStar, AllVisit
from twoface.db.core import Base
from twoface.db.helper import copy_from_table

def main():
    # allVisit_path = '/Users/adrian/projects/thejoker/data/allVisit-l30e.2.fits'
    # allvisit_tbl = Table.read(allVisit_path, format='fits', hdu=1)

    # allStar_path = '/Users/adrian/projects/thejoker/data/allStar-l30e.2.fits'
    # allstar_tbl = Table.read(allStar_path, format='fits', hdu=1)
    allvisit_tbl = Table.read("/Users/adrian/Downloads/test-allVisit.fits", format='fits', hdu=1)
    allstar_tbl = Table.read("/Users/adrian/Downloads/test-allStar.fits", format='fits', hdu=1)

    # Columns to skip
    allstar_skip = ['VISITS', 'ALL_VISITS', 'ALL_VISIT_PK', 'VISIT_PK']
    allvisit_skip = []

    # TODO: move credentials out
    engine = db_connect(user='adrian', dbname='apogee')

    # populate columns of the tables
    for name,col in table_to_sql_columns(allstar_tbl, skip=allstar_skip).items():
        setattr(AllStar, name, col)

    for name,col in table_to_sql_columns(allvisit_tbl, skip=allvisit_skip).items():
        setattr(AllVisit, name, col)

    # this is the magic that creates the tables based on the definitions in twoface/db/model.py
    Base.metadata.drop_all()
    Base.metadata.create_all()

    # Load data using COPY FROM
    raw_conn = engine.raw_connection()
    _cursor = raw_conn.cursor()

    copy_from_table(_cursor, allstar_tbl, 'allstar', skip=allstar_skip)
    copy_from_table(_cursor, allvisit_tbl, 'allvisit', skip=allvisit_skip)

    _cursor.execute("commit")
    raw_conn.close()

    # loop through and match up allstar to allvisit
    session = Session()
    for star in session.query(AllStar).all():
        print('star', star)
        visits = session.query(AllVisit).filter(AllVisit.apogee_id == star.apogee_id).all()
        star.visits = visits

    session.commit()

    visits = session.query(AllVisit).filter(AllVisit.apogee_id == '2M00000032+5737103').all()
    print(visits[0])
    print(visits[0].stars)

    session.close()

if __name__ == "__main__":
    main()
