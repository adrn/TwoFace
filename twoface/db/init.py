# Standard library
from os.path import abspath, expanduser, join

# Third-party
from astropy.table import Table
import numpy as np

# Project
from ..config import TWOFACE_CACHE_PATH
from ..util import Timer
from ..log import log as logger
from .connect import db_connect, Base
from .model import (AllStar, AllVisit, AllVisitToAllStar, Status, RedClump,
                    StarResult, CaoVelocity, NessRG)

__all__ = ['initialize_db', 'load_red_clump', 'load_cao', 'load_nessrg']

def tblrow_to_dbrow(tblrow, colnames, varchar_cols=[]):
    row_data = dict()
    for k in colnames:
        if k in varchar_cols:
            row_data[k.lower()] = tblrow[k].strip()
        else:
            row_data[k.lower()] = tblrow[k]
    return row_data

def initialize_db(allVisit_file, allStar_file, database_file,
                  drop_all=False, overwrite=False, batch_size=4096):
    """Initialize the database given FITS filenames for the APOGEE data.

    Parameters
    ----------
    allVisit_file : str
        Full path to APOGEE allVisit file.
    allStar_file : str
        Full path to APOGEE allStar file.
    database_file : str
        Filename (not path) of database file in cache path.
    drop_all : bool (optional)
        Drop all existing tables and re-create the database.
    overwrite : bool (optional)
        Overwrite any data already loaded into the database.
    batch_size : int (optional)
        How many rows to create before committing.
    """

    database_path = join(TWOFACE_CACHE_PATH, database_file)

    norm = lambda x: abspath(expanduser(x))
    allvisit_tbl = Table.read(norm(allVisit_file), format='fits', hdu=1)
    allvisit_tbl = allvisit_tbl[np.isfinite(allvisit_tbl['VHELIO'])]
    allstar_tbl = Table.read(norm(allStar_file), format='fits', hdu=1)

    Session, engine = db_connect(database_path)
    logger.debug("Connected to database at '{}'".format(database_path))

    if drop_all:
        # this is the magic that creates the tables based on the definitions in
        # twoface/db/model.py
        Base.metadata.drop_all()
        Base.metadata.create_all()

    session = Session()

    logger.debug("Loading allStar, allVisit tables...")

    # Figure out what data we need to pull out of the FITS files based on what
    # columns exist in the (empty) database
    allstar_skip = ['ID']
    allstar_colnames = []
    allstar_varchar = []
    for x in AllStar.__table__.columns:
        col = str(x).split('.')[1].upper()
        if col in allstar_skip:
            continue

        if str(x.type) == 'VARCHAR':
            allstar_varchar.append(col)

        allstar_colnames.append(col)

    allvisit_skip = ['ID']
    allvisit_colnames = []
    allvisit_varchar = []
    for x in AllVisit.__table__.columns:
        col = str(x).split('.')[1].upper()
        if col in allvisit_skip:
            continue

        if str(x.type) == 'VARCHAR':
            allvisit_varchar.append(col)

        allvisit_colnames.append(col)

    # What APOGEE IDs are already loaded?
    ap_ids = [x[0] for x in session.query(AllStar.apstar_id).all()]
    logger.debug("{0} stars already loaded".format(len(ap_ids)))

    stars = []
    all_visits = dict()
    with Timer() as t:
        for i,row in enumerate(allstar_tbl): # Load every star
            row_data = tblrow_to_dbrow(row, allstar_colnames, allstar_varchar)

            # If this APOGEE ID is already in the database and we are
            # overwriting data, delete that row
            if row_data['apstar_id'] in ap_ids:
                q = session.query(AllStar).filter(
                    AllStar.apstar_id == row_data['apstar_id'])
                star = q.one()

                if overwrite:
                    visits = session.query(AllVisit).join(AllVisitToAllStar,
                                                          AllStar)\
                        .filter(AllStar.apstar_id == star.apstar_id).all()
                    session.delete(star)
                    for v in visits:
                        session.delete(v)
                    session.commit()

                    star = AllStar(**row_data)
                    stars.append(star)

                    logger.log(1, 'Overwriting star {0} in database'
                                  .format(star))

                else:
                    logger.log(1, 'Loaded star {0} from database'.format(star))

            else:
                star = AllStar(**row_data)
                stars.append(star)
                logger.log(1, 'Adding star {0} to database'.format(star))

            if not star.visits or (star.visits and overwrite):
                if star.apogee_id not in all_visits:
                    visits = []
                    rows = allvisit_tbl[allvisit_tbl['APOGEE_ID']==row['APOGEE_ID']]
                    for j,visit_row in enumerate(rows):
                        _data = tblrow_to_dbrow(visit_row, allvisit_colnames,
                                                allvisit_varchar)
                        visits.append(AllVisit(**_data))
                    all_visits[star.apogee_id] = visits

                else:
                    visits = all_visits[star.apogee_id]

                star.visits = visits

            if i % batch_size == 0 and i > 0:
                session.add_all(stars)
                session.add_all([item for sublist in all_visits.values()
                                 for item in sublist])
                session.commit()
                logger.debug("Loaded batch {} ({:.2f} seconds)"
                             .format(i, t.elapsed()))
                t.reset()

                all_visits = dict()
                stars = []

    if len(stars) > 0:
        session.add_all(stars)
        session.add_all([item for sublist in all_visits.values()
                         for item in sublist])
        session.commit()

    # Load the status table
    if session.query(Status).count() == 0:
        logger.debug("Populating Status table...")
        statuses = list()
        statuses.append(Status(id=0, message='untouched'))
        statuses.append(Status(id=1, message='needs more prior samples'))
        statuses.append(Status(id=2, message='needs mcmc'))
        statuses.append(Status(id=3, message='error'))
        statuses.append(Status(id=4, message='completed'))

        session.add_all(statuses)
        session.commit()
        logger.debug("...done")

    session.close()

def load_red_clump(filename, database_file, overwrite=False, batch_size=4096):
    """Load the red clump catalog stars into the database.

    Information about this catalog can be found here:
    http://www.sdss.org/dr13/data_access/vac/

    Parameters
    ----------
    filename : str
        Full path to APOGEE red clump FITS file.
    database_file : str
        Filename (not path) of database file in cache path.
    overwrite : bool (optional)
        Overwrite any data already loaded into the database.
    batch_size : int (optional)
        How many rows to create before committing.
    """

    database_path = join(TWOFACE_CACHE_PATH, database_file)

    norm = lambda x: abspath(expanduser(x))
    rc_tbl = Table.read(norm(filename), format='fits', hdu=1)

    Session, engine = db_connect(database_path)
    logger.debug("Connected to database at '{}'".format(database_path))
    session = Session()

    # What columns do we load?
    rc_skip = ['ID', 'ALLSTAR_ID']
    rc_colnames = []
    rc_varchar = []
    for x in RedClump.__table__.columns:
        col = str(x).split('.')[1].upper()
        if col in rc_skip:
            continue

        if str(x.type) == 'VARCHAR':
            rc_varchar.append(col)

        rc_colnames.append(col)

    # What APOGEE IDs are already loaded as RC stars?
    rc_ap_ids = session.query(AllStar.apstar_id).join(RedClump).all()
    rc_ap_ids = [x[0] for x in rc_ap_ids]

    rcstars = []
    with Timer() as t:
        for i,row in enumerate(rc_tbl):
            # Only data for columns that exist in the table
            row_data = tblrow_to_dbrow(row, rc_colnames, rc_varchar)

            # Retrieve the parent AllStar record
            try:
                star = session.query(AllStar).filter(
                    AllStar.apstar_id == row['APSTAR_ID']).one()
            except:
                logger.debug('Red clump star not found in AllStar - skipping')
                continue

            if row['APSTAR_ID'] in rc_ap_ids:
                q = session.query(RedClump).join(AllStar).filter(
                    AllStar.apstar_id == row['APSTAR_ID'])

                if overwrite:
                    q.delete()
                    session.commit()

                    rc = RedClump(**row_data)
                    rc.star = star
                    rcstars.append(rc)

                    logger.log(1, 'Overwriting rc {0} in database'
                                  .format(rc))

                else:
                    rc = q.one()
                    logger.log(1, 'Loaded rc {0} from database'.format(rc))

            else:
                rc = RedClump(**row_data)
                rc.star = star
                rcstars.append(rc)
                logger.log(1, 'Adding rc {0} to database'.format(rc))

            if i % batch_size == 0 and i > 0:
                session.add_all(rcstars)
                session.commit()
                logger.debug("Loaded rc batch {} ({:.2f} seconds)"
                             .format(i, t.elapsed()))
                t.reset()
                rcstars = []

    if len(rcstars) > 0:
        session.add_all(rcstars)
        session.commit()

    logger.debug("tables loaded in {:.2f} seconds".format(t.elapsed()))

    session.close()

# ------------------------------------------------------------------------------
def cao_visit_to_visit_id(visit):
    """This is some whack magic shit"""
    visit = visit.replace('apVisit', 'apogee.apo25m.s').replace('-', '.')
    pieces = []
    for piece in visit.split('.'):
        try:
            pieces.append(str(int(piece)))
        except:
            pieces.append(piece)
    return '.'.join(pieces)

def load_cao(filename, database_file, overwrite=False, batch_size=1024):
    """Load Jason Cao's Cannon-measured radial velocities into the database.

    Parameters
    ----------
    filename : str
        Full path to APOGEE red clump FITS file.
    database_file : str
        Filename (not path) of database file in cache path.
    overwrite : bool (optional)
        Overwrite any data already loaded into the database.
    batch_size : int (optional)
        How many rows to create before committing.
    """

    database_path = join(TWOFACE_CACHE_PATH, database_file)

    norm = lambda x: abspath(expanduser(x))
    tbl = Table.read(norm(filename), format='fits', hdu=1)

    Session, engine = db_connect(database_path)
    logger.debug("Connected to database at '{}'".format(database_path))
    session = Session()

    # What columns do we load?
    skip = ['ID', 'ALLVISIT_ID',
            'APOGEEID', 'RA', 'DEC', 'AirMass', 'FIBER', 'SNR']
    colnames = []
    varchar = []
    for x in CaoVelocity.__table__.columns:
        col = str(x).split('.')[1].upper()
        if col in skip:
            continue

        if str(x.type) == 'VARCHAR':
            varchar.append(col)

        colnames.append(col)

    # HACK: Jason uses the name "VISIT" but I use the column name "VISIT_NAME"
    colnames.pop(colnames.index('VISIT_NAME'))
    colnames.append('VISIT')

    # APOGEE ID's for all RC stars
    RC_apogee_ids = [x[0] for x in
                     session.query(AllStar.apogee_id).join(RedClump)
                            .distinct().all()]

    # What visits are already loaded?
    visit_ids = [x[0] for x in
                 session.query(AllVisit.visit_id).join(CaoVelocity).all()]
    logger.debug("{0} CaoVelocity visits already loaded".format(len(visit_ids)))

    cvs = []
    with Timer() as t:
        for i,row in enumerate(tbl):

            if row['APOGEEID'] not in RC_apogee_ids:
                continue

            # Only data for columns that exist in the table
            row_data = tblrow_to_dbrow(row, colnames, varchar)
            row_data['visit_name'] = row_data['visit']
            del row_data['visit']

            # Convert Jason's visit filename to an APOGEE visit_id
            visit_id = cao_visit_to_visit_id(row_data['visit_name'])

            # Retrieve the parent AllVisit record
            try:
                visit = session.query(AllVisit).filter(
                    AllVisit.visit_id == visit_id).one()
            except:
                logger.debug('AllVisit row not found for CaoVelocity '
                             'measurement - skipping')
                continue

            if visit_id in visit_ids: # already in there
                q = session.query(CaoVelocity).join(AllVisit).filter(
                    AllVisit.visit_id == visit_id)

                if overwrite:
                    q.delete()
                    session.commit()

                    cv = CaoVelocity(**row_data)
                    cv.visit = visit
                    cvs.append(cv)

                    logger.log(1, 'Overwriting caovelocity {0} in database'
                                  .format(cv))

                else:
                    cv = q.one()
                    logger.log(1, 'Loaded caovelocity {0} from database'
                               .format(cv))

            else:
                cv = CaoVelocity(**row_data)
                cv.visit = visit
                cvs.append(cv)
                logger.log(1, 'Adding caovelocity {0} to database'.format(cv))

            if i % batch_size == 0 and i > 0:
                session.add_all(cvs)
                session.commit()
                logger.debug("Loaded caovelocity batch {0} ({1:.2f} seconds)"
                             .format(i, t.elapsed()))
                t.reset()
                cvs = []

    if len(cvs) > 0:
        session.add_all(cvs)
        session.commit()

    logger.debug("tables loaded in {:.2f} seconds".format(t.elapsed()))

    session.close()

# ----------------------------------------------------------------------------
# Ness RG masses

def ness_tblrow_to_dbrow(tblrow, colnames):
    row_data = dict()
    for _c in tblrow.colnames:
        c = _c.replace('(','').replace(')','').replace('[','').replace(']','')
        c = c.replace('/', '_')

        if c in colnames:
            row_data[c] = tblrow[_c]

    return row_data

def load_nessrg(filename, database_file, overwrite=False, batch_size=4096):
    """Load the Ness red giant mass catalog.

    Parameters
    ----------
    filename : str
        Full path to Ness red giant info file.
    database_file : str
        Filename (not path) of database file in cache path.
    overwrite : bool (optional)
        Overwrite any data already loaded into the database.
    batch_size : int (optional)
        How many rows to create before committing.
    """

    database_path = join(TWOFACE_CACHE_PATH, database_file)

    norm = lambda x: abspath(expanduser(x))
    tbl = Table.read(norm(filename), format='fits', hdu=1)

    Session, engine = db_connect(database_path)
    logger.debug("Connected to database at '{}'".format(database_path))
    session = Session()

    # What columns do we load?
    skip = ['ID', 'ALLSTAR_ID']
    colnames = []
    for x in NessRG.__table__.columns:
        col = str(x).split('.')[1]
        if col in skip:
            continue
        colnames.append(col)

    # What 2MASS IDs are already loaded?
    ap_ids = [x[0] for x in session.query(AllStar.apogee_id).join(NessRG).all()]

    ness_rows = []
    with Timer() as t:
        for i,row in enumerate(tbl):
            # Only data for columns that exist in the table
            row_data = ness_tblrow_to_dbrow(row, colnames)

            # Retrieve the parent AllStar record
            try:
                star = session.query(AllStar).filter(
                    AllStar.apogee_id == row['2MASS']).one()
            except:
                logger.log(1, 'Star not found in AllStar - skipping')
                continue

            logger.debug('Loading star {0}'.format(row['2MASS']))
            if row['2MASS'] in ap_ids:
                q = session.query(NessRG).join(AllStar).filter(
                    AllStar.apogee_id == row['2MASS'])

                if overwrite:
                    q.delete()
                    session.commit()

                    nrg = NessRG(**row_data)
                    nrg.star = star
                    ness_rows.append(nrg)

                    logger.log(1, 'Overwriting NessRG {0} in database'
                                  .format(nrg.star.apogee_id))

                else:
                    nrg = q.one()
                    logger.log(1, 'Loaded NessRG {0} from database'.format(nrg))

            else:
                nrg = NessRG(**row_data)
                nrg.star = star
                ness_rows.append(nrg)
                logger.log(1, 'Adding NessRG {0} to database'.format(nrg))

            if i % batch_size == 0 and i > 0:
                session.add_all(ness_rows)
                session.commit()
                logger.debug("Loaded batch {} ({:.2f} seconds)"
                             .format(i, t.elapsed()))
                t.reset()
                ness_rows = []

    if len(ness_rows) > 0:
        session.add_all(ness_rows)
        session.commit()

    logger.debug("tables loaded in {:.2f} seconds".format(t.elapsed()))

    session.close()
