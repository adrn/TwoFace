from __future__ import division, print_function

# Standard library
import os

# Third-party
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import OperationalError

# Project
from ..log import log

Session = scoped_session(sessionmaker(autoflush=True, autocommit=False))
Base = declarative_base()

def db_connect(host="localhost", port=5432, user=None, dbname=None, password=None,
               ensure_db_exists=True):
    """
    Establish a connection to the Postgres database.
    After running :func:`connect`, sessions can be established.
    >> import starplex
    >> starplex.database.connect(**args)
    >> session = starplex.database.Session()
    Parameters
    ----------
    host : str
        Hostname
    port : int
        Server port
    user : str
        Username for server
    password : str
        Password to log into server
    name : str
        Name of database
    kwargs : dict
        Additional keyword arguments passed to ``sqlalchemy.create_engine``.
    """

    url = "postgresql://{user}:{password}@{host}:{port:d}/{dbname}".format(host=host,
                                                                           port=port,
                                                                           user=user,
                                                                           dbname=dbname,
                                                                           password=password)

    engine = create_engine(url)
    try:
        conn = engine.connect()
        conn.execute("select * from information_schema.tables")
        log.debug("Database '{}' exists".format(dbname))

    except OperationalError:
        if ensure_db_exists:
            log.info("Database '{}' does not exist -- creating for you...".format(dbname))

            # make sure the database exists
            _url = os.path.join(os.path.dirname(str(url)), 'postgres')
            _engine = create_engine(_url)
            _conn = _engine.connect()
            _conn.execute("commit")
            _conn.execute("CREATE DATABASE {0}".format(dbname))
            _conn.close()
            del _engine
        else:
            raise

    Session.configure(bind=engine)
    Base.metadata.bind = engine

    return engine
