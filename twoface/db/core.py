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

def db_connect(host="localhost", port=5432, user=None, database=None, password=None,
               ensure_db_exists=True):
    """
    Connect to the specified postgres database.

    Parameters
    ----------
    host : str (optional)
        Hostname.
    port : int (optional)
        Server port.
    user : str (optional)
        Username.
    password : str (optional)
        Password.
    database : str (optional)
        Name of database
    ensure_db_exists : bool (optional)
        Ensure the database ``database`` exists.

    Returns
    -------
    engine :
        The sqlalchemy database engine.
    """

    url = "postgresql://{user}:{password}@{host}:{port}/{database}".format(host=host,
                                                                           port=port,
                                                                           user=user,
                                                                           database=database,
                                                                           password=password)

    engine = create_engine(url)
    try:
        conn = engine.connect()
        conn.execute("select * from information_schema.tables")
        log.debug("Database '{}' exists".format(database))

    except OperationalError:
        if ensure_db_exists:
            log.info("Database '{}' does not exist -- creating for you...".format(database))

            # make sure the database exists
            _url = os.path.join(os.path.dirname(str(url)), 'postgres')
            _engine = create_engine(_url)
            _conn = _engine.connect()
            _conn.execute("commit")
            _conn.execute("CREATE DATABASE {0}".format(database))
            _conn.close()
            del _engine
        else:
            raise

    Session.configure(bind=engine)
    Base.metadata.bind = engine

    return engine
