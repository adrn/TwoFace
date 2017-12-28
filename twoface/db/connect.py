# Standard library
import os
import warnings

# Third-party
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine import Engine
from sqlalchemy import event
from sqlalchemy.exc import SAWarning

Base = declarative_base()

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

    warnings.filterwarnings('ignore',
        r"^Dialect sqlite\+pysqlite does \*not\* support Decimal objects natively\, "
        "and SQLAlchemy must convert from floating point - rounding errors and other "
        "issues may occur\. Please consider storing Decimal numbers as strings or "
        "integers on this platform for lossless storage\.$",
        SAWarning, r'^sqlalchemy\.sql\.type_api$')

def db_connect(database_path, ensure_db_exists=False):
    """
    Connect to the specified database.

    Parameters
    ----------
    database : str (optional)
        Name of database
    ensure_db_exists : bool (optional)
        Ensure the database ``database`` exists.

    Returns
    -------
    engine :
        The sqlalchemy database engine.
    """

    engine = create_engine("sqlite:///{}"
                           .format(os.path.abspath(database_path)))
    Session = scoped_session(sessionmaker(bind=engine, autoflush=True,
                                          autocommit=False))
    Base.metadata.bind = engine

    if ensure_db_exists:
        Base.metadata.create_all(engine)

    return Session, engine
