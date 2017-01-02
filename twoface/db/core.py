from __future__ import division, print_function

# Third-party
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base

Session = scoped_session(sessionmaker(autoflush=True, autocommit=False))
Base = declarative_base()

def db_connect(host="localhost", port=5432, user=None, dbname=None, password=None):
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
    Session.configure(bind=engine)
    Base.metadata.bind = engine

    return engine
