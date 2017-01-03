from __future__ import division, print_function

# Third-party
from sqlalchemy import Table, Column, types
from sqlalchemy.schema import ForeignKey
from sqlalchemy.orm import relationship

# Project
from .core import Base

__all__ = ['AllStar', 'AllVisit', 'JokerState', 'StarStatus']

join_table = Table('allvisit_to_allstar',
                   Base.metadata,
                   Column('allstar_id', types.Integer, ForeignKey('allstar.id')),
                   Column('allvisit_id', types.Integer, ForeignKey('allvisit.id')))

class AllStar(Base):
    __tablename__ = 'allstar'

    id = Column(types.Integer, primary_key=True)
    visits = relationship("AllVisit", secondary=join_table, back_populates="stars")
    # Note: columns must be auto-populated based on the FITS table columns

    jokerstate_id = Column('jokerstate_id', types.Integer, ForeignKey('jokerstate.id'))
    jokerstate = relationship("JokerState")

    def __repr__(self):
        msg = self.jokerstate.status.message
        return "<ApogeeStar(id='{}', apogee_id='{}', status='{}')>".format(self.id,
                                                                           self.apogee_id,
                                                                           msg)

class AllVisit(Base):
    __tablename__ = 'allvisit'

    id = Column(types.Integer, primary_key=True)
    stars = relationship("AllStar", secondary=join_table, back_populates="visits")
    # Note: columns must be auto-populated based on the FITS table columns

    def __repr__(self):
        return "<ApogeeVisit(APOGEE_ID='{}', MJD='{}')>".format(self.apogee_id, self.mjd)

class JokerState(Base):
    __tablename__ = 'jokerstate'

    id = Column(types.Integer, primary_key=True)

    status_id = Column('starstatus_id', types.Integer, ForeignKey('starstatus.id'))
    status = relationship("StarStatus")
    notes = Column('notes', types.String)

    def __repr__(self):
        return "<JokerState(status='{}')>".format(self.status.message)

class StarStatus(Base):
    __tablename__ = 'starstatus'

    id = Column('id', types.Integer, primary_key=True, autoincrement=False)
    message = Column('message', types.String)

"""
TODO: status needs to be something like

    0 - untouched
    1 - pending
    2 - needs more prior samples
    3 - needs mcmc
    4 - error
    5 - done

"""
