# Standard library
import os

# Third-party
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
import pytest
import yaml

# Project
from ...config import TWOFACE_CACHE_PATH
from ..core import db_connect, Session
from ..model import AllStar, AllVisit, StarResult, Status, JokerRun

class TestDB(object):

    def setup(self):
        # connect to database
        with open(get_pkg_data_filename('travis_db.yml')) as f:
            config = yaml.load(f)

        self.db_path = os.path.join(TWOFACE_CACHE_PATH, config['database_file'])
        if not os.path.exists(self.db_path):
            cmd = "python scripts/initdb.py --test -v"
            raise IOError("Test database file doest not exist! Before running the tests, you "
                          "should run: \n {}".format(cmd))

        self.engine = db_connect(self.db_path)

    def test_one(self):
        # a target in both test FITS files included in the repo
        test_target_ID = "4264.2M00000032+5737103"

        session = Session()

        # get star entry and check total num of visits
        star = session.query(AllStar).filter(AllStar.target_id == test_target_ID).one()
        assert len(star.visits) == 6

        # get a visit and check that it has one star
        visit = session.query(AllVisit).filter(AllVisit.target_id == test_target_ID).limit(1).one()
        assert len(visit.stars) == 2

        session.close()

    def test_jokerrun_cascade(self):
        # make sure the Results and Statuses are deleted when a JokerRun is deleted

        NAME = 'test-cascade'
        session = Session()

        # first set up:
        stars = session.query(AllStar).all()

        run = JokerRun()
        run.config_file = ''
        run.name = NAME
        run.P_min = 1.*u.day
        run.P_max = 100.*u.day
        run.requested_samples_per_star = 128
        run.max_prior_samples = 1024
        run.prior_samples_file = ''
        run.stars = stars
        session.add(run)
        session.commit()

        assert session.query(JokerRun).count() == 1
        assert session.query(AllStar).count() == session.query(StarResult).count()

        for run in session.query(JokerRun).filter(JokerRun.name == NAME).all():
            session.delete(run)
        session.commit()

        assert session.query(JokerRun).count() == 0
        assert session.query(StarResult).count() == 0
        assert session.query(Status).count() > 0 # just to be safe

        session.close()

    def teardown(self):
        # TODO: should I delete the test database?
        self.db_path
