from __future__ import division, print_function

# Standard library
import os

# Third-party
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

        path = os.path.join(TWOFACE_CACHE_PATH, config['database_file'])
        if not os.path.exists(path):
            cmd = "python scripts/initdb.py --test -v"
            raise IOError("Test database file doest not exist! Before running the tests, you "
                          "should run: \n {}".format(cmd))

        self.engine = db_connect(path)

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

    def teardown(self):
        # delete_db = conf['testing']['delete_test_db']
        pass
        # TODO: teardown should delete the test database!?
