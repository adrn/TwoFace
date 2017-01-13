from __future__ import division, print_function

# Standard library
import os

# Third-party
from astropy.utils.data import get_pkg_data_filename
import pytest

# Project
from ...config import conf
from ..core import db_connect, Session
from ..model import AllStar, AllVisit, StarResult, Status, JokerRun

if 'CI' in os.environ and os.environ['CI']:
    # when in continuous integration, only run database tests on Travis
    if 'TRAVIS' in os.environ and os.environ['TRAVIS']:
        conf.read(get_pkg_data_filename("travis_db.cfg"))
        skip_db_tests = conf['testing']['skip_db_tests']

    else:
        skip_db_tests = True

else:
    skip_db_tests = conf['testing']['skip_db_tests']

class TestDB(object):

    def setup(self):
        credentials = dict()
        credentials['host'] = conf['testing']['host']
        credentials['database'] = conf['testing']['database']
        credentials['port'] = conf['testing']['port']
        credentials['user'] = conf['testing']['user']
        credentials['password'] = conf['testing']['password']

        # connect to database
        self.engine = db_connect(**credentials)

    @pytest.mark.skipif(skip_db_tests, reason="skipping database tests")
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
