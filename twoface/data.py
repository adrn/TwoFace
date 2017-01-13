# Third-party
import astropy.units as u
from astropy.time import Time
import numpy as np
from thejoker.data import RVData

class APOGEERVData(RVData):

    @classmethod
    def from_visits(cls, visits):
        """

        Parameters
        ----------
        visits : list
            List of ``AllVisit`` instances.

        """

        jd = [float(v.jd) for v in visits]
        rv = [float(v.vhelio) for v in visits]
        rv_rand_err = [float(v.vrelerr) for v in visits]
        noise_floor = 100 * u.m/u.s # from APOGEE team

        t = Time(jd, format='jd', scale='utc')
        rv = rv * u.km/u.s
        rv_err = np.sqrt(noise_floor**2 + (rv_rand_err*u.km/u.s)**2)
        return cls(t=t, rv=rv, stddev=rv_err)
