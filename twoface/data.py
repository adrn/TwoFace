# Third-party
import astropy.units as u
from astropy.time import Time
import numpy as np
from thejoker.data import RVData

def star_to_apogeervdata(star, clean=False):
    """Return a `twoface.data.APOGEERVData` instance for this star.

    Parameters
    ----------

    """

    jd = []
    rv = []
    rv_rand_err = []
    for v in star.visits:
        rv.append(float(v.vhelio))
        jd.append(float(v.jd))
        rv_rand_err.append(float(v.vrelerr))

    t = Time(jd, format='jd', scale='utc')
    rv = rv * u.km/u.s
    rv_err = rv_rand_err*u.km/u.s

    data = APOGEERVData(t=t, rv=rv, stddev=rv_err)

    if clean:
        bad_mask = (np.isclose(np.abs(data.rv.value), 9999.) &
                    (data.stddev.to(u.km/u.s).value >= 100.))
        data = data[~bad_mask]

    return data

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

        t = Time(jd, format='jd', scale='utc')
        rv = rv * u.km/u.s
        rv_err = rv_rand_err*u.km/u.s
        return cls(t=t, rv=rv, stddev=rv_err)
