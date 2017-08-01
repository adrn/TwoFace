# Third-party
import astropy.units as u
from astropy.time import Time
from thejoker.data import RVData

# Project
from .log import log as logger

def star_to_apogeervdata(star, cao=False):
    """Return a `twoface.data.APOGEERVData` instance for this star.

    Parameters
    ----------
    cao : bool (optional)
        Return the data object using Jason Cao's re-measured velocities
        instead of the APOGEE visit velocities.
    """

    jd = []
    rv = []
    rv_rand_err = []
    for v in star.visits:
        jd.append(float(v.jd))
        rv_rand_err.append(float(v.vrelerr))

        if cao:
            if v.cao_velocity is None:
                logger.warning('Skipping visit {0} because no CaoVelocity '
                               'found!'.format(v))
                continue

            rv.append(v.cao_velocity.vbary + v.cao_velocity.vshift)

        else:
            rv.append(float(v.vhelio))

    t = Time(jd, format='jd', scale='utc')
    rv = rv * u.km/u.s
    rv_err = rv_rand_err*u.km/u.s

    return APOGEERVData(t=t, rv=rv, stddev=rv_err)

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
