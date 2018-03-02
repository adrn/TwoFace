# Third-party
import astropy.units as u
import numpy as np

__all__ = ['unimodal_P', 'max_likelihood_sample']


def unimodal_P(samples, data):
    """Check whether the samples returned are within one period mode.

    Parameters
    ----------
    samples : `~thejoker.JokerSamples`

    Returns
    -------
    is_unimodal : bool
    """

    P_samples = samples['P'].to(u.day).value
    P_min = np.min(P_samples)
    T = np.ptp(data.t.mjd)
    delta = 4*P_min**2 / (2*np.pi*T)

    return np.std(P_samples) < delta


def max_likelihood_sample(data, samples):
    """Return the maximum-likelihood sample.

    Parameters
    ----------
    data : `~thejoker.RVData`
    samples : `~thejoker.JokerSamples`

    """
    chisqs = np.zeros(len(samples))

    for i in range(len(samples)):
        orbit = samples.get_orbit(i)
        residual = data.rv - orbit.radial_velocity(data.t)
        err = np.sqrt(data.stddev**2 + samples['jitter'][i]**2)
        chisqs[i] = np.sum((residual**2 / err**2).decompose())

    return samples[np.argmin(chisqs)]
