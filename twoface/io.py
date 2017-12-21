# Third-party
import h5py
from twobody.celestial import VelocityTrend1
from thejoker.sampler import JokerSamples
from thejoker.utils import quantity_from_hdf5

__all__ = ['load_samples']

def load_samples(group_or_filename, apogee_id=None):
    """Load posterior samples from The Joker into a dictionary.

    Parameters
    ----------
    group_or_filename : :class:`h5py.Group` or str
    apogee_id : str, optional
        If a filename is passed to ``group_or_filename``, you must also specify
        the APOGEE ID of the source to load.
    """
    # HACK: so far, we've only used VelocityTrend1, so assume that
    trend_cls = VelocityTrend1

    if isinstance(group_or_filename, str):
        if apogee_id is None:
            raise ValueError("If a filename is passed, you must also specify "
                             "the APOGEE ID of the source to load.")

        f = h5py.File(group_or_filename)
        group = f[apogee_id]

    else:
        f = None
        group = group_or_filename

    samples_dict = dict()
    for k in group.keys():
        if k == 'ln_prior_probs': # skip
            continue

        samples_dict[k] = quantity_from_hdf5(group, k)

    if f is not None:
        f.close()

    return JokerSamples(trend_cls=trend_cls, **samples_dict)
