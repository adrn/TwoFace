import astropy.units as u
import h5py
import numpy as np
from thejoker.sampler import pack_prior_samples

def make_prior_cache(filename, joker, N, max_batch_size=2**24):
    """

    Parameters
    ----------
    filename : str
        The HDF5 file name to cache to.
    joker : `~thejoker.sampler.TheJoker`
        An instance of ``TheJoker``.
    N : int
        Number of samples to generate in the cache.
    max_batch_size : int (optional)
        The batch size to generate each iteration.

    """

    max_batch_size = int(max_batch_size)
    N = int(N)

    # first just make an empty file
    with h5py.File(filename, 'w') as f:
        pass

    num_added = 0
    for i in range(10000): # HACK: magic number, maximum num. iterations
        samples, ln_probs = joker.sample_prior(max_batch_size, return_logprobs=True)
        packed_samples, units = pack_prior_samples(samples, u.km/u.s) # TODO: make rv_unit configurable?

        batch_size,K = packed_samples.shape

        if (num_added + batch_size) > N:
            packed_samples = packed_samples[:N - (num_added + batch_size)+1]
            batch_size,K = packed_samples.shape
            if batch_size <= 0:
                break

        with h5py.File(filename, 'r+') as f:
            if 'samples' not in f:
                # make the HDF5 file with placeholder datasets
                f.create_dataset('samples', shape=(N, K), dtype=np.float32)
                f.create_dataset('ln_prior_probs', shape=(N,), dtype=np.float32)
                f.attrs['units'] = np.array([str(x) for x in units]).astype('|S6')

            i1 = num_added
            i2 = num_added+batch_size

            f['samples'][i1:i2,:] = packed_samples
            f['ln_prior_probs'][i1:i2] = ln_probs

        num_added += batch_size
