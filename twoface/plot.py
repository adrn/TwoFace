# Third-party
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from thejoker.plot import plot_rv_curves
from thejoker.sampler.utils import get_ivar

__all__ = ['plot_data_orbits']

def plot_data_orbits(data, samples, n_orbits=128, jitter=None,
                     xlim_choice='wide', n_times=4096, title=None,
                     ax=None):
    """
    Plot the APOGEE RV data vs. time and orbits computed from The Joker samples.

    Parameters
    ----------
    data : :class:`~twoface.data.APOGEERVData`
        The radial velocity data.
    samples : :class:`~thejoker.samples.JokerSamples`
        Posterior samples from The Joker.
    n_orbits : int, optional
        Number of orbits to plot over the data.
    jitter : :class:`~astropy.units.Quantity`, optional
        The jitter used to do the sampling. Only relevant if the jitter was
        fixed. Used to inflate the error bars for the data.
    xlim_choice : str, optional
        Multiple options for how to set the x-axis limits for the plot.
        ``xlim_choice = 'wide'`` sets the xlim to be twice the longest period
        sample.
        ``xlim_choice = 'tight'`` sets the xlim to be twice the time span of the
        data.

    """

    if jitter is not None:
        data = data.copy()

        ivar = get_ivar(data, jitter.to(data.rv.unit).value)
        data.ivar = ivar / data.rv.unit**2

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12,6))
    else:
        fig = ax.figure

    now = Time.now()

    if xlim_choice == 'tight': # twice the span of the data
        w = np.ptp(data.t.mjd)
        t_grid = np.linspace(data.t.mjd.min() - w*0.05,
                             data.t.mjd.max() + w*1.05,
                             n_times)

    elif xlim_choice == 'wide': # twice the longest period sample
        t_min = data.t.mjd.min()
        t_max = max(data.t.mjd.min() + 2*samples['P'].max().value,
                    data.t.mjd.max())
        span = t_max - t_min
        t_grid = np.linspace(t_min - 0.05*span, t_max + 0.05*span, 2048)

    else:
        raise ValueError('Invalid xlim_choice {0}. Can be "wide" or "tight".'
                         .format(xlim_choice))

    plot_rv_curves(samples, t_grid, rv_unit=u.km/u.s, data=data, ax=ax,
                   n_plot=min(len(samples['P']), n_orbits),
                   plot_kwargs=dict(color='#aaaaaa', alpha=0.25, linewidth=0.5),
                   data_plot_kwargs=dict(zorder=5, elinewidth=1,))

    # Darken the shortest period sample
    dark_style = dict(color='#333333', alpha=0.5, linewidth=1, zorder=10)

    P_min_samples = samples[samples['P'].argmin()]
    plot_rv_curves(P_min_samples, t_grid, rv_unit=u.km/u.s, ax=ax,
                   n_plot=1, plot_kwargs=dark_style)

    # Darken the longest period sample
    P_max_samples = samples[samples['P'].argmax()]
    plot_rv_curves(P_max_samples, t_grid, rv_unit=u.km/u.s, ax=ax,
                   n_plot=1, plot_kwargs=dark_style)

    if now.mjd < t_grid.max():
        ax.axvline(now.mjd, zorder=-10, color='#31a354', alpha=0.4)

    ax.set_xlim(t_grid.min(), t_grid.max())

    _rv = data.rv.to(u.km/u.s).value
    h = np.ptp(_rv)
    ax.set_ylim(_rv.min()-2*h, _rv.max()+2*h)

    if title is not None:
        ax.set_title(title)

    return fig
