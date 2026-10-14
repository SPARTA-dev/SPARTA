"""TimeSeries: observation times paired with values -- spectra or scalars.

A pure container. The values are either ``Spectrum`` objects (a spectroscopic
time series, which ``Observations`` drives through the RV engines: UNICOR
``CCF1d``, TODCOR ``CCF2d``, TIRAVEL) or scalars such as RVs (on which
``PeriodicityDetector`` runs GLS and PDC). After a single-lined RV run,
``Observations`` stores the measured RVs and their uncertainties here, in
``calculated_vrad_list`` and ``errors``, so the periodograms use them.

Conventions: velocities in km/s, times in days.
"""
import numpy as np


def is_numeric_scalar(v):
    '''
    Whether ``v`` is a single numeric value.

    True for any numpy or Python numeric type (float32, int, ...), which marks
    a scalar (e.g. RV) time series rather than a spectrum one. Booleans are
    not numeric here.

    Parameters
    ----------
    v : object
        One value of a time series.

    Returns
    -------
    bool
    '''
    return np.isscalar(v) and np.issubdtype(np.asarray(v).dtype, np.number)


class TimeSeries:
    '''
    Observation times paired with values (spectra or scalars).

    The keyword arguments of ``__init__`` become attributes of the same
    names.
    '''

    # =============================================================================
    # =============================================================================
    def __init__(self, **kwargs):
        '''
        Pair observation times with values; keyword arguments only.

        Parameters
        ----------
        times : list of float
            Observation times (days). Required.
        vals : list
            Observation values -- Spectrum objects or scalars. Required.
        size : int, optional
            Number of data points; derived from ``len(times)`` when omitted.
        errors : list of float, optional
            Uncertainties of scalar values (the values' unit; km/s for RVs).
            Default empty.
        calculated_vrad_list : list of float, optional
            One measured RV (km/s) per epoch of a spectrum series. Default
            empty; Observations fills it after a single-lined RV run.
        fold_period : float, optional
            Period (days) at which ``plot_velocities`` phase-folds the values.
            Default empty (no folding).

        Raises
        ------
        KeyError
            If ``times`` or ``vals`` is missing.
        ValueError
            If ``size`` disagrees with ``len(times)``.
        '''
        self.times = kwargs['times']
        self.vals = kwargs['vals']
        self.size = kwargs.get('size', len(self.times))
        if self.size != len(self.times):
            raise ValueError(f"size={self.size} disagrees with len(times)={len(self.times)}; "
                             "omit size to derive it from times.")

        self.calculated_vrad_list = kwargs.get('calculated_vrad_list', [])
        self.fold_period = kwargs.get('fold_period', [])
        self.errors = kwargs.get('errors', [])

    # =============================================================================
    # =============================================================================
    def plot_velocities(self, figsize=(7, 6)):
        '''
        Plot a scalar time series against time and, if ``fold_period`` is
        set, against orbital phase.

        Black points, with error bars when ``errors`` is set
        (``sparta.plotting.plot_points``).

        Parameters
        ----------
        figsize : (float, float)
            Figure width and height (inches).

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ValueError
            For a spectrum time series.
        '''
        # The plotting stack loads on the first plot, not on import.
        import matplotlib.pyplot as plt
        from sparta import plotting as sp_plot

        if not is_numeric_scalar(self.vals[0]):
            raise ValueError('plot_velocities is available for scalar values time-series only')

        times = np.asarray(self.times, float)
        errors = np.asarray(self.errors, float) if len(self.errors) else None
        n_panels = 2 if self.fold_period else 1
        fig, axs = plt.subplots(n_panels, squeeze=False, figsize=figsize,
                                layout='constrained')
        sp_plot.plot_points(axs[0, 0], times, self.vals, errors)
        axs[0, 0].set_xlabel('Time (days)')
        axs[0, 0].set_ylabel('RV (km/s)')
        if self.fold_period:
            sp_plot.plot_points(axs[1, 0], (times / self.fold_period) % 1.0, self.vals, errors)
            axs[1, 0].set_xlabel(f'Phase (P = {self.fold_period:g} days)')
            axs[1, 0].set_ylabel('RV (km/s)')
        return fig
