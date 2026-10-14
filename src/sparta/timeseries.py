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
import matplotlib.pyplot as plt


def is_numeric_scalar(v):
    '''
    Whether ``v`` is a single numeric value.

    True for any numpy or Python numeric type (float32, int, ...), which marks
    a scalar (e.g. RV) time series rather than a spectrum one. Booleans are
    not numeric here.

    :param v: one value of a time series.
    :return: bool.
    '''
    return np.isscalar(v) and np.issubdtype(np.asarray(v).dtype, np.number)


class TimeSeries:

    # =============================================================================
    # =============================================================================
    def __init__(self, **kwargs):
        '''
        Input: keyword arguments; ``times`` and ``vals`` are required.

              times: float list, observation times (days).
              vals: list, observation values -- Spectrum objects or scalars.
              size: integer, number of data points. Optional: derived from
                    len(times); a ValueError is raised if it disagrees.
              errors: list, uncertainties of scalar values (same unit as the
                      values; km/s for RVs). Default empty.
              calculated_vrad_list: list, one measured RV (km/s) per epoch of a
                      spectrum series. Default empty; Observations fills it
                      after a single-lined RV run.
              fold_period: float, period (days) used by plot_velocities to
                      phase-fold the values. Default empty (no folding).
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
    def plot_velocities(self, figsize=(7,7), sty='ok'):
        '''
        Plot a scalar time series against time and, if ``fold_period`` is
        set, against phase.

        :param figsize: (width, height) of the figure, inches.
        :param sty: matplotlib format string, used when there are no errors.
        :raises ValueError: for a spectrum time series.
        '''
        if not is_numeric_scalar(self.vals[0]):
            raise ValueError('plot_velocities is available for scalar values time-series only')

        fig, axs = plt.subplots(2, figsize=figsize)
        if len(self.errors) == 0:
            axs[0].plot(self.times, self.vals,sty)
        else:
            axs[0].errorbar(self.times, self.vals, self.errors, fmt='o')

        axs[0].set_title("Time VS. Value")
        if self.fold_period:
            times_phased = [t % self.fold_period for t in self.times]
            if len(self.errors) == 0:
                axs[1].plot(times_phased, self.vals, sty)
            else:
                axs[1].errorbar(times_phased, self.vals, self.errors, fmt='o')
            axs[1].set_title("Phase Folded Values")
        else:
            axs[1] = None
        plt.tight_layout()

        plt.show()
