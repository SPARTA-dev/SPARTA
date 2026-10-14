"""PeriodicityDetector: periodograms of a TimeSeries.

GLS (astropy ``LombScargle``) and the phase-distance-correlation periodogram
(PDC; biased, unbiased, and error-aware) on scalar series such as RVs;
USuRPER and the shift/shape partial periodograms on single-order spectra.
The detector holds the series, the frequency grid and the results; the PDC
math -- distance matrices, centering, the frequency scan -- is the pure array
functions of ``sparta.pdc``. Nothing here writes to the time series.

Conventions: times in days, frequency grid in 1/day (``freq_range`` and
``periodogram_grid_resolution`` points per 1/day).

References: PDC -- Zucker 2018, MNRAS 474, L86; USuRPER -- Binnenfeld, Shahaf
& Zucker 2020, A&A 642, A146; partial periodograms -- Binnenfeld et al. 2022,
A&A 659, A189; PDC p-values -- Shen et al. 2019 (arXiv:1912.12150).
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.stats.distributions import chi2
from unicor import dsp
from sparta import pdc
from sparta.timeseries import is_numeric_scalar


class PeriodicityDetector:

    # =============================================================================
    # =============================================================================
    def __init__(self, **kwargs):
        '''
        Input: keyword arguments.

              time_series: TimeSeries object, the time series for periodicity detection.

              freq_range: (f_min, f_max) tuple, frequency grid boundaries in
                          1/day (required); f_min must be > 0, since the phase
                          distance takes dt mod (1/f).
              periodogram_grid_resolution: float, grid points per 1/day
                          (required); the grid step is 1/resolution in 1/day.

        The grid is ``self.frequency_grid`` (1/day). Every periodogram is
        stored under its name in ``self.results_frequency`` and
        ``self.results_power``: 'GLS'; 'PDC_unbiased' / 'PDC_biased';
        'PDC_errors_unbiased' / 'PDC_errors_biased' (error-aware PDC);
        'USURPER_unbiased' / 'USURPER_biased'; 'partial_shift' /
        'partial_shape'. ``self.period_truth`` (a list of periods,
        days; default None) marks known periods in ``plot_periodograms``.

        Raises TypeError if freq_range or periodogram_grid_resolution is missing,
        and ValueError if they do not define a non-empty grid of positive
        frequencies.
        '''
        if 'time_series' in kwargs:
            self.time_series = kwargs['time_series']
        if not all(k in kwargs for k in ('freq_range', 'periodogram_grid_resolution')):
            raise TypeError("PeriodicityDetector requires both freq_range=(f_min, f_max), "
                            "in 1/day, and periodogram_grid_resolution (grid points per 1/day).")
        self.periodogram_grid_resolution = kwargs['periodogram_grid_resolution']
        self.freq_range = kwargs['freq_range']
        if self.freq_range[0] <= 0:
            raise ValueError(f"freq_range={self.freq_range}: frequencies are in 1/day and must be "
                             "> 0 (the phase distance divides by f).")
        if self.freq_range[1] <= self.freq_range[0]:
            raise ValueError(f"freq_range={self.freq_range} must satisfy f_min < f_max (1/day).")
        if self.periodogram_grid_resolution <= 0:
            raise ValueError("periodogram_grid_resolution must be > 0 (grid points per 1/day); "
                             f"got {self.periodogram_grid_resolution}.")
        self.period_truth = None

        self.results_frequency = {}
        self.results_power = {}

        self.frequency_grid = np.arange(self.freq_range[0], self.freq_range[1],
                                        1/self.periodogram_grid_resolution)

    # =============================================================================
    # =============================================================================
    def calc_pdc(self, calc_biased_flag=False, calc_unbiased_flag=True, errors=False):
        '''
        PDC periodogram of a scalar series: the values themselves, or, for a
        spectrum series, the RVs in time_series.calculated_vrad_list.

        :param calc_biased_flag: bool, compute the biased PDC ('PDC_biased').
        :param calc_unbiased_flag: bool, compute the unbiased PDC ('PDC_unbiased').
        :param errors: bool, use the error-aware distance (Binnenfeld, Shahaf &
                       Zucker 2024, Eq. 3) with time_series.errors (same unit
                       as the values); the results are then stored as
                       'PDC_errors_biased' / 'PDC_errors_unbiased', so both
                       variants can sit side by side.
        :raises ValueError: for a spectrum series without RVs, or n <= 3 for
                            the unbiased PDC.
        '''
        y = self._scalar_values('PDC')
        if errors:
            distance = pdc.error_aware_distance_matrix(y, self.time_series.errors)
        else:
            distance = pdc.scalar_distance_matrix(y)
        name = "PDC_errors" if errors else "PDC"
        self._scan(distance, name, calc_biased_flag, calc_unbiased_flag)

    # =============================================================================
    # =============================================================================
    def calc_usurper(self, calc_biased_flag=False, calc_unbiased_flag=True):
        '''
        USuRPER periodogram of a single-order spectrum series: the PDC of the
        spectra's chord distances on the unit sphere.

        :param calc_biased_flag: bool, compute the biased statistic ('USURPER_biased').
        :param calc_unbiased_flag: bool, compute the unbiased one ('USURPER_unbiased').
        :raises ValueError: for a scalar series, multi-order or non-overlapping
                            spectra, or n <= 3 for the unbiased statistic.
        '''
        if is_numeric_scalar(self.time_series.vals[0]):
            raise ValueError('USURPER is available for spectrum time-series only')
        waves, fluxes = _single_order_arrays(self.time_series.vals)
        distance = pdc.spectral_distance_matrix(waves, fluxes)
        self._scan(distance, "USURPER", calc_biased_flag, calc_unbiased_flag)

    # =============================================================================
    # =============================================================================
    def calc_partial_periodogram(self, partial_type="shift"):
        '''
        Partial periodogram of a single-order spectrum series (Binnenfeld et
        al. 2022): the periodicity of the RV shifts with the line-shape
        changes partialled out ('shift'), or of the shape changes with the RV
        shifts partialled out ('shape').

        The shape distance is the USuRPER distance of the spectra shifted to
        rest by time_series.calculated_vrad_list (km/s; copies -- the caller's
        spectra are not modified); the shift distance is |v_i - v_j|. Both are
        recomputed on every call.

        :param partial_type: 'shift' or 'shape'; the result is stored as
                             'partial_<partial_type>'.
        :raises ValueError: for an unknown partial_type, a scalar series,
                            multi-order spectra, or not one RV per epoch.
        '''
        if partial_type not in ("shift", "shape"):
            raise ValueError(f"partial_type must be 'shift' or 'shape', not {partial_type!r}.")
        if is_numeric_scalar(self.time_series.vals[0]):
            raise ValueError('Partial_periodogram is available for spectrum time-series only')

        waves, fluxes = _single_order_arrays(self.time_series.vals)
        vrads = self.time_series.calculated_vrad_list
        if len(vrads) != len(waves):
            raise ValueError(
                f"The partial periodogram needs one RV (km/s) per epoch in "
                f"time_series.calculated_vrad_list; got {len(vrads)} for {len(waves)} epochs.")
        rest_waves = [dsp.doppler_shift_wave(w, -v) for w, v in zip(waves, vrads)]
        shift = pdc.scalar_distance_matrix(vrads)
        shape = pdc.spectral_distance_matrix(rest_waves, fluxes)

        a, c = (shift, shape) if partial_type == "shift" else (shape, shift)
        a_centered, control = pdc.partial_centering(a, c)
        power = pdc.pdc_power(self.time_series.times, self.frequency_grid, a_centered,
                              control=control)
        self._store("partial_" + partial_type, power)

    # =============================================================================
    # =============================================================================
    def calc_gls(self):
        '''
        Generalized Lomb-Scargle periodogram (astropy) of a scalar series: the
        values themselves, or the RVs in time_series.calculated_vrad_list.

        :return: (frequency (1/day), power) arrays, also stored as 'GLS'.
        :raises ValueError: for a spectrum series without RVs.
        '''
        frequency = self.frequency_grid.copy()
        power = LombScargle(self.time_series.times, self._scalar_values('GLS')).power(frequency)
        self.results_frequency["GLS"] = frequency
        self.results_power["GLS"] = power
        return frequency, power

    # =============================================================================
    # =============================================================================
    def _scalar_values(self, method):
        '''
        The scalar values a periodogram runs on.

        :param method: str, the periodogram's name, for the error message.
        :return: the values of a scalar series, or the RVs (km/s) stored for a
                 spectrum series.
        :raises ValueError: for a spectrum series with no RVs stored.
        '''
        if is_numeric_scalar(self.time_series.vals[0]):
            return self.time_series.vals
        if len(self.time_series.calculated_vrad_list) != 0:
            return self.time_series.calculated_vrad_list
        raise ValueError(f'{method} is available for scalar time-series only, or for a '
                         'spectrum series with RVs in time_series.calculated_vrad_list')

    def _scan(self, distance, name, calc_biased_flag, calc_unbiased_flag):
        '''
        Center ``distance`` and scan it over the frequency grid, biased and/or
        unbiased, storing the results as '<name>_biased' / '<name>_unbiased'.
        '''
        times = self.time_series.times
        if calc_biased_flag:
            self._store(name + "_biased", pdc.pdc_power(times, self.frequency_grid,
                                                   pdc.double_centering(distance),
                                                   unbiased=False))
        if calc_unbiased_flag:
            centered = pdc.unbiased_u_centering(distance, len(times))
            self._store(name + "_unbiased", pdc.pdc_power(times, self.frequency_grid, centered))

    def _store(self, name, power):
        self.results_frequency[name] = self.frequency_grid.copy()
        self.results_power[name] = power

    # =============================================================================
    # =============================================================================
    def calc_pdc_pval(self, values, inverse_sf=False):
        '''
        This function returns the p-value of a given peak value in the PDC periodogram,
        based on Shen et. al. 2019 (https://arxiv.org/abs/1912.12150).
        '''

        N = len(self.time_series.times)
        if not inverse_sf:
            output = np.array(
                [chi2.sf(val * N + 1, 1) for val in values]
            )
        else:
            output = np.array(
                [(chi2.isf(val, 1)-1)/N for val in values]
            )
        return output

    # =============================================================================
    # =============================================================================
    def plot_periodograms(self, plot_vals=False, annotate_pval=None, figsize=(11, 4)):
        '''
        Plot the calculated periodograms, one panel each.

        :param plot_vals: bool, add a panel of the scalar values (or stored
                          RVs) against time, folded at period_truth[0] (days)
                          when period_truth is set.
        :param annotate_pval: iterable of p-values whose PDC power levels are
                              drawn on the PDC/USuRPER panels.
        :param figsize: (width, height per panel), inches.
        '''

        index = 0
        count = len(self.results_frequency) + plot_vals
        colors = ['red', 'orange', 'blue', 'green', 'purple', 'dodgerblue']
        if annotate_pval is not None:
            annotate_levels = self.calc_pdc_pval(annotate_pval, inverse_sf=True)

        fig, axs = plt.subplots(count, squeeze=False,
                                figsize=(figsize[0], count * figsize[1]))

        for method in self.results_frequency:
            axs[index, 0].plot(self.results_frequency[method],
                               self.results_power[method], 'k', linewidth=1.0)
            l = [method]
            # The p-value levels (Shen et al. 2019) hold for the unbiased
            # statistics only.
            if (annotate_pval is not None) and (method != 'GLS') and \
                    not method.endswith("_biased"):
                for anlev_ind, anlev in enumerate(annotate_levels):
                    axs[index, 0].hlines(y=anlev, xmin=self.freq_range[0], xmax=self.freq_range[1],
                                         linewidth=0.5, alpha=0.5, ls='--', color=colors[anlev_ind % 6])
                    l.append(f"p-value: {annotate_pval[anlev_ind]:.2g}")
            axs[index, 0].legend(l)

            if self.period_truth is not None:
                for p in self.period_truth:
                    axs[index, 0].axvline(x=1/p, alpha=0.5, ls='--')
            index = index + 1

        if plot_vals:
            v_list = self._scalar_values('plot_vals')
            times = np.asarray(self.time_series.times, float)
            if self.period_truth is not None:
                times = times % self.period_truth[0]

            axs[index, 0].scatter(times, v_list, alpha=0.6)
            axs[index, 0].set(xlabel="time (days)", ylabel="Vrad (km/s)")
            axs[index, 0].set_ylim(min(v_list), max(v_list))

        if index == 0:
            axs[index, 0].set(xlabel="Frequency [1/day]")
        else:
            axs[count - 1 - plot_vals, 0].set(xlabel="Frequency [1/day]")


def _single_order_arrays(sp_list):
    '''
    The wavelength and flux arrays of single-order spectra, validated for the
    common-grid resampling.

    :param sp_list: list of Spectrum-like objects (``.wv``/``.sp`` lists of
                    per-order arrays); wavelengths in Angstrom.
    :return: (waves, fluxes) -- lists of 1-D float arrays, one per spectrum.
    :raises ValueError: if a spectrum has more than one order, or its
                        wavelengths are not strictly increasing.
    '''
    waves, fluxes = [], []
    for i, s in enumerate(sp_list):
        if len(s.wv) != 1:
            raise ValueError(
                f"Spectrum {i} has {len(s.wv)} orders, but USuRPER and the partial "
                "periodograms use a single order. Select one first (e.g. "
                "Observations.select_orders).")
        w = np.asarray(s.wv[0], float)
        if not np.all(np.diff(w) > 0):
            raise ValueError(
                f"Spectrum {i}: wavelengths must be strictly increasing for the "
                "common-grid resampling.")
        waves.append(w)
        fluxes.append(s.sp[0])
    return waves, fluxes
