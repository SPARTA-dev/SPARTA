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
from unicor import dsp
from sparta import pdc
from sparta import plotting as sp_plot
from sparta.timeseries import is_numeric_scalar


class PeriodicityDetector:
    '''
    Periodograms of one TimeSeries on one frequency grid.

    Attributes
    ----------
    time_series : TimeSeries
        The series searched (times in days).
    frequency_grid : ndarray
        Trial frequencies (1/day).
    results_frequency, results_power : dict of str to ndarray
        Every periodogram computed, under its name: 'GLS'; 'PDC_unbiased' /
        'PDC_biased'; 'PDC_errors_unbiased' / 'PDC_errors_biased'
        (error-aware PDC); 'USURPER_unbiased' / 'USURPER_biased';
        'partial_shift' / 'partial_shape'. Frequencies in 1/day, power
        dimensionless.
    period_truth : list of float or None
        Known periods (days), marked by ``plot_periodograms``; default None.
    '''

    # =============================================================================
    # =============================================================================
    def __init__(self, **kwargs):
        '''
        Set up the frequency grid; keyword arguments only.

        Parameters
        ----------
        time_series : TimeSeries
            The time series to search.
        freq_range : (float, float)
            (f_min, f_max), the grid boundaries (1/day); required. f_min must
            be > 0, since the phase distance takes dt mod (1/f).
        periodogram_grid_resolution : float
            Grid points per 1/day; required. The grid step is
            1 / resolution (1/day), and f_max is excluded.

        Raises
        ------
        TypeError
            If ``freq_range`` or ``periodogram_grid_resolution`` is missing.
        ValueError
            If they do not define a non-empty grid of positive frequencies.
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
        Phase-distance-correlation (PDC) periodogram of a scalar series.

        Runs on the values themselves or, for a spectrum series, on the RVs
        in ``time_series.calculated_vrad_list`` (Zucker 2018).

        Parameters
        ----------
        calc_biased_flag : bool
            Compute the biased (double-centered) PDC, 'PDC_biased'.
        calc_unbiased_flag : bool
            Compute the unbiased (U-centered) PDC, 'PDC_unbiased'.
        errors : bool
            Use the error-aware distance (Binnenfeld, Shahaf & Zucker 2024,
            Eq. 3) with ``time_series.errors`` (the values' unit); the results
            are then stored as 'PDC_errors_biased' / 'PDC_errors_unbiased', so
            both variants can sit side by side.

        Returns
        -------
        None
            The power (dimensionless) on ``frequency_grid`` (1/day) is stored
            in ``results_power``.

        Raises
        ------
        ValueError
            For a spectrum series without RVs, or n <= 3 for the unbiased PDC.
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
        USuRPER periodogram of a single-order spectrum series.

        The PDC of the spectra's chord distances on the unit sphere
        (Binnenfeld, Shahaf & Zucker 2020): sensitive to any periodic change
        of the spectra, shifts and shape changes alike, with no template.

        Parameters
        ----------
        calc_biased_flag : bool
            Compute the biased statistic, 'USURPER_biased'.
        calc_unbiased_flag : bool
            Compute the unbiased one, 'USURPER_unbiased'.

        Returns
        -------
        None
            The power on ``frequency_grid`` (1/day) is stored in
            ``results_power``.

        Raises
        ------
        ValueError
            For a scalar series, multi-order or non-overlapping spectra, or
            n <= 3 for the unbiased statistic.
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

        Parameters
        ----------
        partial_type : {'shift', 'shape'}
            Which partial periodogram; stored as 'partial_<partial_type>'.

        Returns
        -------
        None
            The power on ``frequency_grid`` (1/day) is stored in
            ``results_power``.

        Raises
        ------
        ValueError
            For an unknown ``partial_type``, a scalar series, multi-order
            spectra, or not one RV per epoch.
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

        Returns
        -------
        frequency : ndarray
            A copy of ``frequency_grid`` (1/day).
        power : ndarray
            GLS power (astropy's default 'standard' normalization). Both are
            also stored as 'GLS'.

        Raises
        ------
        ValueError
            For a spectrum series without RVs.
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

        Parameters
        ----------
        method : str
            The periodogram's name, for the error message.

        Returns
        -------
        list or ndarray
            The values of a scalar series, or the RVs (km/s) stored for a
            spectrum series.

        Raises
        ------
        ValueError
            For a spectrum series with no RVs stored.
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
        '''Store ``power`` and a copy of the frequency grid under ``name``.'''
        self.results_frequency[name] = self.frequency_grid.copy()
        self.results_power[name] = power

    # =============================================================================
    # =============================================================================
    def calc_pdc_pval(self, values, inverse_sf=False):
        '''
        Convert unbiased PDC power to p-values, or p-values to power levels.

        Uses the chi-square approximation of the distance-correlation test
        (Shen et al. 2019, arXiv:1912.12150):
        p = chi2_1.sf(n * power + 1), n the number of epochs
        (``sparta.pdc.pdc_pvalue`` / ``pdc_power_level``). It holds for the
        unbiased statistics at a single trial frequency; it is not a
        false-alarm probability over the whole grid.

        Parameters
        ----------
        values : iterable of float
            PDC power values, or p-values if ``inverse_sf``.
        inverse_sf : bool
            If True, return the power level of each p-value instead.

        Returns
        -------
        ndarray
            The p-values, or the power levels.
        '''
        n = len(self.time_series.times)
        if inverse_sf:
            return pdc.pdc_power_level(list(values), n)
        return pdc.pdc_pvalue(list(values), n)

    # =============================================================================
    # =============================================================================
    def plot_periodograms(self, plot_vals=False, annotate_pval=None, figsize=(11, 4)):
        '''
        Plot the calculated periodograms, one panel each.

        A periodogram is a derived curve on the chosen frequency grid, drawn
        as a black line; the known periods (``period_truth``) are blue dashed
        vertical lines (blue is the truth throughout ``sparta.plotting``) and
        the p-value levels grey lines of different styles.

        Parameters
        ----------
        plot_vals : bool
            Add a panel of the scalar values (or stored RVs, km/s) as black
            points, with error bars when ``time_series.errors`` holds one per
            epoch, against time (days) -- or against phase when
            ``period_truth`` is set, folded at ``period_truth[0]``.
        annotate_pval : iterable of float, optional
            p-values whose power levels (``calc_pdc_pval``) are drawn on the
            unbiased PDC / USuRPER / partial panels.
        figsize : (float, float)
            Width and height per panel (inches).

        Returns
        -------
        matplotlib.figure.Figure
        '''
        count = len(self.results_frequency) + plot_vals
        linestyles = ['--', ':', '-.', (0, (5, 1, 1, 1, 1, 1))]
        if annotate_pval is not None:
            annotate_levels = self.calc_pdc_pval(annotate_pval, inverse_sf=True)

        fig, axs = plt.subplots(count, squeeze=False, layout='constrained',
                                figsize=(figsize[0], count * figsize[1]))
        axs = axs[:, 0]

        for ax, method in zip(axs, self.results_frequency):
            sp_plot.plot_curve(ax, self.results_frequency[method], self.results_power[method],
                               label=method)
            # The p-value levels (Shen et al. 2019) hold for the unbiased
            # statistics only.
            if (annotate_pval is not None) and (method != 'GLS') and \
                    not method.endswith("_biased"):
                for k, level in enumerate(annotate_levels):
                    ax.axhline(level, color=sp_plot.GREY, lw=0.8,
                               ls=linestyles[k % len(linestyles)],
                               label=f"p-value: {annotate_pval[k]:.2g}")
            if self.period_truth is not None:
                for k, p in enumerate(self.period_truth):
                    ax.axvline(1 / p, color=sp_plot.EXTRA_COLOR, lw=0.8, ls='--',
                               label='known period' if k == 0 else None)
            ax.set_ylabel('Power')
            ax.legend(loc='upper right')
        if len(self.results_frequency):
            axs[len(self.results_frequency) - 1].set_xlabel('Frequency (1/day)')

        if plot_vals:
            ax = axs[-1]
            v_list = np.asarray(self._scalar_values('plot_vals'), float)
            times = np.asarray(self.time_series.times, float)
            errors = np.asarray(self.time_series.errors, float)
            errors = errors if errors.shape == v_list.shape else None
            if self.period_truth is not None:
                times = (times / self.period_truth[0]) % 1.0
                ax.set_xlabel(f'Phase (P = {self.period_truth[0]:g} days)')
            else:
                ax.set_xlabel('Time (days)')
            sp_plot.plot_points(ax, times, v_list, errors)
            ax.set_ylabel('RV (km/s)')

        return fig


def _single_order_arrays(sp_list):
    '''
    The wavelength and flux arrays of single-order spectra, validated for the
    common-grid resampling.

    Parameters
    ----------
    sp_list : list of Spectrum
        Spectrum-like objects (``.wv`` / ``.sp`` lists of per-order arrays);
        wavelengths in Angstrom.

    Returns
    -------
    waves, fluxes : list of ndarray
        1-D arrays, one per spectrum.

    Raises
    ------
    ValueError
        If a spectrum has more than one order, or its wavelengths are not
        strictly increasing.
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
