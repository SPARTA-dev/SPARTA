"""CCF1d: UNICOR cross-correlation RVs of a (multi-order) Spectrum against a Template.

Each order is correlated with the template on a log-lambda (velocity-even)
grid, the per-order CCFs are combined by maximum likelihood, and the RV and its
uncertainty are read from the combined peak. The array-level engine (CCF vs
velocity, order combination, sub-pixel peak, ML uncertainty, bisector) is the
standalone ``unicor`` package; this class handles the Spectrum/Template
orchestration and the astropy units at its boundary.

Conventions: velocities in km/s, in the 'cz' Doppler convention
(lambda_obs/lambda_rest = 1 + v/c) unless ``velocity_convention='relativistic'``;
wavelengths in Angstrom.

Reference: Zucker 2003, MNRAS 342, 1291 (ML combination of orders and the RV
uncertainty).
"""

import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from copy import deepcopy
import warnings
from unicor import ccf as unicor_ccf, dsp
from sparta import plotting as sp_plot

class CCF1d:
    '''
    UNICOR cross-correlation of one (multi-order) spectrum against a template.

    ``cross_correlate`` fills ``Corr`` (per order); ``combine`` fills
    ``CorrCombined`` (all orders, by maximum likelihood).
    '''
    # =============================================================================
    # =============================================================================
    def __init__(self):
        '''
        Set the default CCF velocity step ``self.default_dv`` (0.1 km/s).

        It is used when neither ``dv`` nor a log-lambda grid of the spectrum
        sets the step.
        '''
        self.default_dv = 0.1 * u.kilometer / u.second

    # =============================================================================
    # =============================================================================
    def cross_correlate(self, spectrum, template,
                        dv=None, vel_bound=150, errors_per_order=False,
                        velocity_convention='cz'):
        '''
        Cross-correlate every order of a spectrum with a template (UNICOR).

        The template is resampled onto each order's log-lambda grid continued
        by K = round(vel_bound / dv) steps on each side, and every lag is the
        Pearson correlation of the same observed pixels with the template
        shifted by that lag (``unicor.ccf_vs_velocity``). Correlating against
        a template on the observed span only would overlap fewer pixels at
        larger lags and pull the RV toward zero.

        Template coverage: only observed pixels for which the template is
        known at every lag are correlated. A template spanning just the
        observed range (e.g. another epoch, a co-added spectrum) therefore
        drops vel_bound worth of pixels at each end of the order; a template
        that covers no pixel across the whole scan makes that order NaN, with
        a warning. The template is never extrapolated, since constant padding
        would bring back the pull toward zero.

        The ML uncertainties (Zucker 2003, MNRAS 342, 1291) need the number of
        independent spectral bins. Resampling onto a finer grid adds samples
        but no information, so order i counts
        ``n_pix_i * min(1, dv / native_step_i)`` bins -- ``n_pix_i`` the
        pixels actually correlated (see template coverage) -- with the native step
        from ``spectrum.native_steps()`` (km/s; the coarsest sampling the
        spectrum has had). For an archive-resampled product, Observations
        seeds it with the detector step (e.g. HARPS ADP: ~0.609 km/s, where
        the 0.01 A archive grid is ~0.60 km/s at 5000 A but ~0.44 km/s at
        6800 A), so the archive's oversampling is not counted as information.

        That count assumes white noise on the native pixels. Filtering and
        per-order normalisation (``filter_spectrum`` / ``preprocess``)
        remove noise power and correlate it, so the uncertainty is then no
        longer calibrated: in Monte Carlo the ratio eRV / (RV scatter) is
        1.17 with preprocess's default filter_highcut=0.15 and 1.38 with
        filter_highcut=0.9, i.e. the errors are 17-38% conservative. Calibrate by
        simulation for preprocessed spectra.

        Parameters
        ----------
        spectrum : Spectrum
            The observation (not modified), barycentric-corrected
            (``Observations`` does this on ingest). Resampled onto a log-lambda grid
            of step ``dv`` unless it already is on one (``Info['GridType']``
            'log'), in which case its own step is the lag step.
        template : Template
            The template, at rest (not modified). A single-order model serves
            every order; a multi-order one is matched order by order.
        dv : float or astropy.units.Quantity, optional
            Log-lambda grid step (km/s if a plain number) for a spectrum not
            already on one. Default ``self.default_dv``, 0.1 km/s.
        vel_bound : float, (float, float) or astropy.units.Quantity
            The CCF is scanned over [-V, V] (km/s if plain numbers), V the
            largest ``|vel_bound|``. Default 150 km/s, the same in CCF2d and
            Observations: it spans the RV range of most binaries, and it
            equals the default margin of ``Template.cut_multiorder_like``, so
            a template cut with the defaults covers every observed pixel at
            every lag.
        errors_per_order : bool
            Also compute each order's ML uncertainty (otherwise 'eRV' is
            NaN; the combined uncertainty comes from ``combine``).
        velocity_convention : {'cz', 'relativistic'}
            How a lag maps to a radial velocity (see
            ``unicor.dsp.doppler_factor``). Lag k multiplies the wavelength by
            exp(k Delta), Delta the actual ln(lambda) step of the grid, and is
            reported as c expm1(k Delta) ('cz', lambda_obs/lambda_rest =
            1 + v/c; IAU 2000, IRAF fxcor) or c tanh(k Delta)
            ('relativistic'). For the same spectrum the 'cz' velocity exceeds
            the 'relativistic' one by ~v^2/2c (37.5 m/s at 150 km/s).

        Returns
        -------
        CCF1d
            self, with ``n_ord`` and ``Corr``, a dict:

            'vel'
                Lag velocities (Quantity, km/s): increasing, 0 at the central
                lag, not exactly uniform.
            'corr'
                (n_orders, n_lags) Pearson correlation per order; NaN rows for
                orders the template does not cover.
            'RV', 'eRV', 'peakCorr'
                (n_orders, 1) per-order RV (km/s), its ML uncertainty (km/s;
                NaN unless ``errors_per_order``) and CCF peak; NaN where the
                peak cannot be located (e.g. at the scan edge).
            'n_eff'
                (n_orders,) effective number of independent spectral bins.
            'units'
                The unit of 'vel' (km/s).
            'velocity_convention'
                The convention of 'vel' and 'RV'.

        Raises
        ------
        ValueError
            If ``vel_bound`` is below one grid step, or an order's grid is not
            uniform in ln(lambda) with the declared step, or the orders' steps
            differ.

        Warns
        -----
        UserWarning
            For each order the template does not cover across the scan.
        '''

        # Initialize:
        # ----------
        template = deepcopy(template)
        spec = deepcopy(spectrum)

        if dv is None:
            dv = self.default_dv
        elif type(dv) is not u.quantity.Quantity:
            dv = float(dv) << u.kilometer / u.second

        # A bare number is km/s, like every other velocity in SPARTA -- not
        # the unit dv happens to carry.
        if type(vel_bound) is not u.quantity.Quantity:
            vel_bound = np.array(vel_bound) << u.kilometer / u.second

        # Symmetric velocity half-range for the CCF scan (both signs valid).
        vmax = np.max(np.abs(vel_bound))

        # Read before any resampling below: the sampling that sets the noise
        # is the one the data arrived on (native_steps validates it against
        # the orders).
        native_steps = np.asarray(spec.native_steps(), float)

        # Ensure the observed spectrum is on a log-lambda (velocity-even) grid.
        if spec.Info.get('GridType') != 'log':
            spec.interpolate(delta=dv, method='log')
        dv = spec.Info['GridDelta']

        dv_kms = dv.to(u.km / u.s).value
        vmax_kms = u.Quantity(vmax, dv.unit).to(u.km / u.s).value

        n_lags = int(round(vmax_kms / dv_kms))
        if n_lags < 1:
            raise ValueError(f"vel_bound ({vmax_kms:g} km/s) must be at least one "
                             f"velocity step ({dv_kms:g} km/s)")

        # The lag axis is k * Delta in ln(lambda) only if every order really is
        # velocity-even with the declared step (e.g. dropping pixels breaks it
        # silently), and the orders' CCFs share one velocity axis only if they
        # share Delta.
        dln = np.array([dsp.log_grid_dln(w) for w in spec.wv])
        for I, step in enumerate(dsp.C_KMS * dln):
            if not np.isclose(step, dv_kms, rtol=1e-3):
                raise ValueError(
                    f"order {I}: the wavelength grid step is {step:.6g} km/s but "
                    f"Info['GridDelta'] declares {dv_kms:.6g} km/s")
        dln_common = float(np.median(dln))
        # Tolerate a thousandth of a lag of misregistration at the scan edge.
        if np.max(np.abs(dln / dln_common - 1)) * n_lags > 1e-3:
            raise ValueError(
                "the orders' ln(lambda) steps differ (relative spread "
                f"{np.ptp(dln) / dln_common:.2g}), so their CCFs cannot share one "
                "velocity axis; resample them with interpolate(method='log')")
        V = dsp.velocity_from_log_shift(np.arange(-n_lags, n_lags + 1) * dln_common,
                                        velocity_convention) << (u.km / u.s)

        Nord = len(spec.wv)
        RV = np.full((Nord, 1), np.nan)
        eRV = np.full((Nord, 1), np.nan)
        SpecCorr = np.full((Nord, 1), np.nan)
        corr = np.full((Nord, len(V)), np.nan)
        native_ratio = np.minimum(1.0, dv_kms / native_steps)
        # Overwritten below by the pixels each order actually correlates; an
        # order that fails keeps its full count, as its NaN CCF makes the
        # combined CCF NaN anyway.
        n_eff = np.array([len(w) for w in spec.wv], float) * native_ratio

        for I, w in enumerate(spec.wv):
            s = np.asarray(spec.sp[I], float)

            # A single-order template (e.g. a full-range library spectrum) is
            # reused for every order, as in CCF2d.
            It = I if len(template.model.wv) > 1 else 0
            t_wave = np.asarray(template.model.wv[It], float)
            t_flux = np.asarray(template.model.sp[It], float)

            # The template goes on this order's grid continued by n_lags steps
            # each side, so every lag correlates the same observed pixels
            # (no taper pulling the peak toward v = 0). An order the template
            # cannot serve is NaN, with a warning, instead of failing the epoch.
            w_ext = unicor_ccf.extend_log_grid(w, n_lags)
            spT = dsp.resample(t_wave, t_flux, w_ext)
            try:
                lo, hi = _template_covered_pixels(w_ext, t_wave, n_lags)
                v_kms, C = unicor_ccf.ccf_vs_velocity(spT[lo: hi + 2 * n_lags],
                                                      s[lo:hi], w[lo:hi],
                                                      velocity_convention)
            except ValueError as err:
                warnings.warn(f"order {I}: {err}; its CCF is set to NaN", stacklevel=2)
                continue
            n_eff[I] = (hi - lo) * native_ratio[I]
            corr[I, :] = C

            # Find the radial velocity by fitting a parabola to the CCF peak.
            # An order whose peak cannot be located (e.g. at the edge of the
            # velocity scan) is reported as NaN rather than failing the epoch.
            try:
                vPeak, ccfPeak = unicor_ccf.subpixel_peak(v_kms, C)
                vPeakErr = (unicor_ccf.rv_uncertainty(v_kms, C, vPeak, n_eff[I])
                            if errors_per_order else np.nan)
            except (ValueError, np.linalg.LinAlgError):
                vPeak, vPeakErr, ccfPeak = np.nan, np.nan, np.nan

            RV[I] = vPeak
            eRV[I] = vPeakErr
            SpecCorr[I] = ccfPeak

        self.Corr = {
            'vel': V,
            'corr': corr,
            'RV': RV,
            'eRV': eRV,
            'units': V.unit,
            'peakCorr': SpecCorr,
            'n_eff': n_eff,
            'velocity_convention': velocity_convention}
        self.n_ord = Nord

        return self

# =============================================================================
# =============================================================================
    def combine(self):
        '''
        Combine the per-order CCFs into one by maximum likelihood (Zucker 2003).

        Orders are weighted by their effective number of independent bins
        (``self.Corr['n_eff']``). The RV is read from the peak of the combined
        CCF, and its uncertainty from its curvature with the total number of
        bins. Run ``cross_correlate`` first; all orders share its lag axis.

        Returns
        -------
        CCF1d
            self, with ``CorrCombined``, a dict like ``Corr`` with scalar
            entries: 'vel', 'corr' (the combined CCF, (n_lags,)), 'RV' and
            'eRV' (km/s), 'peakCorr', 'units', 'n_eff' (the total) and
            'velocity_convention'. 'RV', 'eRV' and 'peakCorr' are NaN when the
            combined peak cannot be located.
        '''

        velocities = self.Corr['vel']
        CombinedCorr = unicor_ccf.combine_ccfs(self.Corr['corr'], n_eff=self.Corr['n_eff'])
        n_eff_total = float(np.sum(self.Corr['n_eff']))

        try:
            V, eRV, CorrPeak, vUnit = self.extract_rv(velocities, CombinedCorr)

            # Return the corresponding velocity grid.
            self.CorrCombined = {
                'vel': velocities,
                'corr': CombinedCorr,
                'RV': V,
                'eRV': eRV,
                'units': vUnit,
                'peakCorr': CorrPeak,
                'n_eff': n_eff_total,
                'velocity_convention': self.Corr['velocity_convention']}

        except (ValueError, np.linalg.LinAlgError):
            self.CorrCombined = {
                'vel': velocities,
                'corr': CombinedCorr,
                'RV': np.nan,
                'eRV': np.nan,
                'units': '',
                'peakCorr': np.nan,
                'n_eff': n_eff_total,
                'velocity_convention': self.Corr['velocity_convention']}

        return self

# =============================================================================
# =============================================================================
    def extract_rv(self, x, y, vel=None, n_ord=None):
        """
        Radial velocity and its maximum-likelihood uncertainty from a CCF.

        The uncertainty uses the effective number of independent bins stored
        by :meth:`cross_correlate` (``self.Corr['n_eff']``), summed over the
        orders behind ``y``.

        Parameters
        ----------
        x : astropy.units.Quantity or array_like
            Velocity lags (km/s).
        y : array_like
            CCF values at ``x``.
        vel : float, optional
            If given, the uncertainty is evaluated at this velocity (km/s)
            instead of at the fitted peak, and the peak value is NaN.
        n_ord : int, optional
            Number of orders behind ``y``: all of them (the default,
            ``self.n_ord``), i.e. the combined CCF. A single order of a
            multi-order spectrum is ambiguous here -- use
            ``unicor.rv_uncertainty`` with ``self.Corr['n_eff'][i]``.

        Returns
        -------
        RV, eRV : float
            The velocity and its uncertainty (km/s).
        ccf_peak : float
            The CCF at the peak (NaN if ``vel`` was given).
        unit : astropy.units.Unit or None
            The unit of ``x``; None for a plain array.

        Raises
        ------
        ValueError
            Before ``cross_correlate``, for a subset ``n_ord``, or if the peak
            cannot be located.
        """
        x_kms = x.value if isinstance(x, u.Quantity) else np.asarray(x, float)
        if vel is None:
            RV, ccf_value = unicor_ccf.subpixel_peak(x_kms, y)
        else:
            RV, ccf_value = vel, np.nan

        eRV = unicor_ccf.rv_uncertainty(x_kms, y, RV, self._n_eff_total(n_ord))

        xUnit = x.unit if isinstance(x, u.Quantity) else None
        return RV, eRV, ccf_value, xUnit

# =============================================================================
# =============================================================================
    def calc_bis(self, x, y, bisect_val=(0.35, 0.95), n_ord=None):
        """
        Bisector span of a single-peaked CCF (see :func:`unicor.bisector_span`).

        .. warning:: The BIS formula is known to be incorrect and is under
           review: it differences CCF values at the bisector levels, not
           velocities. It is pinned, unchanged, for a later fix; do not use
           its output scientifically.

        Parameters
        ----------
        x : astropy.units.Quantity
            Velocity lags (km/s).
        y : array_like
            CCF values at ``x``.
        bisect_val : (float, float)
            Lower and upper bisector levels, as fractions of the CCF maximum.
        n_ord : int, optional
            Number of orders behind ``y``; see :meth:`extract_rv`.

        Returns
        -------
        BIS, eBIS : float
            NaN when a level does not cut the CCF exactly twice.
        unit : astropy.units.Unit or None
            The unit of ``x``; None with the NaN result.
        """
        BIS, eBIS = unicor_ccf.bisector_span(x.value, y, self._n_eff_total(n_ord),
                                             levels=bisect_val)
        if np.isnan(BIS):
            return np.nan, np.nan, None
        return BIS, eBIS, x.unit

    def _n_eff_total(self, n_ord):
        """
        Independent spectral bins behind a CCF built from ``n_ord`` orders.

        Parameters
        ----------
        n_ord : int or None
            Number of orders; None means all of them.

        Returns
        -------
        float
            The summed ``self.Corr['n_eff']``.

        Raises
        ------
        ValueError
            Before :meth:`cross_correlate` has run, or when ``n_ord`` names a
            strict subset of the orders (which orders is then unknown).
        """
        if not hasattr(self, 'Corr'):
            raise ValueError("run cross_correlate first: the uncertainty needs "
                             "the number of independent spectral bins it records")
        if n_ord is not None and n_ord != self.n_ord:
            raise ValueError(
                f"n_ord={n_ord} is a subset of the {self.n_ord} orders; for one "
                "order use unicor.rv_uncertainty / unicor.bisector_span with "
                "self.Corr['n_eff'][i]")
        return float(np.sum(self.Corr['n_eff']))

    def subpixel_peak(self, vels, ccf, v=None, npts=5):
        """
        Second-order (parabolic) estimate of the CCF around its peak.

        Parameters
        ----------
        vels : astropy.units.Quantity or array_like
            Velocity lags (km/s).
        ccf : array_like
            CCF values at ``vels``.
        v : float, optional
            If given, return the interpolated CCF value at this velocity
            (km/s) instead of the peak.
        npts : int
            Odd number of points around the peak used in the fit.

        Returns
        -------
        (float, float) or float
            (v_peak (km/s), ccf_peak), or the CCF value at ``v``.
        """
        if isinstance(vels, u.Quantity):
            vels = vels.value
        if v is None:
            return unicor_ccf.subpixel_peak(vels, ccf, npts=npts)
        return unicor_ccf.ccf_at(vels, ccf, v)


    # =============================================================================
    # =============================================================================
    def plot(self, plot_combined=True, plot_orders=True, orders=None, alpha=0.125):
        '''
        Plot the per-order and combined CCFs against velocity.

        The CCFs are derived curves on the lag grid, drawn as black lines (the
        orders faint, the combined CCF bold); the combined RV and its +- eRV
        band are red (``sparta.plotting``).

        Parameters
        ----------
        plot_combined : bool
            Plot the combined CCF and its RV +- eRV band, if ``combine`` ran.
        plot_orders : bool
            Plot the CCF of each order in ``orders``.
        orders : sequence of int, optional
            Orders to plot; default all.
        alpha : float
            Opacity of the per-order CCFs.

        Returns
        -------
        matplotlib.figure.Figure
        '''
        if orders is None:
            orders = np.arange(self.n_ord)
        vel = u.Quantity(self.Corr['vel'], u.km / u.s).value

        fig, ax = plt.subplots(figsize=(8, 3.2), layout='constrained')

        if plot_orders:
            for o in orders:
                sp_plot.plot_curve(ax, vel, self.Corr['corr'][o], lw=0.75, alpha=alpha)

        combined = getattr(self, 'CorrCombined', None)
        if plot_combined and combined is not None:
            sp_plot.plot_curve(ax, vel, combined['corr'], lw=1.6, label='combined CCF')
            rv, erv = combined['RV'], combined['eRV']
            if np.isfinite(rv):
                ax.axvline(rv, color=sp_plot.MODEL_COLOR, lw=0.8,
                           label=rf'RV = {rv:.3f} $\pm$ {erv:.3f} km/s')
                ax.axvspan(rv - erv, rv + erv, color=sp_plot.MODEL_COLOR, alpha=0.25, lw=0)
            ax.legend(loc='upper right')

        ax.set_xlabel('Radial velocity (km/s)')
        ax.set_ylabel('CCF')

        return fig


def _template_covered_pixels(w_ext, t_wave, n_lags):
    """Observed pixels whose template is known at every lag of the scan.

    Parameters
    ----------
    w_ext : ndarray
        The order's log-lambda grid extended by ``n_lags`` steps on each side
        (Angstrom), see ``unicor.extend_log_grid``.
    t_wave : ndarray
        The template's wavelengths (Angstrom, increasing).
    n_lags : int
        Number of lags K on each side of zero.

    Returns
    -------
    lo, hi : int
        Observed pixels ``lo:hi`` need exactly the template nodes
        ``w_ext[lo: hi + 2K]``, all inside the template's span.

    Raises
    ------
    ValueError
        If no observed pixel is covered at every lag.
    """
    # A millionth of a step absorbs the round-off of exp() when the template
    # sits on the observed grid itself.
    tol = 1e-6 * np.log(w_ext[1] / w_ext[0])
    ln_ext = np.log(w_ext)
    inside = np.flatnonzero((ln_ext >= np.log(t_wave[0]) - tol) &
                            (ln_ext <= np.log(t_wave[-1]) + tol))
    n_obs = len(w_ext) - 2 * n_lags
    # The covered nodes are contiguous; pixel i needs nodes i .. i + 2K.
    lo, hi = (inside[0], inside[-1] - 2 * n_lags + 1) if inside.size else (0, 0)
    if hi <= lo:
        raise ValueError(
            f"the template ({t_wave[0]:.2f}-{t_wave[-1]:.2f} A) does not cover any "
            f"pixel of the order ({w_ext[n_lags]:.2f}-{w_ext[n_lags + n_obs - 1]:.2f} A) "
            f"across the +-{n_lags}-lag scan; use a wider template or a smaller vel_bound")
    return lo, hi
