#                   ----------------------------------
#                      CCF1d.py (SPARTA UNICOR class)
#                   ----------------------------------
# This file defines the "CCF1d" class. An object of this class stores
# a Spectrum object, saved in the self.spec field and a Template object
# stored in the self.template field.
#
# ---------------------------------------------
# A CCF1d class stores the following methods:
# ---------------------------------------------
# 1) CrossCorrelateSpec - multi-order cross-correlation of a spectrum vs a template.
# 2) CombineCCFs        - combine the per-order CCFs (Zucker 2003 maximum likelihood).
# 3) extract_RV         - radial velocity + uncertainty from the CCF peak.
# 4) calcBIS            - bisector span (BIS) of the (single-peaked) CCF.
# 5) subpixel_CCF       - parabolic sub-pixel estimate of the CCF peak / value.
# 6) plotCCFs           - plot the per-order and combined CCFs.
#
# The array-level engine (CCF vs velocity, Zucker combination, sub-pixel peak,
# ML uncertainty, bisector) is the standalone ``unicor`` package; this class
# handles the Spectrum/Template orchestration and the astropy units at its
# boundary.
#
# Conventions: radial velocities in km/s, in the 'cz' Doppler convention
# (lambda_obs/lambda_rest = 1 + v/c) unless velocity_convention='relativistic';
# the CCF is computed on a log-lambda (velocity-even) grid.
#
# Dependencies: numpy, astropy, matplotlib, copy, unicor.
# Last update: 2026-10-14 (lag -> velocity by the Doppler convention).


import numpy as np
from astropy import constants as consts, units as u
import matplotlib.pyplot as plt
from copy import deepcopy
import warnings
from unicor import ccf as unicor_ccf, dsp

class CCF1d:
    # =============================================================================
    # =============================================================================
    def __init__(self):
        '''
        No input required. Sets the speed of light ``self.c`` (km/s) and the
        default CCF velocity step ``self.default_dv`` (0.1 km/s), used when a
        spectrum carries no log-lambda grid step of its own.
        '''
        self.c = consts.c.to('km/s')  # speed of light, km/s
        self.default_dv = 0.1 * u.kilometer / u.second

    # =============================================================================
    # =============================================================================
    def CrossCorrelateSpec(self, spec_in, template_in,
                           dv=None, VelBound=100, err_per_ord=False, fastccf=False,
                           velocity_convention='cz'):
        '''
        All input is optional, and needs to be called along
        with its keyword. Below appears a list of the possible input
        variables.

        :param: template - the template for the CCF (see template class)
        :param: dv       - scalar. If the spectrum is not logarithmically
                         evenly-spaced it is resampled, with a stepsize
                         determined according to dv (in km/s). Default 0.1
        :param: VelBounds - scalar. The velocity bounds for the CCF.
                         detemined according to [-VelBound, VelBound]
                         Default is 100 km/s
        :param: err_per_ord - Boolean. Indicates if the error should be
                         calculated (from maximum likelihood) to each
                         order.
        :param: velocity_convention - 'cz' (default) or 'relativistic': how a
                         lag maps to a radial velocity (see
                         ``unicor.dsp.doppler_factor``). Lag k multiplies the
                         wavelength by exp(k * Delta), Delta the actual
                         ln(lambda) step of the grid, and is reported as
                         c * expm1(k * Delta) ('cz', lambda_obs/lambda_rest =
                         1 + v/c; IAU 2000, IRAF fxcor) or c * tanh(k * Delta)
                         ('relativistic'). The two differ by ~v^2/2c (37.5 m/s
                         at 150 km/s); reading the lag as k * dv was off from
                         'cz' by -v^2/2c.

        The template is resampled onto each order's log-lambda grid continued
        by K = round(VelBound / dv) steps on each side, and every lag is the
        Pearson correlation of the same observed pixels with the template
        shifted by that lag (``unicor.ccf_vs_velocity``). Correlating against
        a template on the observed span only would overlap fewer pixels at
        larger lags and pull the RV toward zero.

        Template coverage: only observed pixels for which the template is
        known at every lag are correlated. A template spanning just the
        observed range (e.g. another epoch, a co-added spectrum) therefore
        drops VelBound worth of pixels at each end of the order; a template
        that covers no pixel across the whole scan makes that order NaN, with
        a warning. The template is never extrapolated, since constant padding
        would bring back the pull toward zero.

        The ML uncertainties (Zucker 2003, MNRAS 342, 1291) need the number of
        independent spectral bins. Resampling onto a finer grid adds samples
        but no information, so order i counts
        ``n_pix_i * min(1, dv / native_step_i)`` bins -- ``n_pix_i`` the
        pixels actually correlated (see template coverage) -- with the native step
        from ``spec_in.native_steps()`` (km/s; the coarsest sampling the
        spectrum has had).

        That count assumes white noise on the native pixels. Filtering and
        per-order normalisation (``FilterSpectrum`` / ``SpecPreProccess``)
        remove noise power and correlate it, so the uncertainty is then no
        longer calibrated: in Monte Carlo the ratio eRV / (RV scatter) is
        1.17 with SpecPreProccess's default FilterHC=0.15 and 1.38 with
        FilterHC=0.9, i.e. the errors are 17-38% conservative. Calibrate by
        simulation for preprocessed spectra.

        :return: self.Corr - a dictionary with the following fields:
                         'vel' - velocity vector at which correlation
                                 was calculated (increasing, 0 at the
                                 central lag; not exactly uniform).
                         'corr'- correlation matrix,
                                 (# orders X length of velocity vector)
                         'RV'  - Derived radial velocity
                         'eRV' - Corresponding uncertainty.
                         'peakCorr' - Corresponding CCF peak.
                         'n_eff' - effective number of independent spectral
                                   bins per order (dimensionless).
                         'velocity_convention' - the convention of 'vel'
                                   and 'RV'.
        '''

        # Initialize:
        # ----------
        template = deepcopy(template_in)
        spec = deepcopy(spec_in)

        if dv is None:
            dv = self.default_dv
        elif type(dv) is not u.quantity.Quantity:
            dv = float(dv) << u.kilometer / u.second

        # A bare number is km/s, like every other velocity in SPARTA -- not
        # the unit dv happens to carry.
        if type(VelBound) is not u.quantity.Quantity:
            VelBound = np.array(VelBound) << u.kilometer / u.second

        # Symmetric velocity half-range for the CCF scan (both signs valid).
        vmax = np.max(np.abs(VelBound))

        # Read before any resampling below: the sampling that sets the noise
        # is the one the data arrived on (native_steps validates it against
        # the orders).
        native_steps = np.asarray(spec.native_steps(), float)

        # Ensure the observed spectrum is on a log-lambda (velocity-even) grid.
        if spec.Info.get('GridType') != 'log':
            spec.InterpolateSpectrum(delta=dv, InterpMethod='log')
        dv = spec.Info['GridDelta']

        dv_kms = dv.to(u.km / u.s).value
        vmax_kms = u.Quantity(vmax, dv.unit).to(u.km / u.s).value

        n_lags = int(round(vmax_kms / dv_kms))
        if n_lags < 1:
            raise ValueError(f"VelBound ({vmax_kms:g} km/s) must be at least one "
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
                "velocity axis; resample them with InterpolateSpectrum(InterpMethod='log')")
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
                            if err_per_ord else np.nan)
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
    def CombineCCFs(self):
        '''
        This routine takes a matrix of CCF values (# orders X length of velocity vector)
        that were calculated by the CrossCorrelateSpec routine, combines the CCFs into
        a single one, based on a maximum-likelihood approach (Zucker, 2003, MNRAS).
        Orders are weighted by their effective number of independent bins
        (``self.Corr['n_eff']``). The RV is derived from the peak of the
        cross-correlation, and its uncertainty from the curvature of the combined
        CCF with the total number of bins.

        :param: none.
        :return: A 'CorrCombined' dictionary, with derived combined correlation,
                derived velocity and uncertainty. The structure is similar
                to the 'Corr' dictionary; 'n_eff' is the total number of
                independent bins, 'velocity_convention' that of 'Corr'.

        NOTE: the number of lags is assumed to be identical for all orders
        '''

        velocities = self.Corr['vel']
        CombinedCorr = unicor_ccf.combine_ccfs(self.Corr['corr'], n_eff=self.Corr['n_eff'])
        n_eff_total = float(np.sum(self.Corr['n_eff']))

        try:
            V, eRV, CorrPeak, vUnit = self.extract_RV(velocities, CombinedCorr)

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
    def extract_RV(self, x, y, vel=None, n_ord=None):
        """
        Radial velocity and its maximum-likelihood uncertainty from a CCF.

        The uncertainty uses the effective number of independent bins stored
        by :meth:`CrossCorrelateSpec` (``self.Corr['n_eff']``), summed over the
        orders behind ``y``.

        :param x: velocity lags (Quantity or km/s array).
        :param y: CCF values at ``x``.
        :param vel: if given, the uncertainty is evaluated at this velocity
                    (km/s) instead of at the fitted peak, and no peak value
                    is returned.
        :param n_ord: number of orders behind ``y``: all of them (the default,
                      ``self.n_ord``), i.e. the combined CCF. A single order of
                      a multi-order spectrum is ambiguous here -- use
                      ``unicor.rv_uncertainty`` with ``self.Corr['n_eff'][i]``.
        :return: (RV, eRV, ccf_peak, unit) -- velocities in the unit of ``x``.
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
    def calcBIS(self, x, y, bisect_val=(0.35, 0.95), n_ord=None):
        """
        Bisector span of a single-peaked CCF (see :func:`unicor.bisector_span`).

        :param x: velocity lags (Quantity).
        :param y: CCF values at ``x``.
        :param bisect_val: lower and upper bisector levels, as fractions of the
                           CCF maximum.
        :param n_ord: number of orders behind ``y``; see :meth:`extract_RV`.
        :return: (BIS, eBIS, unit); (nan, nan, None) when a level does not cut
                 the CCF exactly twice.
        """
        BIS, eBIS = unicor_ccf.bisector_span(x.value, y, self._n_eff_total(n_ord),
                                             levels=bisect_val)
        if np.isnan(BIS):
            return np.nan, np.nan, None
        return BIS, eBIS, x.unit

    def _n_eff_total(self, n_ord):
        """
        Independent spectral bins behind a CCF built from ``n_ord`` orders.

        :param n_ord: number of orders; None means all of them.
        :return: float, the summed ``self.Corr['n_eff']``.
        :raises ValueError: before :meth:`CrossCorrelateSpec` has run, or when
                            ``n_ord`` names a strict subset of the orders
                            (which orders is then unknown).
        """
        if not hasattr(self, 'Corr'):
            raise ValueError("run CrossCorrelateSpec first: the uncertainty needs "
                             "the number of independent spectral bins it records")
        if n_ord is not None and n_ord != self.n_ord:
            raise ValueError(
                f"n_ord={n_ord} is a subset of the {self.n_ord} orders; for one "
                "order use unicor.rv_uncertainty / unicor.bisector_span with "
                "self.Corr['n_eff'][i]")
        return float(np.sum(self.Corr['n_eff']))

    def subpixel_CCF(self, vels, ccf, v=None, Npts=5):
        """
        Second-order (parabolic) estimate of the CCF around its peak.

        :param vels: velocity lags (Quantity or km/s array).
        :param ccf: CCF values at ``vels``.
        :param v: if given, return the interpolated CCF value at this velocity
                  (km/s) instead of the peak.
        :param Npts: odd number of points around the peak used in the fit.
        :return: (v_peak, ccf_peak) in km/s, or the CCF value at ``v``.
        """
        if isinstance(vels, u.Quantity):
            vels = vels.value
        if v is None:
            return unicor_ccf.subpixel_peak(vels, ccf, npts=Npts)
        return unicor_ccf.ccf_at(vels, ccf, v)


    # =============================================================================
    # =============================================================================
    def plotCCFs(self, PlotCombined=True, PlotSingleOrds=True, ords=None, alpha=0.125, **kwargs):
        '''
        Produce plots of the calculated CCFs.
        :param PlotCombined: Boolean. Plot the combined CCF (if exists)
        :param PlotSingleOrds: Boolean. Plot the ccf of each order required.
        :param ords: a specifiec list of order numbers to plot.
        :param kwargs: maybe will contain some plot spec.
        :return: fig object
        '''
        if ords is None:
            ords = np.arange(self.n_ord)

        fig = plt.figure(figsize=(13, 4), dpi= 80, facecolor='w', edgecolor='k')

        if PlotSingleOrds:
            for o in ords:
                plt.plot(self.Corr['vel'].value,
                         self.Corr['corr'][o], 'k', alpha=alpha, linewidth=0.75)

        if PlotCombined:
            try:
                plt.plot(self.CorrCombined['vel'],
                         self.CorrCombined['corr'], 'k', linewidth=2.5)
                plt.axvspan(self.CorrCombined['RV']-self.CorrCombined['eRV'],
                            self.CorrCombined['RV']+self.CorrCombined['eRV'],
                            color='red', alpha=0.35)
            except AttributeError:
                pass

        plt.xlabel(r'Velocity ' + '[' + str(self.Corr['vel'].unit) + ']')
        plt.ylabel(r'CCF')
        plt.grid()

        return fig


def _template_covered_pixels(w_ext, t_wave, n_lags):
    """Observed pixels whose template is known at every lag of the scan.

    :param w_ext: the order's log-lambda grid extended by ``n_lags`` steps on
                  each side (Angstrom), see ``unicor.extend_log_grid``.
    :param t_wave: the template's wavelengths (Angstrom, increasing).
    :param n_lags: number of lags K on each side of zero.
    :return: (lo, hi) -- observed pixels ``lo:hi`` need exactly the template
             nodes ``w_ext[lo: hi + 2K]``, all inside the template's span.
    :raises ValueError: if no observed pixel is covered at every lag.
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
            f"across the +-{n_lags}-lag scan; use a wider template or a smaller VelBound")
    return lo, hi
