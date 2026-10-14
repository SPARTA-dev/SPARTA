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
# Conventions: radial velocities in km/s; the CCF is computed on a log-lambda
# (velocity-even) grid.
#
# Dependencies: numpy, astropy, matplotlib, copy, unicor.
# Last update: 2026-10-14 (array engine moved to the unicor package).


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
                           dv=None, VelBound=100, err_per_ord=False, fastccf=False):
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

        The ML uncertainties (Zucker 2003, MNRAS 342, 1291) need the number of
        independent spectral bins. Resampling onto a finer grid adds samples
        but no information, so order i counts
        ``len(order) * min(1, dv / native_step_i)`` bins, with the native step
        from ``spec_in.native_steps()`` (km/s).

        :return: self.Corr - a dictionary with the following fields:
                         'vel' - velocity vector at which correlation
                                 was calculated.
                         'corr'- correlation matrix,
                                 (# orders X length of velocity vector)
                         'RV'  - Derived radial velocity
                         'eRV' - Corresponding uncertainty.
                         'peakCorr' - Corresponding CCF peak.
                         'n_eff' - effective number of independent spectral
                                   bins per order (dimensionless).
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

        # Read before any resampling below; InterpolateSpectrum preserves it,
        # but the sampling that sets the noise is the one the data arrived on.
        native_steps = np.asarray(spec.native_steps(), float)
        if native_steps.shape != (len(spec.wv),):
            raise ValueError(f"{native_steps.size} native steps recorded for "
                             f"{len(spec.wv)} orders")
        if not np.all(np.isfinite(native_steps) & (native_steps > 0)):
            raise ValueError(f"native steps must be finite and positive, got {native_steps}")

        # Ensure the observed spectrum is on a log-lambda (velocity-even) grid.
        if spec.Info.get('GridType') != 'log':
            spec.InterpolateSpectrum(delta=dv, InterpMethod='log')
        dv = spec.Info['GridDelta']

        dv_kms = dv.to(u.km / u.s).value
        vmax_kms = u.Quantity(vmax, dv.unit).to(u.km / u.s).value

        Nord = len(spec.wv)
        RV = np.full((Nord, 1), np.nan)
        eRV = np.full((Nord, 1), np.nan)
        SpecCorr = np.full((Nord, 1), np.nan)
        n_eff = np.array([len(w) for w in spec.wv], float) * np.minimum(1.0, dv_kms / native_steps)
        V, corr = None, None

        for I, w in enumerate(spec.wv):
            s = spec.sp[I]

            # The lag axis is k * dv only if this order really is velocity-even
            # with the declared step (e.g. dropping pixels breaks it silently).
            step = dsp.log_grid_step(w)
            if not np.isclose(step, dv_kms, rtol=1e-3):
                raise ValueError(
                    f"order {I}: the wavelength grid step is {step:.6g} km/s but "
                    f"Info['GridDelta'] declares {dv_kms:.6g} km/s")

            # Resample the template onto the SAME grid as the observation, so
            # the cross-correlation is equal-length and symmetric. The legacy
            # code truncated the template to a shifted sub-grid, which made
            # only negative velocities recoverable (see KNOWN_TRAPS.md). A
            # single-order template (e.g. a full-range library spectrum) is
            # reused for every order, as in CCF2d.
            It = I if len(template.model.wv) > 1 else 0
            spT = dsp.resample(np.asarray(template.model.wv[It]),
                               np.asarray(template.model.sp[It]), w)

            # A template that is flat over this order (e.g. it does not cover
            # it) has nothing to correlate: that order becomes NaN, as before,
            # instead of aborting the whole epoch.
            try:
                v_kms, C = unicor_ccf.ccf_vs_velocity(spT, s, dv_kms, vmax_kms)
            except ValueError as err:
                warnings.warn(f"order {I}: {err}; its CCF is set to NaN", stacklevel=2)
                K = min(int(round(vmax_kms / dv_kms)), len(w) - 1)
                v_kms = np.arange(-K, K + 1) * dv_kms
                C = np.full(len(v_kms), np.nan)

            if V is None:
                V = v_kms << (u.km / u.s)
                corr = np.full((Nord, len(v_kms)), np.nan)

            # Find the radial velocity by fitting a parabola to the CCF peak.
            # An order whose peak cannot be located (e.g. at the edge of the
            # velocity scan) is reported as NaN rather than failing the epoch.
            try:
                vPeak, ccfPeak = unicor_ccf.subpixel_peak(v_kms, C)
                vPeakErr = (unicor_ccf.rv_uncertainty(v_kms, C, vPeak, n_eff[I])
                            if err_per_ord else np.nan)
            except (ValueError, np.linalg.LinAlgError):
                vPeak, vPeakErr, ccfPeak = np.nan, np.nan, np.nan

            corr[I, :] = C
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
            'n_eff': n_eff}
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
                independent bins.

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
                'n_eff': n_eff_total}

        except (ValueError, np.linalg.LinAlgError):
            self.CorrCombined = {
                'vel': velocities,
                'corr': CombinedCorr,
                'RV': np.nan,
                'eRV': np.nan,
                'units': '',
                'peakCorr': np.nan,
                'n_eff': n_eff_total}

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