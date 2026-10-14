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
# Last update: Avraham Binnenfeld, 20210510.


import numpy as np
from astropy import constants as consts, units as u
import matplotlib.pyplot as plt
from copy import deepcopy
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

        :return: self.Corr - a dictionary with the following fields:
                         'vel' - velocity vector at which correlation
                                 was calculated.
                         'corr'- correlation matrix,
                                 (# orders X length of velocity vector)
                         'RV'  - Derived radial velocity
                         'eRV' - Corresponding uncertainty.
                         'peakCorr' - Corresponding CCF peak.
        '''

        # Initialize:
        # ----------
        template = deepcopy(template_in)
        spec = deepcopy(spec_in)

        if dv is None:
            try:
                dv = self.Info['GridDelta']
            except (AttributeError, KeyError):
                dv = self.default_dv
        elif type(dv) is not u.quantity.Quantity:
            dv = float(dv) << u.kilometer / u.second

        if type(VelBound) is not u.quantity.Quantity:
            VelBound = np.array(VelBound) << dv.unit

        # Symmetric velocity half-range for the CCF scan (both signs valid).
        vmax = np.max(np.abs(VelBound))

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
        V, corr = None, None

        for I, w in enumerate(spec.wv):
            s = spec.sp[I]

            # Resample the template onto the SAME grid as the observation, so
            # the cross-correlation is equal-length and symmetric. The legacy
            # code truncated the template to a shifted sub-grid, which made
            # only negative velocities recoverable (see KNOWN_TRAPS.md).
            spT = dsp.resample(np.asarray(template.model.wv[I]),
                                        np.asarray(template.model.sp[I]), w)

            v_kms, C = unicor_ccf.ccf_vs_velocity(spT, s, dv_kms, vmax_kms)

            if V is None:
                V = v_kms << (u.km / u.s)
                corr = np.full((Nord, len(v_kms)), np.nan)

            # Find the radial velocity by fitting a parabola to the CCF peak.
            try:
                if err_per_ord:
                    vPeak, vPeakErr, ccfPeak, _ = self.extract_RV(V, C, n_ord=1)
                else:
                    vPeak, ccfPeak = self.subpixel_CCF(V, C)
                    vPeakErr = np.nan
            except Exception:
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
            'peakCorr': SpecCorr}
        self.n_ord = Nord

        return self

# =============================================================================
# =============================================================================
    def CombineCCFs(self):
        '''
        This routine takes a matrix of CCF values (# orders X length of velocity vector)
        that were calculated by the CrossCorrelateSpec routine, combines the CCFs into
        a single one, based on a maximum-likelihood approach (Zucker, 2003, MNRAS).
        The RV is derived from the peak of the cross-coreelation, and the uncertainty
        is calculated as well

        :param: none.
        :return: A 'CorrCombined' dictionary, with derived combined correlation,
                derived velocity and uncertainty. The structure is similar
                to the 'Corr' dictionary.

        NOTE: the number of lags is assumed to be identical for all orders
        '''

        velocities = self.Corr['vel']
        CombinedCorr = unicor_ccf.combine_ccfs(self.Corr['corr'])

        try:
            V, eRV, CorrPeak, vUnit = self.extract_RV(velocities, CombinedCorr)

            # Return the corresponding velocity grid.
            self.CorrCombined = {
                'vel': velocities,
                'corr': CombinedCorr,
                'RV': V,
                'eRV': eRV,
                'units': vUnit,
                'peakCorr': CorrPeak}

        except Exception:
            self.CorrCombined = {
                'vel': velocities,
                'corr': CombinedCorr,
                'RV': np.nan,
                'eRV': np.nan,
                'units': '',
                'peakCorr': np.nan}

        return self

# =============================================================================
# =============================================================================
    def extract_RV(self, x, y, vel=None, n_ord=None):
        """
        Radial velocity and its maximum-likelihood uncertainty from a CCF.

        :param x: velocity lags (Quantity or km/s array).
        :param y: CCF values at ``x``.
        :param vel: if given, the uncertainty is evaluated at this velocity
                    (km/s) instead of at the fitted peak, and no peak value
                    is returned.
        :param n_ord: number of orders behind ``y``; defaults to ``self.n_ord``.
        :return: (RV, eRV, ccf_peak, unit) -- velocities in the unit of ``x``.
        """
        x_kms = x.value if isinstance(x, u.Quantity) else np.asarray(x, float)
        if vel is None:
            RV, ccf_value = unicor_ccf.subpixel_peak(x_kms, y)
        else:
            RV, ccf_value = vel, np.nan

        Nord = self.n_ord if n_ord is None else n_ord
        eRV = unicor_ccf.rv_uncertainty(x_kms, y, RV, Nord)

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
        :param n_ord: number of orders behind ``y``; defaults to ``self.n_ord``.
        :return: (BIS, eBIS, unit); (nan, nan, None) when a level does not cut
                 the CCF exactly twice.
        """
        Nord = self.n_ord if n_ord is None else n_ord
        BIS, eBIS = unicor_ccf.bisector_span(x.value, y, Nord, levels=bisect_val)
        if np.isnan(BIS):
            return np.nan, np.nan, None
        return BIS, eBIS, x.unit

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