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
# The velocity-space cross-correlation itself lives in sparta.spectral_ops
# (ccf_vs_velocity); this class handles multi-order orchestration, RV/error
# extraction, combination, and BIS.
#
# Conventions: radial velocities in km/s; the CCF is computed on a log-lambda
# (velocity-even) grid.
#
# Dependencies: scipy, numpy, astropy, matplotlib and copy.
# Last update: Avraham Binnenfeld, 20210510.


from scipy import interpolate
import numpy as np
from astropy import constants as consts, units as u
import matplotlib.pyplot as plt
from copy import deepcopy
from sparta import spectral_ops

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
            spT = spectral_ops.resample(np.asarray(template.model.wv[I]),
                                        np.asarray(template.model.sp[I]), w)

            v_kms, C = spectral_ops.ccf_vs_velocity(spT, s, dv_kms, vmax_kms)

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

        # Arrage the correlation matrix. Clip to [-1, 1] so a per-order value
        # that overshoots numerically cannot drive 1 - C**2 negative (which
        # would make the combined CCF NaN).
        CorrMat = np.clip(self.Corr['corr'], -1.0, 1.0)
        velocities = self.Corr['vel']

        # Read the number of orders in the spectrum
        Nord = self.n_ord

        # Combine the CCFs according to Zucker (2003, MNRAS), section 3.1
        CombinedCorr = np.sqrt(1-(np.prod(1-CorrMat**2, axis=0))**(1/Nord))

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
        Get the radial velocity and its uncertainty from maximum likelihood.
        If velocity is given, the uncertainty at this specific point is calculated.
        """
        if vel is None:
            RV, ccf_value = self.subpixel_CCF(x, y)
        else:
            RV = self.subpixel_CCF(x, y, v=vel)
            ccf_value = np.nan

        if n_ord is None:
            Nord = self.n_ord
        else:
            Nord = n_ord

        Nvels = len(x)

        # Generate the second derivative from spline
        spl = interpolate.UnivariateSpline(x.value, y, k=4, s=0)
        spl_vv = spl.derivative(n=2)

        # Calculate the uncertainty. The variance is -1 / (ml_factor*Nord*Nvels)
        # and is only real at a true CCF maximum (ml_factor < 0); guard the
        # degenerate cases rather than emitting a silent NaN from sqrt().
        ml_factor = spl_vv(RV) * spl(RV) / (1 - spl(RV)**2)
        denom = ml_factor * Nord * Nvels
        var = -1.0 / denom if denom != 0 else np.nan
        eRV = np.sqrt(var) if (np.isfinite(var) and var > 0) else np.nan

        if type(x) is u.quantity.Quantity:
            xUnit = x.unit
        else:
            xUnit = None

        return RV, eRV, ccf_value, xUnit

# =============================================================================
# =============================================================================
    def calcBIS(self, x, y, bisect_val=[0.35, 0.95], n_ord=None):
        """
        Determine full-with-half-maximum of a peaked set of points, x and y.
        Assumes that there is only one peak present in the datasset.
        The function uses a spline interpolation of order k.
        """
        y_low = np.max(y)*bisect_val[0]
        y_high = np.max(y)*bisect_val[1]

        s_low = interpolate.splrep(x.value, y - y_low)
        s_high = interpolate.splrep(x.value, y - y_high)

        roots_low = interpolate.sproot(s_low, mest=2)
        roots_high = interpolate.sproot(s_high, mest=2)

        if (len(roots_low) == 2) and (len(roots_high) == 2):
            low = [self.subpixel_CCF(x, y, v=roots_low[0]),
                   self.subpixel_CCF(x, y, v=roots_low[1])]
            high = [self.subpixel_CCF(x, y, v=roots_high[0]),
                    self.subpixel_CCF(x, y, v=roots_high[1])]

            _, err, _, _ = self.extract_RV(x, y, n_ord=n_ord)

            BIS = (low[1]+low[0])/2 - (high[1]+high[0])/2
            eBIS = np.sqrt(2)*err

            return BIS, eBIS, x.unit
        else:
            return np.nan, np.nan, None

    def subpixel_CCF(self, vels, ccf, v=None, Npts=5):
        """
        This function is using a second order approximation to estimate the ccf
        at a given velocity, v. If no velocity was provided, the CCFs peak velocity
        is returned.

        :param vels: velocity array for a CCF at a given order.
        :param ccf: CCF values that correspond with the velocities in x.
        :return: Tuple containing parameters:x_max, y_max
        """

        if type(vels) is u.quantity.Quantity:
            vels = vels.value

        if v is None:
            assert Npts >= 3, "Must have at least 3 points."
            assert Npts % 2 == 1, "Provide an odd number of points to fit around the peak."

            x_n = np.argmax(ccf)
            indlist =[int(x_n - Npts//2 + k) for k in np.arange(Npts)]
            x = np.array(vels[indlist])
            y = np.array(ccf[indlist])

            # Define the design matrix at the given phases.
            DesignMatrix = np.array(
                [[1, x, x**2]
                 for x in x])

            # Solve to obtain the parameters and uncertainties
            C = (np.linalg.inv(
                np.dot(DesignMatrix.transpose(), DesignMatrix)))

            # Derive the parameters:
            pars = C.dot(np.dot(DesignMatrix.transpose(), y))
            y_max = pars[0] - pars[1] * pars[1] / (4 * pars[2])  # maximal CCF result value
            x_max = -pars[1] / (2 * pars[2])  # velocity in maximal value
            return x_max, y_max

        else:
            vels_diff = [i - v for i in vels]
            x_n = np.argmin(np.abs(vels_diff))

            indlist =[int(x_n - 1), int(x_n), int(x_n + 1)]
            x = np.array(vels[indlist])
            y = np.array(ccf[indlist])

            y_interp = (y[0]*(v-x[1])*(v-x[2])/(x[0]-x[1])/(x[0]-x[2]) +
                        y[1]*(v-x[0])*(v-x[2])/(x[1]-x[0])/(x[1]-x[2]) +
                        y[2]*(v-x[0])*(v-x[1])/(x[2]-x[0])/(x[2]-x[1])
                        )
            return y_interp


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