#                   ----------------------------------
#                      CCF1d.py (SPARTA UNICOR class)
#                   ----------------------------------
# This file defines the "CCF1d" class. An object of this class stores
# a Spectrum object, saved in the self.spec field and a Template object
# stored in the self.template field.
#
# A Template class has the following methods (under construction):
# CrossCorrelateSpec - multi-order cross correlation.
# CombineCCFs - sums the CCFs of a multi-order spectrum
# TBD
#
# Dependencies: Spectrum and Template class (of UNICOR).
#               scipy and numpy.
#
# Last update: Sahar Shahaf, 20200718.


from scipy import interpolate
from scipy.signal import correlate
import numpy as np
from astropy import constants as consts, units as u
from numba import njit
import matplotlib.pyplot as plt
import copy


class CCF1d:
    # =============================================================================
    # =============================================================================
    def __init__(self):
        '''
        No input required.
        Some defaults are set...
        '''
        self.c = consts.c.to('km/s') # The speed of light in km/sec
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
        spec = copy.deepcopy(spec_in)
        template = copy.deepcopy(template_in)

        if dv is None:
            try:
                dv = self.Info['GridDelta']
            except:
                dv = self.default_dv

        elif type(dv) is not u.quantity.Quantity:
            dv = float(dv) << u.kilometer / u.second

        if type(VelBound) is not u.quantity.Quantity:
            VelBound = np.array(VelBound) << dv.unit

        if VelBound.size == 2:
            Vi, Vrange = np.min(VelBound), np.abs(np.diff(VelBound))
        elif VelBound.size == 1:
            Vi, Vrange = -np.abs(VelBound), np.abs(2*VelBound)

        # In case that the spectum is not logarithmically spaced,
        # it must be interpolated to a logarithmically evenly-spaced
        # grid. The parameter of the grid is dv [km/s].
        if ('GridType' not in spec.Info) or ('GridDelta' not in spec.Info) or spec.Info['GridType'] == 'linear':
            spec.InterpolateSpectrum(delta=dv, InterpMethod='log')

        # If the data is already logarithmically spaced, then read the dv
        # of the wavelegth grid (used to set the velocity axis of the CCF)
        elif spec.Info['GridType'] == 'log':
            dv = spec.Info['GridDelta']
        else:
            if spec.Info['GridType'] == 'linear':
                spec.InterpolateSpectrum(dv, InterpMethod='log')
                spec.Info['GridDelta'] = dv
                spec.Info['GridUnits'] = 'velocity'
            elif spec.Info['GridDelta'] != dv:
                spec.InterpolateSpectrum(dv, InterpMethod='log')
                spec.Info['GridDelta'] = dv
                spec.Info['GridUnits'] = 'velocity'

        # The cross correlation is performed on a velociry range defined by
        # the user. The range is converted to a number of CCF lags, using
        # the velocity spacing dv. Default is 100 km/s.
        Nlags = np.floor((Vrange/dv).decompose().value)
        Nord = len(spec.wv)

        # Calculate the velocity from the lags
        V = Vi + dv * np.arange(Nlags+1)

        # Initialize arrays
        corr = np.full((len(spec.wv), len(V)), np.nan)
        RV = np.full((len(spec.wv), 1), np.nan)
        eRV = np.full((len(spec.wv), 1), np.nan)
        SpecCorr = np.full((len(spec.wv), 1), np.nan)

        for I, w in enumerate(spec.wv):
            # In order for the CCF to be normalized to the [-1,1] range
            # the signals must be divided by their standard deviation.
            s = spec.sp[I]

            # Interpolate the template to the wavelength scale of the
            # observations. We assume here that the template is broadened
            # to match the width of the observed line profiles.
            interpX = (np.asarray(template.model.wv[I][:]) *
                       (1+((Vi/self.c).decompose()).value))

            interpY = np.asarray(template.model.sp[I][:])

            InterpF = interpolate.interp1d(interpX,
                                           interpY,
                                           kind='quadratic')

            GridChoice = np.logical_and(w > np.min(interpX),
                                        w < np.max(interpX))
            wGrid = np.extract(GridChoice, w)
            spT = InterpF(wGrid)

            # The data and the template are cross-correlated.
            # We assume that the wavelengths are logarithmically-evenly spaced
            C = self.correlate1d(spT, s, Nlags+1, fastccf=fastccf)

            # Find the radial velocity by fitting a parabola to the CCF peaks
            try:
                if err_per_ord:
                    # Calculate the uncertainty of each order (if required)
                    vPeak, vPeakErr, ccfPeak, vUnit = self.extract_RV(V, C, n_ord=1)
                else:
                    vPeak, ccfPeak = self.subpixel_CCF(V, C)
                    vPeakErr = np.NaN

            except:
                vPeak, vPeakErr, ccfPeak = np.NaN, np.NaN, np.NaN

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

        # Arrage the correlation matrix
        CorrMat = self.Corr['corr']
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

        except:
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

        # Calculate the uncertainty:
        ml_factor = spl_vv(RV) * spl(RV) / (1 - spl(RV)**2)
        eRV = np.sqrt(-(ml_factor*Nord*Nvels)**(-1))

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

# =============================================================================
# =============================================================================
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
            x_n = np.amin(np.abs(vels_diff))

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
    def correlate1d(self, template, signal, maxlag=None, fastccf=False):
        '''
        This is a wrapper of the jitted one-dimensional correlation
        :param template: template (model) that will be compared to the signal.
        :param signal: measured signal.
        :param maxlag: maximum number of lags to calculate.
        :return: the correlation values.
        '''

        if maxlag is None:
            maxlag = 1
        maxlag = np.int(np.minimum(signal.shape[0], maxlag))
        # maxlag = np.int(np.minimum(t.shape[0], maxlag))
        if not fastccf:
            C = __correlate1d__(template, signal, maxlag)
        else:
            C = __correlate1d_fast__(template, signal, maxlag)
        return C

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


@njit
def __correlate1d__(template, signal, maxlag):
    """
    Compute correlation of two signals defined at uniformly-spaced points.

    The correlation is defined only for positive lags. The zero shift is represented as 1 lag.
    The input arrays represent signals sampled at evenly-spaced points.

    Arguments:
    :param template: the template (model) that is compared to the observed signal
    :param signal: the observed signal
    :param maxlag: maximum number of lags to calculate.
    :return: an array with Pearsons correlation for each lag.
    """

    # Initialize an empty array
    C = np.full(maxlag, np.nan)

    # Calculate the cross-correlation
    for lag in range(C.size):
        template_max = np.minimum(signal.size - lag, template.size)
        signal_max = np.minimum(signal.size, template.size + lag)
        C[lag] = np.sum(template[:template_max] * signal[lag:signal_max])

    # Calculate the normalization factor
    normFac = np.sqrt((template**2).sum() * (signal**2).sum())
    return C/normFac


def __correlate1d_fast__(template, signal, maxlag):
    """
    :param template: the template (model) that is compared to the observed signal
    :param signal: the observed signal
    :param maxlag: maximum number of lags to calculate
    :return: an array of Pearsons correlation for each lag
    """
    FC = correlate(signal, template, mode='full', method='fft')
    normFac = np.sqrt((template**2).sum() * (signal**2).sum())
    N = len(template) - 1
    C = FC[N: N+maxlag]
    return C/normFac
