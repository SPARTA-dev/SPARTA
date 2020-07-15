#                   ------------------------------
#                     Spectrum.py (SPARTA UNICOR class)
#                   ------------------------------
# This file defines the "Spectrum" class. An object of this class stores
# the measured spectrum (wavelength values and their corresponding fluxes)
# and some basic information about the measurement: time of observation,
# baricentric velocity correction, etc. See more details under __init__.
# The spectrum can be single- or multi-order.
#
# A Spectrum class stores the following methods:
# ---------------------------------------------
#
# 1) InterpolateSpectrum - resamples the spectrum on a linear or logarithmic scale
# 2) TrimSpec - Cuts the edges of the spectrum (mostly to remove zero paddding)
# 3) FilterSpectrum - A Butterworth bandpass filter
# 4) ApplyCosineBell - Apllies a Tuckey window on the data
# 5) RemoveCosmics - Removes outliers that deviate above the measured spectrum.
# 6) BarycentricCorrection - Preforms a barycentric correction.
# There is also an 'overall' procedure, that calls all the routines with
# some default values:
# 8) SpecPreProccess - Calls all the above a suggested order.
#
# Dependencies: numpy, scipy and astropy.
# Last update: Avraham Binnenfeld, 20200607.

import numpy as np
from scipy import interpolate, signal
from astropy.stats import sigma_clip, mad_std
from astropy import constants as consts, units as u


class Spectrum:

    # =============================================================================
    # =============================================================================
    def __init__(self, wv=[], sp=[], wvUnits=1.0*u.Angstrom,
                 bjd=0.0, bcv=0.0, name='Anonymous'):
        '''
        :param: wv, sp: lists data from a single observation.
                       Each entry in the list represents a different order.
        :param: name: str, the name of the target.
        :param: bcv: float, barycentric correction to the velocity.
        :param: bjd: barycentric julian day.

        '''
        assert len(wv) == len(sp), "Dimensions mismatch (wv vs. sp)"
        self.c = consts.c.to('km/s') # The speed of light in km/sec
        self.Info = {'GridType': 'raw',
                     'WavelengthUnits': wvUnits,
                     'TargetName': name}
        self.wv = wv
        self.sp = sp
        self.bjd = bjd
        self.bcv = bcv
        self.name = name
        self.n_ord = len(self.sp)

        if type(bcv) is not u.quantity.Quantity:
            self.bcv = self.bcv*u.kilometer/u.second

# =============================================================================
# =============================================================================
    def SpecPreProccess(self, Ntrim=10, CleanMargins=True, RemoveNaNs=True,
                        delta=0.5, RemCosmicNum=3, FilterLC=3, FilterHC=0.15,
                        alpha=0.3):
        '''
        This function applies the an overall processing, in the
        recommended order.
        '''

        self.TrimSpec(Ntrim=Ntrim, CleanMargins=CleanMargins, RemoveNaNs=RemoveNaNs)
        self.InterpolateSpectrum(delta=delta)
        self.RemoveCosmics(sigma_upper=RemCosmicNum)
        self.FilterSpectrum(lowcut=FilterLC, highcut=FilterHC, order=1)
        self.ApplyCosineBell(alpha=alpha)

        return self

    # =============================================================================
    # =============================================================================
    def InterpolateSpectrum(self, delta=None, InterpMethod='linear'):
        '''
        This function resamples a list of wavelength and spectra
        so that the wavelength values will be evenly spaced.

        :param: self.wv - list of 1D wavelength arrays.
        :param: self.sp - list of 1D flux arrays, correspond the wv arrays.
        :param: delta   - positive float.
                          If InterpMethod is linear, the step of the evenly
                          spaced array is taken to be delta*(min delta lambda).
                          In this case delta is a unitless parameter.
                          If InterpMethod is log, then the array is resampled
                          to be logarithmically evenly-spaed. The step is
                          taken to be log(1+delta/c), where c is the speed of
                          speed of light in km/sec, and so is delta.
        :param: InterpMethod - string. 'linear' to generate an evenly spaced
                               wavelength vector. 'log' to generate a
                               logarithmically evenly-spaced vector.
        :return: self.wv - list of evenly spaced wavelegth arrays.
        :return: self.sp - list of flux values, interpolated at wvI.

        Notes: The input, wv and sp, are data from a single observation.
               Each entry in the list represents a different order.
               The step size of the interpolated wavelength array is set
               to be f times the minimal (non zero, positive) wavelength
               step in the input data.-
        '''
        # Initialize data
        wvI, spI, wnt = [], [], self.Info['WavelengthUnits']

        # Set the interpolation deltas
        if InterpMethod == 'linear':
            if delta is None:
                delta = 0.5 * wnt
            elif type(delta) is not u.quantity.Quantity:
                delta = float(delta) * wnt
            deltawv = ((delta/wnt).decompose()).value

        if InterpMethod == 'log':
            if delta is None:
                delta = 0.1 * u.kilometer / u.second
            elif type(delta) is not u.quantity.Quantity:
                delta = float(delta) * u.kilometer / u.second
            betavel = ((delta/self.c).decompose()).value

        # Interpolate per order
        for I, w in enumerate(self.wv):

            # Calculate the median wavelength difference.
            medianDiff = np.median(np.diff(w))

            # If the grid is interpolated in wavelength:
            if InterpMethod == 'linear':
                dlam = deltawv*medianDiff
                wvI.append(np.arange(min(w) + abs(dlam), max(w) - abs(dlam), abs(dlam)))

            # If the grid is interpolated in velocity
            elif InterpMethod == 'log':
                dlam = np.log(1+betavel)
                wvI.append(np.exp(
                           np.arange(np.log(min(w)+medianDiff),
                                     np.log(max(w)-medianDiff),
                                     dlam)))

            # Inpterpolate the flux over the calculated grid
            InterpF = interpolate.interp1d(w, self.sp[I], kind='quadratic')
            spI.append(InterpF(wvI[I]))

        # Replace the data with the interpolated wavelengths
        self.wv, self.sp = wvI, spI

        # Document the type of grid that is currently
        # saved with the function.
        if InterpMethod == 'log':
            self.Info['GridType'], self.Info['GridDelta'], self.Info['GridUnits'] = 'log', delta, 'velocity'
        else:
            self.Info['GridType'], self.Info['GridDelta'], self.Info['GridUnits'] = 'linear', delta, 'wavelength'

        return self

    # =============================================================================
    # =============================================================================
    def TrimSpec(self, Ntrim=None ,NtrimLeft=0, NtrimRight=0,
                 CleanMargins=False, RemoveNaNs=True):
        '''
        In some cases the spectra is zero-padded at the edges, which may affect
        the filtering stage. This routine trims the edges of the data.
        The number of trimmed indices is given as Ntrim.
        If Ntrim is not provided, all points that equal to zero,
        are removed.

        :param:  self.wv - list of 1D wavelength arrays.
        :param: self.sp - list of 1D flux arrays, correspond the wv arrays.
        :param: Ntrim   - Integer. Number of points to remove from each side
                          of the spectrum.
        :return: self.wv - list of wavelegth arrays, trimmed.
                self.sp - list of flux values, trimmed.
        '''
        # Initialize data
        wvT, spT = [], []

        if Ntrim is not None:
            NtrimLeft, NtrimRight = Ntrim, Ntrim

        # Interpolate per order
        for I, w in enumerate(self.wv):
            s = self.sp[I]

            if RemoveNaNs:
                w = w[~np.isnan(s)]
                s = s[~np.isnan(s)]

            if CleanMargins:
                non_zero_loc = np.nonzero(self.sp[I])
                first_non_zero = np.min(non_zero_loc)
                last_non_zero = np.max(non_zero_loc)
            else:
                first_non_zero = 0
                last_non_zero = len(w)

            s = s[(first_non_zero+NtrimLeft):(last_non_zero-NtrimRight)]
            w = w[(first_non_zero+NtrimLeft):(last_non_zero-NtrimRight)]

            spT.append(s)
            wvT.append(w)

        self.wv, self.sp = wvT, spT
        return self

    # =============================================================================
    # =============================================================================
    def FilterSpectrum(self, lowcut, highcut, order):
        '''
        This function applies a Butterworth fitler to the input
        spectrum, removes the low-pass instrumental repsponse and the high-pass
        instrumental noise.

        :param: self.sp - list of 1D flux arrays, correspond to the wv arrays.
        :param: lowcut  - float. Stopband freq for low-pass filter. Given in
                          units of the minimal frequency (max(w)-min(w))**(-1)
        :param: highcut - float. Stopband freq for the high-pass filter. Given
                          in units of the Nyquist frequency.
        :param: order -   integer. The order of the Butterworth filter.

        :return: spF - list of filtered flux values at wv.
        '''
        # Initialize data
        spF = []

        # Filter the spectrum per order
        for I, w in enumerate(self.wv):
            # the bandpass values are normalized to the Nyquist frequency.
            # here we assume that the spectrum in evenly sampled, therefore:
            nyq = 0.5/np.abs(w[1]-w[0])

            # The minimal frequency is set by the full range of the datapoints
            # here we use express the minimal frequency in terms of the Nyquist
            df = (max(w)-min(w))**(-1)/nyq

            # The highpass frequency is already given in terms of the Nyquist
            # the lowpass filter is provided in terms of the minimal frequency.
            # it is therefore expressed by the Nyquist frequency.
            low = lowcut*df

            # Define the Butterworth filter
            b, a = signal.butter(order, [low, highcut], btype='band')

            # Winsorize the data
            s = signal.filtfilt(b, a, self.sp[I]-np.percentile(self.sp[I], 80),
                                padlen=10*max(len(b), len(a)))

            sLen = len(s)
            s = (s-np.mean(s))/np.std(s)  #(np.sum((s-np.mean(s))**2)/sLen)

            # Filter the data
            spF.append(s)

        self.sp = spF
        return self

    # =============================================================================
    # =============================================================================
    def ApplyCosineBell(self, alpha=0.1):
        '''
        This function applies a cosine bell (Tukey Window) to the spectrum,

        :param:  self.wv - list of 1D wavelength arrays (evenly sampled).
        :param:  self.sp - list of 1D flux arrays, correspond to the wv arrays.
        :param:  alpha -   (optional) float. Shape parameter of the Tukey win.

        :return: spC - list of flux values at wv,
                      after the cosine bell was applied
        '''
        # Initialize data
        spC = []

        # Interpolate per order
        for I, w in enumerate(self.wv):
            window = signal.tukey(len(w), alpha)
            spC.append(
                    np.multiply(self.sp[I], window))

        self.sp = spC
        return self

# =============================================================================
# =============================================================================
    def RemoveCosmics(self, **kwargs):
        '''
        This function removes positive outliers, i.e., points that
        significantly deviate above the measured spectra. Significant deviation
        is defined as points that deviate more than (sigma_upper)X(1.48mad_std)
        from the main part of the spectrum. Points that deviate below are
        not removed.

        :param: self.wv - list of 1D wavelength arrays (evenly sampled).
        :param: self.sp - list of 1D flux arrays, correspond to the wv arrays.
        :param: sigma_upper - float. Number of sigma beyond which points will
                              be removed. (Optional. default=3).

        :return: spR - list of flux values at wvR,
                      after exclusion of the outliers.
        '''

        # Initialize data
        if 'sigma_upper' in kwargs:
            sigma_upper = kwargs['sigma_upper']
        else:
            sigma_upper = 3

        spR = []
        wvR = []

        # Interpolate per order
        for I, s in enumerate(self.sp):
            filtered_data = sigma_clip(s, sigma_lower=np.Inf,
                                       sigma_upper=sigma_upper,
                                       masked=True, cenfunc='median',
                                       stdfunc=mad_std)

            spR.append(np.array([x for i, x in enumerate(s) if filtered_data[i]]))
            wvR.append(np.array([x for i, x in enumerate(self.wv[I]) if filtered_data[i]]))

        self.wv, self.sp = wvR, spR
        return self

# =============================================================================
# =============================================================================
    def BarycentricCorrection(self):
        '''
        Apply a barycentric correction to the wavelength vectors
        of each order. Once it is done, the bcv value in the structure
        is set to be 0.0 km/s.
        '''

        # Set Barycentric velocity
        betavel = ((self.bcv/self.c).decompose()).value
        wvBC = []
        # Preform barycentric correction
        for I, w in enumerate(self.wv):
            # wvBC.append(np.reshape(w*(1+betavel), (1, -1)))
            wvBC.append(np.squeeze(w*(1+betavel)))

        # Update the BCV to be zero, so that another activation
        # of the function wont mess up the results.
        self.wv = wvBC
        self.bcv = 0.0 * u.kilometer / u.second

        return self

    # =============================================================================
    # =============================================================================
    def SelectOrders(self, orders, remove=True):
        '''
        This routine enables the user to keep or discard orders from the observed
        data. Orders are numbered according to their index (0-> #ord-1),

        :param orders: Indices of the required orders
        :param include: Boolean. If True, these orders will be removed.
                        otherwise, these will be the orders that will
                        be kept.
        :return: Same structure, with the orders arranged.
        '''
        assert (np.max(orders) < self.n_ord), "Invalid order numbers"
        assert (np.min(orders) >= 0), "Invalid order numbers"

        if remove:
            x = np.setdiff(np.arange(self.n_ord), orders)
        else:
            x = orders

        wv = [self.wv[I] for I in np.sort(x)]
        sp = [self.sp[I] for I in np.sort(x)]

        self.wv = wv
        self.sp = sp
        self.n_ord = len(self.wv)

        return self

