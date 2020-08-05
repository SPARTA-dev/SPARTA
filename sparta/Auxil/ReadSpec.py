#                   -----------------------------------
#                     ReadSpec.py (SPARTA Auxil class)
#                   -----------------------------------
# This file defines the "ReadSpec" class. An object of this class reads and stores
# fits data from various types of instruments and stores it
# into a Spectrum class. Currently, the following spectrum of the following
# instruments are supported: HARPS, APOGEE, NRES, ELODIE, LAMOST ...
#
# A ReadSpec class stores the following methods:
# ---------------------------------------------
#
# 1) load_spectrum_from_fits - loads visit FITS file and parses it to fit the UNICOR Spectrum class.
# 2) retrieve_all_spectrum_parameters - returns the visit parameters read.
# 3) APOGEE_masking - removes bad pixels from an APOGEE spectra.
#
# Dependencies: numpy, astropy, easygui, pandas and datetime.
# Last update: Avraham Binnenfeld, 20200316.

import numpy as np
import easygui
from astropy.io import fits
from datetime import datetime
import pandas as pd

class ReadSpec:

    # =============================================================================
    # =============================================================================
    def __init__(self, **kwargs):
        '''
        Input: All input is optional, and needs to be called along
               with its keyword. Below appears a list of the possible input
               variables.

              s: spectrum vector
              w: wavelength vector
              bool_mask: str, APOGEE bool mask as described here: https://www.sdss.org/dr15/algorithms/bitmasks/#APOGEE_PIXMASK
              DATE-OBS: str, observation date from FITS file
              vrad: float, APOGEE given radial velocity from FITS file
              segment: int. 0, 1 or 2 (choosing 1 APOGEE "color" data band)
        '''

        if 'survey' in kwargs:
            self.survey = kwargs['survey']
            self.read_function = None

        if 'read_function' in kwargs:
            self.survey = None
            self.read_function = kwargs['read_function']

        if 's' and 'w' and 'bool_mask' and 'DATE-OBS' and 'vrad' and 'segment' in kwargs:
            self.s = kwargs['s']
            self.w = kwargs['w']
            self.bool_mask = kwargs['bool_mask']
            self.DATE_OBS = kwargs['DATE-OBS']
            self.vrad = kwargs['vrad']
            self.segment = kwargs['segment']
        else:
            self.s = []
            self.w = []
            self.bool_mask = []
            self.vrad = []
            self.segment = []

        self.snr = -1

        self.bcv = []

        self.n_orders = []

        self.metadata = None

# =============================================================================
# =============================================================================
    def load_spectrum_from_fits(self, path=None, APOGEE_segment=1, min_wv=4900, max_wv=5150):
        '''
        This function loads visit FITS file and parses it to fit the UNICOR Spectrum class.

        INPUT:
        path: visit FITS file path, optional.
        In case no path was provided, th file can be selected using easygui.
        '''
        #  loading a spectrum
        if path:
            fits_fname = path
        else:
            fits_fname = easygui.fileopenbox(msg=None, title='Select data:',
                                               filetypes=None, multiple=False)

        self.APOGEE_segment = APOGEE_segment

        hdul_sp = fits.open(fits_fname)

        if self.read_function is not None:
            self.DATE_OBS, self.s, self.w, \
                self.n_orders, self.bcv, self.metadata = self.read_function(hdul_sp)

        elif self.survey == "APOGEE":
            self.n_orders = 1
            s = hdul_sp[1].data[self.APOGEE_segment]
            w = hdul_sp[4].data[self.APOGEE_segment]

            self.s = np.reshape(s, (1, -1))
            self.w = np.reshape(w, (1, -1))

            # self.DATE_OBS = hdul_sp[0].header['DATE-OBS']
            # self.DATE_OBS = datetime.strptime(self.DATE_OBS, '%Y-%m-%dT%H:%M:%S.%f')
            self.DATE_OBS = hdul_sp[0].header['JD-MID']

            self.vrad = hdul_sp[0].header['VRAD']

            # loading its bool mask data
            flag_s = hdul_sp[3].data[self.APOGEE_segment]
            get_bin_s = lambda x, n: format(x, 'b').zfill(n)
            bool_mask_s = ['0000000000000000' for _ in range(len(flag_s))]
            for i, b in enumerate(flag_s):
                x = get_bin_s(b, 16)
                bool_mask_s[i] = x

            self.bool_mask = bool_mask_s

            self.APOGEE_masking()

        elif self.survey == "HARPS":
            self.n_orders = 1

            s = hdul_sp[1].data['FLUX']
            w = hdul_sp[1].data['WAVE']

            bcv = hdul_sp[0].header["HIERARCH ESO DRS BERV"]

            self.s = s
            self.w = w

            # self.DATE_OBS = hdul_sp[0].header['DATE-OBS']
            # self.DATE_OBS = datetime.strptime(self.DATE_OBS, '%Y-%m-%dT%H:%M:%S.%f')
            self.DATE_OBS = hdul_sp[0].header['HIERARCH ESO DRS BJD']

            min_index = np.where(w[0] > min_wv)[0][0]
            max_index = np.where(w[0] > max_wv)[0][0]

            self.s = s[0, min_index:max_index]
            self.w = w[0, min_index:max_index]

            self.s = np.reshape(self.s, (1, -1))
            self.w = np.reshape(self.w, (1, -1))

            self.bcv = bcv

            self.snr = hdul_sp[0].header["SNR"]

        elif self.survey == "LAMOST":
            self.n_orders = 1

            # http://dr1.lamost.org/doc/data-production-description#toc_3

            s = hdul_sp[0].data[0]

            w = s.copy()
            w[0] = 10 ** hdul_sp[0].header["COEFF0"]
            beta = hdul_sp[0].header["COEFF1"]

            for i in range(len(w) - 1):
                w[i + 1] = w[0] * (1 + beta) ** i

            HELIO_RV = hdul_sp[0].header["HELIO_RV"]

            self.s = s
            self.w = w

            self.DATE_OBS = hdul_sp[0].header["DATE-OBS"]
            self.DATE_OBS = datetime.strptime(self.DATE_OBS, '%Y-%m-%dT%H:%M:%S')

            min_index = np.where(w > 4000)[0][0]
            max_index = np.where(w > 5000)[0][0]

            self.s = s[min_index:max_index]
            self.w = w[min_index:max_index]

            self.s = np.reshape(self.s, (1, -1))
            self.w = np.reshape(self.w, (1, -1))

        elif self.survey == "ELODIE":
            self.n_orders = 1

            s = hdul_sp[0].data

            w = s.copy()
            w[0] = hdul_sp[0].header["CRVAL1"]
            beta = hdul_sp[0].header["CDELT1"]

            self.snr = hdul_sp[0].header["SN"]

            for i in range(len(w) - 1):
                w[i + 1] = w[i] + beta


            self.s = s
            self.w = w

            self.DATE_OBS = hdul_sp[0].header["DATE-OBS"]
            self.DATE_OBS = datetime.strptime(self.DATE_OBS, '%Y-%m-%dT%H:%M:%S')

            self.s = np.reshape(self.s, (1, -1))
            self.w = np.reshape(self.w, (1, -1))

        elif self.survey == "NRES":
            self.n_orders = 67

            s = []
            w = []
            for ordind in np.arange(self.n_orders):
                s.append(np.array(hdul_sp[2].data[ordind]))
                w.append(10* np.array(hdul_sp[6].data[ordind])) # NRES gives the data in nm. Convert to Ang.

            self.bcv = hdul_sp[0].header["RVCC"] - hdul_sp[0].header["RCC"]*299792.458

            self.s = s
            self.w = w

            self.DATE_OBS = hdul_sp[0].header["BJD"]


        elif self.survey == "TRES":
            self.n_orders = 1

            s = hdul_sp[0].data[36]
            w = [i for i in range(len(s))]
            self.s = s
            self.w = w

            self.DATE_OBS = hdul_sp[0].header["WS_BJD"]
            self.bcv = hdul_sp[0].header["BCV"]
            # self.DATE_OBS = datetime.strptime(self.DATE_OBS, '%Y-%m-%dT%H:%M:%S')

            min_index = np.where(np.array(w) > 500)[0][0]
            max_index = np.where(np.array(w) > 1000)[0][0]

            self.s = s[min_index:max_index]
            self.w = w[min_index:max_index]

            self.s = np.reshape(self.s, (1, -1))
            self.w = np.reshape(self.w, (1, -1))

        elif self.survey == "UVES":

            self.n_orders = 1

            s = hdul_sp[1].data[0][4]

            w = hdul_sp[1].data[0][0]

            self.snr = hdul_sp[0].header["SNR"]

            self.s = s # - s.min() + 1
            self.w = w

            self.DATE_OBS = hdul_sp[0].header["MJD-OBS"]
            # self.DATE_OBS = datetime.strptime(self.DATE_OBS, '%Y-%m-%dT%H:%M:%S.%f')

            min_index = np.where(np.array(w) > 6500)[0][0]
            max_index = np.where(np.array(w) > 6600)[0][0]

            self.s = s[min_index:max_index]
            self.w = w[min_index:max_index]

            self.s = np.reshape(self.s, (1, -1))
            self.w = np.reshape(self.w, (1, -1))


# =============================================================================
# =============================================================================
    def retrieve_all_spectrum_parameters(self):
        """
        This function returns the visit parameters read.
        :return: all class features:
        w: wavelength vector
        s: spectrum vector
        bool_mask: str, APOGEE bool mask as described here: https://www.sdss.org/dr15/algorithms/bitmasks/#APOGEE_PIXMASK
        DATE-OBS: str, observation date from fits file
        vrad: float, radial velocity given in the fits file
        """
        return self.w, self.s, self.bool_mask, self.DATE_OBS, self.vrad, self.bcv, self.snr

# =============================================================================
# =============================================================================
    def APOGEE_masking(self):
        '''
        This function removes bad pixels (marked as sky flux or telluric line) from an APOGEE spectra
        and fills them by using linear interpolation
        APOGEE bool mask as described here: https://www.sdss.org/dr15/algorithms/bitmasks/#APOGEE_PIXMASK)
        '''

        if self.survey != "APOGEE":
            assert False, "APOGEE_masking function only takes APOGEE spectra"

        spR = pd.Series(self.s[0])
        for i, flag in enumerate(self.bool_mask):
            if flag[3] == '1' or flag[2] == '1':  # sky flux and telluric line reduction
                spR[i] = float('nan')
        spR.interpolate(inplace=True)

        spR.replace(to_replace=float('nan'), value=0, inplace=True)

        self.s[0] = np.array(spR.tolist())

        return self
