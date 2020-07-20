#                   ----------------------------
#                    Template.py (SPARTA UNICOR class)
#                   ----------------------------
# This file defines the "Template" class. An object of this class stores
# a Spectrum object, saved in the self.model filed. This class has some
# additional attributes added to the ones of the Spectrum class, including
# the ability to downolad, process and store PHOENIX synthetic spectrum.
#
# A syntheticSpectrum class stores the following methods:
# ---------------------------------------------
#
# 1) create_PHOENIX_fname - Prepare the needed spectra files names and paths for downloading
# 2) download_PHOENIX_files - Downloads the synthetic spectrum files from the Goettingen ftp
# 3) doppler - Shifts the synthetic spectrum according to a given velocity (Doppler effect)
# 4) add_noise - Adds simulated noise to the synthetic spectrum according to a given snr
# under construction -
# GaussianBroadening - broadens the template with a Gaussian window,
#                      to account for the instrumental broadening.
# RotationalBroadening - broadens the template with a rotaional profile
#                        to account for stellar rotation.
#
# Dependencies: numpy, astropy, ftplib, pathlib, random and os.
# Last update: Avraham Binnenfeld, 20200607.


from astropy.io import fits
from sparta.UNICOR.Spectrum import Spectrum
import numpy as np
from ftplib import FTP
from pathlib import Path
import random
import os
from PyAstronomy import pyasl
import pandas as pd
from astropy import units as u


TEMPLATE_FILES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'template_files/'
)


class Template:

    # =============================================================================
    # =============================================================================
    def __init__(self, **kwargs):
        '''
        :param: The following stellar parameters are necessary for PHOENIX spectrum definition:
                temp, log_g, metal, alpha.

        :param: min_val - minimal value for wavelength range (Angstrom)
        :param: min_val - maximal value for wavelength range (Angstrom)

        alternatively:
        1) Template spectrum and wavelengths can be passed on to the class and will be saved as-is.
        2) A ready-for-use Spectrum object can be passed on to the class and will be saved as-is.
        '''
        if 'template' in kwargs:
            self.model = kwargs['template']
            if 'vel' in kwargs:
                self.vel = kwargs['vel']
            else:
                self.vel = []

        elif 'temp' and 'log_g' and 'metal' and 'alpha' in kwargs:
            self.temp = kwargs['temp']
            self.log_g = kwargs['log_g']
            self.metal = kwargs['metal']
            self.alpha = kwargs['alpha']
            self.min_val = kwargs['min_val']
            self.max_val = kwargs['max_val']
            self.server_path = r"phoenix.astro.physik.uni-goettingen.de"
            self.wl_f_n = 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'

            p, f_n = self.create_PHOENIX_fname(self.temp, self.log_g, self.metal, self.alpha)

            if not os.path.exists(r'template_files'):
                os.makedirs(r'template_files')

            if not Path(TEMPLATE_FILES_PATH + self.wl_f_n).exists():
                self.download_PHOENIX_files(p='HiResFITS/', f_n=self.wl_f_n)

            if not Path(TEMPLATE_FILES_PATH + f_n).exists():
                self.download_PHOENIX_files(p, f_n)

            # Open the fits file using fits module from astropy.
            wl_fname = self.wl_f_n
            spec_fname = f_n

            hdul_spec = fits.open(
                os.path.join(TEMPLATE_FILES_PATH, spec_fname)
            )

            s = np.squeeze(np.array(hdul_spec[0].data))

            hdul_wl = fits.open(
                os.path.join(TEMPLATE_FILES_PATH, wl_fname)
            )

            w = np.squeeze(np.array(hdul_wl[0].data))

            # in case of loading low-res PHOENIX spectrum,
            # which has to be downloaded an extracted manually,
            # the medium res wv vector is computed in the following way:

            # ----- START of low-res insertion -------

            # w = s.copy()
            # w[0] = np.e ** hdul_spec[0].header["CRVAL1"]
            # beta = hdul_spec[0].header["CDELT1"]

            # for i in range(len(w) - 1):
            #     w[i + 1] = w[0] * (1 + beta) ** i

            # ------- END of low-res insertion ---------

            # star mass g

            # Change to air wavelengths, if required
            # The following transformation is taken from Morton (2000, ApJ. Suppl., 130, 403)
            if 'air' in kwargs:
                if kwargs['air']:
                    s2 = 10**8 * w**(-2)
                    f = 1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2)
                    w = w/f

            self.PHXMASS = hdul_spec[0].header['PHXMASS']

            # Effective stellar radius cm
            self.PHXREFF = hdul_spec[0].header['PHXREFF']

            min_index = np.where(w > self.min_val)[0][0]
            max_index = np.where(w > self.max_val)[0][0]

            spectrum = s[min_index:max_index]
            wavelengths = w[min_index:max_index]

            spectrum = np.reshape(spectrum, (1, -1))
            wavelengths = np.reshape(wavelengths, (1, -1))

            spectrum = spectrum / (10 ** 13)

            self.model = Spectrum(wv=wavelengths, sp=spectrum)

        elif 'spectrum' and 'wavelengths' in kwargs:
            spectrum = kwargs['spectrum']
            wavelengths = kwargs['wavelengths']
            if 'min_val' and 'max_val' in kwargs:
                self.min_val = kwargs['min_val']
                self.max_val = kwargs['max_val']

            spectrum = np.reshape(spectrum, (1, -1))
            wavelengths = np.reshape(wavelengths, (1, -1))

            self.model = Spectrum(wv=wavelengths, sp=spectrum)

        else:
            self.model = Spectrum()
            self.vel = []

    # =============================================================================
    # =============================================================================
    def GaussianBroadening(self, **kwargs):
        '''
        This function broadens the spectrum using a given Gaussian kernel.
        :param: resolution - float, The spectral resolution.
        '''

        resolution = kwargs["resolution"]

        if 'wv' and 'sp' in kwargs:
            flux = pyasl.instrBroadGaussFast(np.reshape(kwargs["wv"][0], (1, -1))[0],
                                             kwargs["sp"][0], resolution)
        else:
            flux = pyasl.instrBroadGaussFast(np.reshape(self.model.wv[0], (1, -1))[0],
                                    self.model.sp[0], resolution)

        flux = np.reshape(flux, (1, -1))

        self.model.sp = flux

        return flux

    # =============================================================================
    # =============================================================================
    def RotationalBroadening(self, **kwargs):
        '''
        This function broadens the spectrum to account for for the broadening effect of stellar rotation.
        :param: vsini - float, rotational velocity [km/s]
        :param: epsilon: float, limb-darkening coefficient (0-1)
        '''

        epsilon = kwargs["epsilon"]
        vsini = kwargs["vsini"]

        w = np.reshape(self.model.wv[0], (1, -1))[0]
        s = self.model.sp[0]

        flux = pyasl.rotBroad(w, s, epsilon, vsini)

        self.model.sp = np.reshape(flux, (1, -1))
        # self.model.wv = np.reshape(flux, (1, -1))
        return self

        # =============================================================================
    # =============================================================================
    def create_PHOENIX_fname(self, temp, log_g, metal, alpha):
        '''
        This function prepares the needed spectra files names and paths for downloading
        :param: The following stellar parameters are necessary for PHOENIX spectrum definition:
                temp, log_g, metal, alpha.
        '''
        if metal > 0:
            p = "/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z+{:.1f}/".format(metal, alpha)
            s = "lte{:05d}-{:.2f}+{:.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(temp, log_g, metal)
        elif alpha != 0:
            p = "/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-{:.1f}.Alpha={:+.2f}/".format(metal, alpha)
            s = "lte{:05d}-{:.2f}-{:.1f}.Alpha={:+.2f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(temp, log_g,
                                                                                                      metal, alpha)
        else:
            metal = abs(metal)
            p = "/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-{:.1f}".format(metal)
            s = "lte{:05d}-{:.2f}-{:.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(temp, log_g, metal)

        return p, s

    # =============================================================================
    # =============================================================================
    def download_PHOENIX_files(self, p, f_n):
        '''
        This function downloads the synthetic spectrum files from the Goettingen ftp
        :param: p - str, an ftp path
        :param: f_n: str, file name to be downloaded
        '''
        tmp_dir = os.getcwd()
        os.chdir(TEMPLATE_FILES_PATH)

        path = p
        filename = f_n

        ftp = FTP(self.server_path)
        ftp.login()

        ftp.cwd(path)
        ftp.retrbinary("RETR " + filename, open(filename, 'wb').write)
        ftp.quit()

        os.chdir(tmp_dir)

    # =============================================================================
    # =============================================================================
    def doppler(self, vel, wv=[]):
        '''
        This function shifts the template spectrum according to a given velocity (a Doppler shift).
        :param: vel - the velocity in which to boost.
        :param: wv - If empty, the template will be boosted.
        :return: Boosted wavelengh array \ template model.

        Note: This only works for a single order template, so boost before you cut.
        '''

        if type(vel) is not u.quantity.Quantity:
            vel = float(vel) << u.kilometer / u.second

        beta = ((vel/self.model.c).decompose()).value

        if wv == []:
            return self.model.wv.astype("float64") * (1 + beta)
        else:
            return (wv * (1 + beta)).astype("float64")

    # =============================================================================
    # =============================================================================
    def add_noise(self, snr, sp=[], rndseed=None):
        '''
        This function adds simulated noise to the synthetic spectrum according to a given snr.
        :param: snr - float, desired signal to noise ration
        :param: rndseed - random seed, for reproducibility.
        :return: Noised spectrum model.
        '''

        if sp!=[]:
            self.model.sp = sp

        if snr == -1:
            return self.model.sp
        else:
            NormFac = np.quantile(self.model.sp[0], 0.98)

            if rndseed is None:
                random.seed()
            else:
                random.seed(rndseed)

            noise = np.random.normal(0, 1, len(self.model.sp[0])) * NormFac / snr

            return self.model.sp + noise

    # =============================================================================
    # =============================================================================
    def integrate_spec(self, integration_ratio, wv=[], sp=[]):
        '''
        Sum-up consecutive pixels in the given Template model.
        This can be used to lower the resolution for computational needs.

        :param: integration_ration - integer. Reduce the the resolution
                                     by this factor.
        '''
        new_sp = []
        new_wv = []

        if wv==[]:
            wv = self.model.wv
            sp = self.model.sp

        l = len(wv[0])

        for i in range(l // integration_ratio - 1):
            new_sp.append(np.sum(sp[0][i * integration_ratio:(i+1)*integration_ratio]))
            new_wv.append(np.mean(wv[0][i * integration_ratio:(i+1)*integration_ratio]))

        new_sp = np.reshape(new_sp, (1, -1))
        new_wv = np.reshape(new_wv, (1, -1))

        self.model.sp = new_sp
        self.model.wv = new_wv

        return new_wv, new_sp

    # =============================================================================
    # =============================================================================
    def save_template_model(self, title="export_template", path='template_files/'):
        '''
        This function export the template model, possibly after being edited and manipulated.
        :param: title - str, output file name, preferably describes the template characteristics
        :param: path - str, dir path in which the template will be saved
        '''
        output_data = pd.DataFrame.from_dict({"wv": self.model.wv,
                                              "sp": self.model.sp,},
                                             orient="index")
        output_data.to_csv(path + title + '.csv')

    # =============================================================================
    # =============================================================================
    def cut_multiorder(self, wv_bounds):
        '''
        This function cuts the template according to a given wavelength bound.
        :param: wv_bounds - an NordX2 numpy array with the upper and lower
                            limit of each order.
        :return: The template model, trimmed to the provided shapes.
        '''

        wvRangeInd = []
        for ordInd, bnd in enumerate(wv_bounds):
            Indices = (self.model.wv >= bnd[0]) & (self.model.wv<=bnd[1])
            wvRangeInd.append(Indices[0])

        spMO = [np.array(self.model.sp[0][x]) for x in wvRangeInd]
        wvMO = [np.array(self.model.wv[0][x]) for x in wvRangeInd]

        self.model.sp = spMO
        self.model.wv = wvMO

        return self

    # =============================================================================
    # =============================================================================
    def cut_multiorder_like(self, obs, margins=150):
        '''
        This function cuts the template to the shape of a given Spectrum object.
        :param: obs - a spectrum object, according to which the template
                      will be sliced.
        :param: margins - the boundaries around the limits, in km/s.
        :return: The template model, trimmed to the provided shapes.
        '''

        wv_bounds = np.zeros((len(obs.wv), 2))
        margins = float(margins) << u.kilometer / u.second
        beta = (margins/obs.c).decompose().value

        for ordInd, w in enumerate(obs.wv):
            wv_bounds[ordInd, 0] = min(w)*(1-beta)
            wv_bounds[ordInd, 1] = max(w)*(1+beta)

        self.cut_multiorder(wv_bounds)

        return self
