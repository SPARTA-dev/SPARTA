#                   ----------------------------
#                    Template.py (SPARTA UNICOR class)
#                   ----------------------------
# This file defines the "Template" class. An object of this class stores
# a Spectrum object, saved in the self.model field. This class has some
# additional attributes added to the ones of the Spectrum class, including
# the ability to download, process and store PHOENIX synthetic spectrum.
#
# A Template class stores the following methods:
# ---------------------------------------------
#
# 1) create_PHOENIX_fname - Prepare the needed spectra files names and paths for downloading
# 2) download_PHOENIX_files - Downloads the synthetic spectrum files over HTTPS from Goettingen
# 3) doppler - Shifts the synthetic spectrum according to a given velocity (Doppler effect)
# 4) add_noise - Adds simulated noise to the synthetic spectrum according to a given snr
# 5) GaussianBroadening - broadens the template with a Gaussian window,
#                      to account for the instrumental broadening.
# 6) RotationalBroadening - broadens the template with a rotational profile
#                        to account for stellar rotation.
# 7) integrate_spec - sums up consecutive pixels in the given Template model,
#                           to lower the resolution for computational needs
# 8) save_template_model - exports the template model, possibly after being edited and manipulated.
# 9) cut_multiorder - cuts the template according to a given wavelength bound.
# 10) cut_multiorder_like - cuts the template to the shape of a given Spectrum object.
#
# Dependencies: numpy, astropy, shutil, pathlib and os.
# Last update: Avraham Binnenfeld, 20210510.


from astropy.io import fits
from astropy.utils.data import download_file
from sparta.UNICOR.Spectrum import Spectrum
from unicor import dsp
import numpy as np
from pathlib import Path
import os
import shutil
from astropy import units as u
from astropy.table import Table


TEMPLATE_FILES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'template_files/'
)


def _per_order(like, orders):
    """Return per-order arrays in the container type of ``like``.

    Single templates hold an (n_orders, n_pixels) ndarray; multi-order ones
    cut to an observation may be ragged lists. Keep whichever the caller had.
    """
    return np.array(orders) if isinstance(like, np.ndarray) else list(orders)


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
           This required passing  "template=[template object]"
        2) A ready-for-use Spectrum object can be passed on to the class and will be saved as-is.
        3) The path of a HARPS DRS mask can be provided.
        '''
        if 'template' in kwargs:
            self.model = kwargs['template']
            if 'vel' in kwargs:
                self.vel = kwargs['vel']
            else:
                self.vel = []

        elif all(k in kwargs for k in ('temp', 'log_g', 'metal', 'alpha')):
            self.temp = kwargs['temp']
            self.log_g = kwargs['log_g']
            self.metal = kwargs['metal']
            self.alpha = kwargs['alpha']
            self.min_val = kwargs['min_val']
            self.max_val = kwargs['max_val']
            self.server_url = "https://phoenix.astro.physik.uni-goettingen.de/data/"
            self.wl_f_n = 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'

            p, f_n = self.create_PHOENIX_fname(self.temp, self.log_g, self.metal, self.alpha)

            # Fetch the wavelength grid and the requested spectrum if they are
            # not already cached locally. Downloads can be disabled with
            # allow_download=False (then a missing file raises instead of
            # silently reaching out to the network).
            allow_download = kwargs.get('allow_download', True)
            for sub_path, fname in [('HiResFITS/', self.wl_f_n), (p, f_n)]:
                if not os.path.exists(os.path.join(TEMPLATE_FILES_PATH, fname)):
                    if not allow_download:
                        raise FileNotFoundError(
                            f"PHOENIX file '{fname}' is not in {TEMPLATE_FILES_PATH} "
                            "and downloads are disabled (allow_download=False).")
                    self.download_PHOENIX_files(sub_path, fname)

            with fits.open(os.path.join(TEMPLATE_FILES_PATH, f_n)) as hdul_spec:
                s = np.squeeze(np.array(hdul_spec[0].data))
                header = hdul_spec[0].header.copy()

            with fits.open(os.path.join(TEMPLATE_FILES_PATH, self.wl_f_n)) as hdul_wl:
                w = np.squeeze(np.array(hdul_wl[0].data))

            # Change to air wavelengths, if required
            # The following transformation is taken from Morton (2000, ApJ. Suppl., 130, 403)
            if 'air' in kwargs:
                if kwargs['air']:
                    s2 = 10**8 * w**(-2)
                    f = 1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2)
                    w = w/f

            self.PHXMASS = header['PHXMASS']

            # Effective stellar radius cm
            self.PHXREFF = header['PHXREFF']

            min_index = np.where(w > self.min_val)[0][0]
            max_index = np.where(w > self.max_val)[0][0]

            spectrum = s[min_index:max_index]
            wavelengths = w[min_index:max_index]

            spectrum = np.reshape(spectrum, (1, -1))
            wavelengths = np.reshape(wavelengths, (1, -1))

            spectrum = spectrum / (10 ** 13)

            self.model = Spectrum(wv=wavelengths, sp=spectrum)

        elif 'mask' in kwargs:
            self.min_val = kwargs['min_val']
            self.max_val = kwargs['max_val']
            self.wl_f_n = 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'

            with fits.open(os.path.join(TEMPLATE_FILES_PATH, self.wl_f_n)) as hdul_wl:
                w = np.squeeze(np.array(hdul_wl[0].data))
            s = np.zeros_like(w)

            min_index = np.where(w > self.min_val)[0][0]
            max_index = np.where(w > self.max_val)[0][0]

            raw_mask = Table.read(kwargs['mask'], format='ascii')
            raw_mask['col1'].name = 'wv1'
            raw_mask['col2'].name = 'wv2'
            raw_mask['col3'].name = 'msk'

            for i, m in enumerate(raw_mask['msk']):
                ii = np.argwhere((w > raw_mask['wv1'][i]) & (w < raw_mask['wv2'][i]))
                s[ii] = -m

            spectrum = s[min_index:max_index]
            wavelengths = w[min_index:max_index]

            spectrum = np.reshape(spectrum, (1, -1))
            wavelengths = np.reshape(wavelengths, (1, -1))

            self.model = Spectrum(wv=wavelengths, sp=spectrum)

        elif all(k in kwargs for k in ('spectrum', 'wavelengths')):
            spectrum = kwargs['spectrum']
            wavelengths = kwargs['wavelengths']
            if all(k in kwargs for k in ('min_val', 'max_val')):
                self.min_val = kwargs['min_val']
                self.max_val = kwargs['max_val']

            spectrum = np.reshape(spectrum, (1, -1))
            wavelengths = np.reshape(wavelengths, (1, -1))

            self.model = Spectrum(wv=wavelengths, sp=spectrum)

        else:
            self.model = Spectrum()
            self.vel = []

        if 'PHXMASS' in kwargs:
            self.PHXMASS = kwargs['PHXMASS']

        if 'PHXREFF' in kwargs:
            self.PHXREFF = kwargs['PHXREFF']


    # =============================================================================
    def GaussianBroadening(self, **kwargs):
        '''
        Broaden every order with a Gaussian instrumental profile.

        :param: resolution - float, the resolving power R; the kernel FWHM is
                             c / R (km/s).
        :param: wv, sp     - optional external (multi-order) wavelengths
                             (Angstrom) and normalized flux. When given, they
                             are broadened and returned, and the template's own
                             model is left untouched.
        :return: the broadened flux, shape (n_orders, n_pixels) per order.
        '''
        resolution = kwargs["resolution"]

        external = 'wv' in kwargs and 'sp' in kwargs
        wvs, sps = (kwargs["wv"], kwargs["sp"]) if external else (self.model.wv, self.model.sp)

        flux = _per_order(sps, [dsp.broaden_gaussian(s, dsp.dv_from_wave(w), dsp.C_KMS / resolution)
                                for w, s in zip(wvs, sps)])

        if not external:
            self.model.sp = flux
        return flux

    # =============================================================================
    # =============================================================================
    def RotationalBroadening(self, **kwargs):
        '''
        Broaden every order with a rotational profile (Gray 2005).

        :param: vsini   - float, projected rotational velocity [km/s]
        :param: epsilon - float, linear limb-darkening coefficient (0-1)
        '''
        epsilon = kwargs["epsilon"]
        vsini = kwargs["vsini"]

        self.model.sp = _per_order(self.model.sp,
                                   [dsp.broaden_rotational(s, dsp.dv_from_wave(w), vsini, epsilon)
                                    for w, s in zip(self.model.wv, self.model.sp)])
        return self

        # =============================================================================
    # =============================================================================
    def create_PHOENIX_fname(self, temp, log_g, metal, alpha):
        '''
        This function prepares the needed spectra files names and paths for downloading
        :param: The following stellar parameters are necessary for PHOENIX spectrum definition:
                temp, log_g, metal, alpha.
        '''
        # PHOENIX names carry the metallicity sign explicitly ("Z-1.0", "Z+0.5"),
        # so format its magnitude; the effective temperature is an integer.
        temp = int(temp)
        if metal > 0:
            p = "/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z+{:.1f}/".format(metal)
            s = "lte{:05d}-{:.2f}+{:.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(temp, log_g, metal)
        elif alpha != 0:
            metal = abs(metal)
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
    def download_PHOENIX_files(self, p, f_n, timeout=120):
        '''
        Download one synthetic-spectrum file from the Goettingen PHOENIX server
        into TEMPLATE_FILES_PATH, over HTTPS.

        Uses astropy.utils.data.download_file (TLS, timeout, atomic
        temp-then-move), so there is no process-CWD mutation, no leaked file
        handle, and no partial file left behind on failure. Any download error
        propagates to the caller.

        :param: p       - str, server sub-path (as built by create_PHOENIX_fname).
        :param: f_n     - str, file name to fetch.
        :param: timeout - float, per-request timeout in seconds.
        :return: str, the local path of the downloaded file.
        '''
        os.makedirs(TEMPLATE_FILES_PATH, exist_ok=True)
        url = self.server_url + p.strip('/') + '/' + f_n
        target = os.path.join(TEMPLATE_FILES_PATH, f_n)

        cached = download_file(url, cache=False, timeout=timeout,
                               show_progress=False)
        try:
            shutil.move(cached, target)
        except BaseException:
            if os.path.exists(cached):
                os.remove(cached)
            raise
        return target

    # =============================================================================
    # =============================================================================
    def doppler(self, vel, wv=[], velocity_convention='cz'):
        '''
        This function shifts the template spectrum according to a given velocity (a Doppler shift).
        :param: vel - the radial velocity to shift by (km/s if unitless);
                      positive red-shifts.
        :param: wv - If empty, the template will be boosted.
        :param: velocity_convention - 'cz' (default; lambda * (1 + v/c)) or
                      'relativistic' (lambda * sqrt((1 + v/c) / (1 - v/c))),
                      see ``unicor.dsp.doppler_factor``. Use the convention the
                      RVs will be measured in.
        :return: Boosted wavelength array / template model (Angstrom).

        Note: This only works for a single order template, so boost before you cut.
        '''

        if type(vel) is not u.quantity.Quantity:
            vel = float(vel) << u.kilometer / u.second

        target = np.array(self.model.wv) if len(wv) == 0 else wv
        return dsp.doppler_shift_wave(target, vel.to_value(u.km / u.s), velocity_convention)

    # =============================================================================
    # =============================================================================
    def add_noise(self, snr, sp=[], rndseed=None):
        '''
        This function adds simulated noise to the synthetic spectrum according to a given snr.
        :param: snr - float, desired signal to noise ration
        :param: rndseed - random seed, for reproducibility.
        :return: Noised spectrum model.
        '''

        if len(sp) != 0:
            self.model.sp = sp

        if snr == -1:
            return self.model.sp
        else:
            NormFac = np.quantile(self.model.sp[0], 0.98)

            # A dedicated numpy Generator; seeding it (rndseed) makes the noise
            # reproducible. The old code seeded the `random` module, which does
            # not affect numpy, so rndseed had no effect.
            rng = np.random.default_rng(rndseed)
            noise = rng.normal(0, 1, len(self.model.sp[0])) * NormFac / snr

            return self.model.sp + noise

    # =============================================================================
    # =============================================================================
    def integrate_spec(self, integration_ratio, wv=[], sp=[]):
        '''
        Sum-up consecutive pixels in the given Template model.
        This can be used to lower the resolution for computational needs.

        :param: integration_ration - integer. Reduce the the resolution
                                     by this factor.

        Only the first order is kept. A native step recorded for the model
        (Spectrum.native_steps) is coarsened to the summed pixels' step if
        that is larger; explicit ``wv``/``sp`` replace the model, so its
        record is dropped.
        '''
        new_sp = []
        new_wv = []

        native = None
        if len(wv) == 0:
            wv = self.model.wv
            sp = self.model.sp
            if 'NativeStep' in self.model.Info:
                native = self.model.native_steps()[0]

        l = len(wv[0])

        for i in range(l // integration_ratio - 1):
            new_sp.append(np.sum(sp[0][i * integration_ratio:(i+1)*integration_ratio]))
            new_wv.append(np.mean(wv[0][i * integration_ratio:(i+1)*integration_ratio]))

        new_sp = np.reshape(new_sp, (1, -1))
        new_wv = np.reshape(new_wv, (1, -1))

        self.model.sp = new_sp
        self.model.wv = new_wv
        self.model.Info.pop('NativeStep', None)
        if native is not None:
            self.model.Info['NativeStep'] = [max(native, dsp.dv_from_wave(new_wv[0]))]

        return new_wv, new_sp

    # =============================================================================
    # =============================================================================
    def save_template_model(self, title="export_template", path='template_files/'):
        '''
        Export the template model, possibly after editing, as a NumPy ``.npz``.

        Each order is stored under its own keys -- ``wv_0``, ``sp_0``,
        ``wv_1``, ... -- so orders of different lengths round-trip exactly.
        (The former CSV export wrote numpy's truncated text form of each
        array, which could not be read back.)

        :param: title - str, output file name (without extension).
        :param: path  - str, directory in which the file is written.
        :return: the path of the written file.
        '''
        arrays = {}
        for I, (w, s) in enumerate(zip(self.model.wv, self.model.sp)):
            arrays[f'wv_{I}'] = np.asarray(w, float)   # Angstrom
            arrays[f'sp_{I}'] = np.asarray(s, float)
        target = os.path.join(path, title + '.npz')
        np.savez(target, **arrays)
        return target

    # =============================================================================
    # =============================================================================
    def cut_multiorder(self, wv_bounds):
        '''
        This function cuts the template according to a given wavelength bound.
        :param: wv_bounds - an NordX2 numpy array with the upper and lower
                            limit of each order.
        :return: The template model, trimmed to the provided shapes.

        Every new order is cut from order 0 of the model, so a recorded native
        step (Spectrum.native_steps) is that of order 0 for each of them.
        '''
        native = (self.model.native_steps()[0]
                  if 'NativeStep' in self.model.Info else None)

        wvRangeInd = []
        for ordInd, bnd in enumerate(wv_bounds):
            Indices = (self.model.wv >= bnd[0]) & (self.model.wv<=bnd[1])
            wvRangeInd.append(Indices[0])

        spMO = [np.array(self.model.sp[0][x]) for x in wvRangeInd]
        wvMO = [np.array(self.model.wv[0][x]) for x in wvRangeInd]

        self.model.sp = spMO
        self.model.wv = wvMO
        if native is not None:
            self.model.Info['NativeStep'] = [native] * len(wvMO)

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
