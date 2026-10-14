"""Template: a reference spectrum for the correlation engines.

A ``Template`` wraps a ``Spectrum`` in ``self.model``. Build it from a ready
``Spectrum`` (``Template(spectrum)``), from wavelength and flux arrays
(``Template.from_arrays``), from a PHOENIX-ACES synthetic spectrum read from
the local cache in ``template_files/`` (``Template.from_phoenix``; fetched from
the Goettingen server only on explicit opt-in, ``allow_download=True``), or
from a line mask (``Template.from_mask``). Its methods broaden (rotational,
instrumental), Doppler-shift, rebin and cut the model to the orders of an
observation.

Conventions: wavelengths in Angstrom (PHOENIX grids are in vacuum; ``air=True``
converts them with Morton 2000, ApJS 130, 403), velocities in km/s.

Reference: PHOENIX-ACES models -- Husser et al. 2013, A&A 553, A6.
"""

from astropy.io import fits
from astropy.utils.data import download_file
from sparta.spectrum import Spectrum
from unicor import dsp
import numpy as np
import os
import shutil
from astropy import units as u


TEMPLATE_FILES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'template_files/'
)
PHOENIX_URL = "https://phoenix.astro.physik.uni-goettingen.de/data/"
PHOENIX_WAVE_FILE = 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'
_PHOENIX_WAVE_SUB_PATH = 'HiResFITS/'


def _per_order(like, orders):
    """Return per-order arrays in the container type of ``like``.

    Single templates hold an (n_orders, n_pixels) ndarray; multi-order ones
    cut to an observation may be ragged lists. Keep whichever the caller had.
    """
    return np.array(orders) if isinstance(like, np.ndarray) else list(orders)


class Template:

    # =============================================================================
    # =============================================================================
    def __init__(self, model):
        '''
        Wrap a ready-for-use spectrum as a template, as-is (no copy).

        :param: model - Spectrum, the template's (multi-order) wavelengths
                        (Angstrom) and fluxes; stored as ``self.model``.

        See ``from_arrays``, ``from_phoenix`` and ``from_mask`` for the other
        ways to build one.
        '''
        if not isinstance(model, Spectrum):
            raise TypeError(f"model must be a Spectrum, not {type(model).__name__}; "
                            "use Template.from_arrays for plain arrays")
        self.model = model

    # =============================================================================
    # =============================================================================
    @classmethod
    def from_arrays(cls, wave, flux):
        '''
        A single-order template from one wavelength and one flux array.

        :param: wave - 1-D wavelength array (Angstrom), increasing.
        :param: flux - 1-D flux array matching ``wave``.
        :return: Template whose model holds one order, as (1, n_pixels)
                 arrays.
        :raises ValueError: if the arrays are not 1-D or differ in length.
        '''
        wave, flux = np.asarray(wave), np.asarray(flux)
        if wave.ndim != 1 or wave.shape != flux.shape:
            raise ValueError(f"wave and flux must be 1-D arrays of one length, got shapes "
                             f"{wave.shape} and {flux.shape}")
        return cls(Spectrum(wv=wave.reshape(1, -1), sp=flux.reshape(1, -1)))

    # =============================================================================
    # =============================================================================
    @classmethod
    def from_phoenix(cls, teff, logg, feh=0.0, alpha=0.0, *, wave_range,
                     air=False, allow_download=False):
        '''
        A single-order PHOENIX-ACES-AGSS-COND-2011 HiRes model (Husser et al.
        2013) over a wavelength range.

        :param: teff  - effective temperature (K), a grid value.
        :param: logg  - log surface gravity (cgs), a grid value.
        :param: feh   - metallicity [Fe/H] (dex), a grid value.
        :param: alpha - alpha enhancement [alpha/Fe] (dex), a grid value.
        :param: wave_range - (lo, hi) wavelengths (Angstrom); the model keeps
                        the grid points from the first above ``lo`` up to,
                        not including, the first above ``hi``.
        :param: air   - bool, convert the (vacuum) PHOENIX wavelengths to air
                        (Morton 2000) before the range is applied.
        :param: allow_download - bool, default False. Fetch files missing from
                        the local cache (TEMPLATE_FILES_PATH) over HTTPS; when
                        False a missing file raises FileNotFoundError.
        :return: Template with one order: wavelengths (Angstrom) and the
                 PHOENIX flux density divided by 1e13.
        :raises FileNotFoundError: for a missing file with downloads disabled.
        :raises ValueError: if ``wave_range`` is not inside the model grid.
        '''
        wave_path = _cached_phoenix_file(_PHOENIX_WAVE_SUB_PATH, PHOENIX_WAVE_FILE,
                                         allow_download)
        flux_path = _cached_phoenix_file(*_phoenix_filename(teff, logg, feh, alpha),
                                         allow_download)

        with fits.open(flux_path) as hdul_spec:
            s = np.squeeze(np.array(hdul_spec[0].data))
        with fits.open(wave_path) as hdul_wl:
            w = np.squeeze(np.array(hdul_wl[0].data))

        if air:
            # Morton (2000, ApJS 130, 403), vacuum to air.
            s2 = 10**8 * w**(-2)
            f = 1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2)
            w = w/f

        cut = _wave_slice(w, wave_range)
        spectrum = np.reshape(s[cut], (1, -1)) / (10 ** 13)
        wavelengths = np.reshape(w[cut], (1, -1))
        return cls(Spectrum(wv=wavelengths, sp=spectrum))

    # =============================================================================
    # =============================================================================
    @classmethod
    def from_mask(cls, path, wave_range, allow_download=False):
        '''
        A line-mask template (e.g. a HARPS DRS ``.mas`` file) on the PHOENIX
        wavelength grid.

        The flux is 0 except inside each mask window, where it is the negative
        of the line's weight, so lines read as absorption.

        :param: path       - text file of three columns: window start and end
                             (Angstrom) and weight; '#' starts a comment.
                             Where windows overlap, the later line wins.
        :param: wave_range - (lo, hi) wavelengths (Angstrom), applied as in
                             ``from_phoenix``.
        :param: allow_download - bool, see ``from_phoenix`` (only the PHOENIX
                             wavelength grid is needed).
        :return: Template with one order on the PHOENIX (vacuum) grid.
        :raises ValueError: if ``wave_range`` is not inside the grid.
        '''
        wave_path = _cached_phoenix_file(_PHOENIX_WAVE_SUB_PATH, PHOENIX_WAVE_FILE,
                                         allow_download)
        with fits.open(wave_path) as hdul_wl:
            w = np.squeeze(np.array(hdul_wl[0].data))
        w = w[_wave_slice(w, wave_range)]

        lines = np.loadtxt(path, usecols=(0, 1, 2), ndmin=2)
        s = np.zeros_like(w)
        # Pixels strictly inside each (start, end) window.
        first = np.searchsorted(w, lines[:, 0], side='right')
        last = np.searchsorted(w, lines[:, 1], side='left')
        for a, b, weight in zip(first, last, lines[:, 2]):
            s[a:b] = -weight

        return cls(Spectrum(wv=np.reshape(w, (1, -1)), sp=np.reshape(s, (1, -1))))

    # =============================================================================
    # =============================================================================
    def broaden_instrumental(self, resolution):
        '''
        Broaden every order with a Gaussian instrumental profile.

        :param: resolution - float, the resolving power R; the kernel FWHM is
                             c / R (km/s).
        :return: self, with the model's flux broadened.
        '''
        self.model.sp = _per_order(self.model.sp,
                                   [dsp.broaden_gaussian(s, dsp.dv_from_wave(w), dsp.C_KMS / resolution)
                                    for w, s in zip(self.model.wv, self.model.sp)])
        return self

    # =============================================================================
    # =============================================================================
    def broaden_rotational(self, vsini, epsilon):
        '''
        Broaden every order with a rotational profile (Gray 2005).

        :param: vsini   - float, projected rotational velocity [km/s]
        :param: epsilon - float, linear limb-darkening coefficient (0-1)
        :return: self, with the model's flux broadened.
        '''
        self.model.sp = _per_order(self.model.sp,
                                   [dsp.broaden_rotational(s, dsp.dv_from_wave(w), vsini, epsilon)
                                    for w, s in zip(self.model.wv, self.model.sp)])
        return self

    # =============================================================================
    # =============================================================================
    def doppler(self, vel, wv=None, velocity_convention='cz'):
        '''
        This function shifts the template spectrum according to a given velocity (a Doppler shift).
        :param: vel - the radial velocity to shift by (km/s if unitless);
                      positive red-shifts.
        :param: wv - wavelengths (Angstrom) to shift; if None, the template's own.
        :param: velocity_convention - 'cz' (default; lambda * (1 + v/c)) or
                      'relativistic' (lambda * sqrt((1 + v/c) / (1 - v/c))),
                      see ``unicor.dsp.doppler_factor``. Use the convention the
                      RVs will be measured in.
        :return: Boosted wavelength array / template model (Angstrom).

        Note: This only works for a single order template, so boost before you cut.
        '''

        if type(vel) is not u.quantity.Quantity:
            vel = float(vel) << u.kilometer / u.second

        target = np.array(self.model.wv) if wv is None else wv
        return dsp.doppler_shift_wave(target, vel.to_value(u.km / u.s), velocity_convention)

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
        for bnd in wv_bounds:
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
    def cut_multiorder_like(self, spectrum, margins=150):
        '''
        This function cuts the template to the shape of a given Spectrum object.
        :param: spectrum - a Spectrum object, according to which the template
                      will be sliced.
        :param: margins - the boundaries around the limits, in km/s.
        :return: The template model, trimmed to the provided shapes.
        '''

        wv_bounds = np.zeros((len(spectrum.wv), 2))
        margins = float(margins) << u.kilometer / u.second
        beta = (margins/spectrum.c).decompose().value

        for ordInd, w in enumerate(spectrum.wv):
            wv_bounds[ordInd, 0] = min(w)*(1-beta)
            wv_bounds[ordInd, 1] = max(w)*(1+beta)

        self.cut_multiorder(wv_bounds)

        return self


def _phoenix_filename(teff, logg, feh, alpha):
    '''
    Server sub-path and file name of a PHOENIX-ACES HiRes model.

    PHOENIX names carry the metallicity sign explicitly ("Z-1.0", "Z+0.5"),
    so its magnitude is formatted; the effective temperature is an integer.

    :param: teff, logg, feh, alpha - see ``Template.from_phoenix``.
    :return: (sub_path, filename), str.
    '''
    teff = int(teff)
    if feh > 0:
        p = "/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z+{:.1f}/".format(feh)
        s = "lte{:05d}-{:.2f}+{:.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(teff, logg, feh)
    elif alpha != 0:
        feh = abs(feh)
        p = "/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-{:.1f}.Alpha={:+.2f}/".format(feh, alpha)
        s = "lte{:05d}-{:.2f}-{:.1f}.Alpha={:+.2f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(teff, logg,
                                                                                                  feh, alpha)
    else:
        feh = abs(feh)
        p = "/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-{:.1f}".format(feh)
        s = "lte{:05d}-{:.2f}-{:.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(teff, logg, feh)

    return p, s


def _phoenix_url(sub_path, filename):
    '''The HTTPS URL of a PHOENIX file (see ``_phoenix_filename``).'''
    return PHOENIX_URL + sub_path.strip('/') + '/' + filename


def _cached_phoenix_file(sub_path, filename, allow_download):
    '''
    Local path of a PHOENIX file, downloading it only if allowed.

    :param: sub_path, filename - see ``_phoenix_filename``.
    :param: allow_download - bool; when False a file missing from
                             TEMPLATE_FILES_PATH raises.
    :return: str, the path in TEMPLATE_FILES_PATH.
    :raises FileNotFoundError: if the file is missing and downloads are off.
    '''
    path = os.path.join(TEMPLATE_FILES_PATH, filename)
    if os.path.exists(path):
        return path
    if not allow_download:
        raise FileNotFoundError(
            f"PHOENIX file '{filename}' is not in {TEMPLATE_FILES_PATH} "
            "and downloads are disabled (allow_download=False); pass "
            "allow_download=True to fetch it.")
    return _download_phoenix_file(sub_path, filename)


def _download_phoenix_file(sub_path, filename, timeout=120):
    '''
    Download one PHOENIX file from the Goettingen server into
    TEMPLATE_FILES_PATH, over HTTPS.

    Uses astropy.utils.data.download_file (TLS, timeout, atomic
    temp-then-move), so there is no process-CWD mutation, no leaked file
    handle, and no partial file left behind on failure. Any download error
    propagates to the caller.

    :param: sub_path - str, server sub-path (see ``_phoenix_filename``).
    :param: filename - str, file name to fetch.
    :param: timeout  - float, per-request timeout in seconds.
    :return: str, the local path of the downloaded file.
    '''
    os.makedirs(TEMPLATE_FILES_PATH, exist_ok=True)
    target = os.path.join(TEMPLATE_FILES_PATH, filename)

    cached = download_file(_phoenix_url(sub_path, filename), cache=False, timeout=timeout,
                           show_progress=False)
    try:
        shutil.move(cached, target)
    except BaseException:
        if os.path.exists(cached):
            os.remove(cached)
        raise
    return target


def _wave_slice(wave, wave_range):
    '''
    Index range of a PHOENIX grid: from the first point above ``lo`` up to,
    not including, the first point above ``hi``.

    :param: wave       - increasing wavelength grid (Angstrom).
    :param: wave_range - (lo, hi), Angstrom.
    :return: slice.
    :raises ValueError: unless lo < hi < wave[-1].
    '''
    lo, hi = wave_range
    if not lo < hi < wave[-1]:
        raise ValueError(f"wave_range={tuple(wave_range)} must satisfy lo < hi < "
                         f"{wave[-1]:.1f} A (the end of the PHOENIX grid)")
    return slice(int(np.argmax(wave > lo)), int(np.argmax(wave > hi)))
