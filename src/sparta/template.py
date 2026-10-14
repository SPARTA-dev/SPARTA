"""Template: a reference spectrum for the correlation engines.

A ``Template`` wraps a ``Spectrum`` in ``self.model``. Build it from a ready
``Spectrum`` (``Template(spectrum)``), from wavelength and flux arrays
(``Template.from_arrays``), from a PHOENIX-ACES synthetic spectrum read from
the local cache in ``template_files/`` (``Template.from_phoenix``; fetched from
the Goettingen server only on explicit opt-in, ``allow_download=True``), or
from a line mask (``Template.from_mask``). Its methods broaden (rotational,
instrumental), Doppler-shift, rebin and cut the model to the orders of an
observation. ``phoenix_spectrum`` returns the PHOENIX model as plain arrays.

Conventions: wavelengths in Angstrom (PHOENIX grids are in vacuum; ``air=True``
converts them with Morton 2000, ApJS 130, 403), velocities in km/s.

Reference: PHOENIX-ACES models -- Husser et al. 2013, A&A 553, A6.
"""

from astropy.io import fits
from astropy.utils.data import download_file
from astropy.utils.exceptions import AstropyUserWarning
from sparta.preprocess import coarsest_native_step
from sparta.spectrum import Spectrum
from unicor import dsp
import numpy as np
import os
import shutil
import tempfile
import warnings
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


def _is_log_uniform(wave, rtol=1e-3):
    """True if ``wave`` is uniform in ln(lambda) to ``rtol`` (as ``unicor.log_grid_dln`` checks)."""
    dln = np.diff(np.log(np.asarray(wave, float)))
    return dln.size > 0 and np.ptp(dln) <= rtol * np.median(dln)


def _broaden_velocity_even(wave, flux, broaden):
    """Apply a velocity kernel to one order on a uniform ln(lambda) grid.

    One kernel has one velocity width only on a velocity-even grid. PHOENIX
    HiRes is linear in wavelength with a step of 0.006 A below 5000 A and
    0.01 A above (0.36 vs 0.60 km/s), and any linear grid's velocity step
    drifts as 1/lambda; on such an order the median step made the kernel up
    to ~1.7x too wide or too narrow. So an order that is not uniform in
    ln(lambda) is resampled onto one at its finest velocity step (cubic, no
    part sampled more coarsely), broadened there, and resampled back onto
    its own wavelengths -- the template keeps the grid it had.

    Parameters
    ----------
    wave : array_like
        1-D increasing wavelengths (Angstrom).
    flux : array_like
        1-D flux matching ``wave``.
    broaden : callable
        ``broaden(flux, dv)`` with ``dv`` the grid's velocity step (km/s).

    Returns
    -------
    ndarray
        The broadened flux on ``wave``.
    """
    wave = np.asarray(wave, float)
    if _is_log_uniform(wave):
        return broaden(flux, dsp.dv_from_wave(wave))
    dv = dsp.C_KMS * np.min(np.diff(wave) / wave[:-1])
    grid, _ = dsp.log_lambda_grid(wave[0], wave[-1], dv)
    broadened = broaden(dsp.resample(wave, flux, grid, kind='cubic'), dv)
    return dsp.resample(grid, broadened, wave, kind='cubic')


class Template:
    '''
    A reference spectrum for UNICOR (``CCF1d``), TODCOR (``CCF2d``) and the
    simulator.

    The engines take the template at rest. A single-order model (e.g. a
    library spectrum covering the whole observed range) is reused for every
    observed order; a multi-order one (``cut_multiorder_like``) is matched
    order by order. State the template's wavelength frame (air or vacuum) to
    match the observation's.

    Attributes
    ----------
    model : Spectrum
        The template's wavelengths (Angstrom) and fluxes; the methods modify
        it in place.
    '''

    # =============================================================================
    # =============================================================================
    def __init__(self, model):
        '''
        Wrap a ready-for-use spectrum as a template, as-is (no copy).

        See ``from_arrays``, ``from_phoenix`` and ``from_mask`` for the other
        ways to build one.

        Parameters
        ----------
        model : Spectrum
            The template's (multi-order) wavelengths (Angstrom) and fluxes;
            stored as ``self.model``.

        Raises
        ------
        TypeError
            If ``model`` is not a Spectrum.
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

        Parameters
        ----------
        wave : array_like
            1-D increasing wavelengths (Angstrom).
        flux : array_like
            1-D flux matching ``wave``.

        Returns
        -------
        Template
            One order, held as (1, n_pixels) arrays.

        Raises
        ------
        ValueError
            If the arrays are not 1-D or differ in length.
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
        2013) over a wavelength range: ``phoenix_spectrum`` as a Template.

        The model is read from the local cache, TEMPLATE_FILES_PATH (the
        package's ``template_files/``). SPARTA ships three models there --
        (teff, logg, feh) = (4900, 4.5, 0.0), (5500, 4.5, 0.0) and
        (5800, 3.5, +0.5) -- and the wavelength grid. Any other model needs
        ``allow_download=True`` once.

        The flux is not continuum-normalized, and the model is not broadened:
        see ``broaden_rotational`` and ``broaden_instrumental``.

        Parameters
        ----------
        teff, logg, feh, alpha, wave_range, air, allow_download
            See ``phoenix_spectrum``.

        Returns
        -------
        Template
            One order: wavelengths (Angstrom) and the PHOENIX flux density
            (erg/s/cm^2/cm) divided by 1e13, held as (1, n_pixels) arrays.

        Raises
        ------
        FileNotFoundError, OSError, ValueError
            See ``phoenix_spectrum``.
        '''
        wave, flux = phoenix_spectrum(teff, logg, feh, alpha, wave_range=wave_range, air=air,
                                      allow_download=allow_download)
        return cls(Spectrum(wv=np.reshape(wave, (1, -1)), sp=np.reshape(flux, (1, -1))))

    # =============================================================================
    # =============================================================================
    @classmethod
    def from_mask(cls, path, wave_range, allow_download=False):
        '''
        A line-mask template (e.g. a HARPS DRS ``.mas`` file) on the PHOENIX
        wavelength grid.

        The flux is 0 except inside each mask window, where it is the negative
        of the line's weight, so lines read as absorption. The windows are
        placed on the vacuum grid as given; no air/vacuum conversion is made.

        Parameters
        ----------
        path : str or path-like
            Text file of three columns: window start and end (Angstrom) and
            weight; '#' starts a comment. Where windows overlap, the later
            line wins.
        wave_range : (float, float)
            (lo, hi) wavelengths (Angstrom), applied as in ``from_phoenix``.
        allow_download : bool
            See ``from_phoenix``; only the PHOENIX wavelength grid is needed.

        Returns
        -------
        Template
            One order on the PHOENIX (vacuum) grid.

        Raises
        ------
        FileNotFoundError
            If the PHOENIX grid is not cached and downloads are disabled.
        OSError
            If the cached grid is corrupt and downloads are disabled.
        ValueError
            If ``wave_range`` is not inside the grid.
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
        Broaden every order with a Gaussian instrumental profile, in place.

        The kernel acts on flux - 1, with continuum 1 beyond the order's ends;
        for flux that is not continuum-normalized (e.g. PHOENIX) only the
        pixels within a kernel width of the ends are affected by that padding.
        The kernel has one velocity width, so it is applied on a uniform
        ln(lambda) grid: an order not on one (e.g. PHOENIX, linear in
        wavelength with a step change at 5000 A) is resampled onto ln(lambda)
        at its finest velocity step, broadened, and resampled back onto its
        own wavelengths. The grid of the model does not change.

        Parameters
        ----------
        resolution : float
            Resolving power R; the kernel FWHM is c / R (km/s).

        Returns
        -------
        Template
            self.
        '''
        def kernel(flux, dv):
            return dsp.broaden_gaussian(flux, dv, dsp.C_KMS / resolution)

        self.model.sp = _per_order(self.model.sp,
                                   [_broaden_velocity_even(w, s, kernel)
                                    for w, s in zip(self.model.wv, self.model.sp)])
        return self

    # =============================================================================
    # =============================================================================
    def broaden_rotational(self, vsini, epsilon):
        '''
        Broaden every order with a rotational profile (Gray 2005), in place.

        As ``broaden_instrumental``: the kernel acts on flux - 1, on a
        uniform ln(lambda) grid, and the model keeps its own grid.

        Parameters
        ----------
        vsini : float
            Projected rotational velocity (km/s).
        epsilon : float
            Linear limb-darkening coefficient (0-1).

        Returns
        -------
        Template
            self.
        '''
        def kernel(flux, dv):
            return dsp.broaden_rotational(flux, dv, vsini, epsilon)

        self.model.sp = _per_order(self.model.sp,
                                   [_broaden_velocity_even(w, s, kernel)
                                    for w, s in zip(self.model.wv, self.model.sp)])
        return self

    # =============================================================================
    # =============================================================================
    def doppler(self, vel, wv=None, velocity_convention='cz'):
        '''
        Doppler-shifted wavelengths of the template (or of given wavelengths).

        The model is not modified: pair the returned wavelengths with
        ``self.model.sp`` to get the shifted template. With ``wv=None`` the
        model must have a single order (or orders of equal length), so shift
        before ``cut_multiorder``.

        Parameters
        ----------
        vel : float or astropy.units.Quantity
            Radial velocity (km/s if a plain number); positive red-shifts.
        wv : array_like, optional
            Wavelengths (Angstrom) to shift; default the model's own.
        velocity_convention : {'cz', 'relativistic'}
            lambda * (1 + v/c) ('cz', default) or
            lambda * sqrt((1 + v/c) / (1 - v/c)), see
            ``unicor.dsp.doppler_factor``. Use the convention the RVs will be
            measured in.

        Returns
        -------
        ndarray
            The shifted wavelengths (Angstrom), shaped like the input.
        '''

        if type(vel) is not u.quantity.Quantity:
            vel = float(vel) << u.kilometer / u.second

        target = np.array(self.model.wv) if wv is None else wv
        return dsp.doppler_shift_wave(target, vel.to_value(u.km / u.s), velocity_convention)

    # =============================================================================
    # =============================================================================
    def integrate_spec(self, integration_ratio, wv=None, sp=None):
        '''
        Bin the model by summing groups of consecutive pixels, in place.

        Lowers the sampling (and the cost of a CCF) by ``integration_ratio``.
        Only the first order is used, and it yields
        ``n_pixels // integration_ratio - 1`` bins: the trailing pixels that
        fill no complete group, and the last complete group, are dropped. Each
        bin's wavelength is its group's mean and its flux the group's sum. A
        native step recorded for the model (``Spectrum.native_steps``) is
        coarsened to the bins' step if that is larger; explicit ``wv`` / ``sp``
        replace the model, so its record is dropped.

        Parameters
        ----------
        integration_ratio : int
            Pixels per bin.
        wv, sp : list of array_like, optional
            Wavelengths (Angstrom) and fluxes to bin instead of the model's;
            only their first order is used.

        Returns
        -------
        (ndarray, ndarray)
            The binned wavelengths (Angstrom) and fluxes, (1, n_bins) each;
            they also replace the model.
        '''
        new_sp = []
        new_wv = []

        native = None
        if wv is None:
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
            self.model.Info['NativeStep'] = [coarsest_native_step(native, new_wv[0])]

        return new_wv, new_sp

    # =============================================================================
    # =============================================================================
    def save_template_model(self, title="export_template", path='template_files/'):
        '''
        Export the template model, possibly after editing, as a NumPy ``.npz``.

        Each order is stored under its own keys -- ``wv_0``, ``sp_0``,
        ``wv_1``, ... -- so orders of different lengths round-trip exactly.

        Parameters
        ----------
        title : str
            Output file name, without the extension.
        path : str
            Existing directory to write into; a relative path is relative to
            the current working directory (not the package's template cache).

        Returns
        -------
        str
            The path of the written file.
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
        Cut the model into orders by wavelength bounds, in place.

        Every new order is cut from order 0 of the model, so a recorded native
        step (``Spectrum.native_steps``) is that of order 0 for each of them.

        Parameters
        ----------
        wv_bounds : array_like
            (n_orders, 2) lower and upper wavelength (Angstrom) of each order,
            both inclusive.

        Returns
        -------
        Template
            self.
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
        Cut the model into the orders of an observed spectrum, in place.

        Order i spans min(w_i) (1 - m/c) to max(w_i) (1 + m/c), w_i the
        observed wavelengths and m the margin, so a CCF scan of +-m km/s
        finds template at every lag (see ``cut_multiorder``).

        Parameters
        ----------
        spectrum : Spectrum
            The observation whose orders define the cut.
        margins : float
            Margin m beyond each order's ends (km/s). The default, 150 km/s,
            equals the default ``vel_bound`` of the RV engines.

        Returns
        -------
        Template
            self.
        '''

        wv_bounds = np.zeros((len(spectrum.wv), 2))
        margins = float(margins) << u.kilometer / u.second
        beta = (margins/spectrum.c).decompose().value

        for ordInd, w in enumerate(spectrum.wv):
            wv_bounds[ordInd, 0] = min(w)*(1-beta)
            wv_bounds[ordInd, 1] = max(w)*(1+beta)

        self.cut_multiorder(wv_bounds)

        return self


def phoenix_spectrum(teff, logg, feh=0.0, alpha=0.0, *, wave_range, air=False,
                     allow_download=False):
    '''
    A PHOENIX-ACES-AGSS-COND-2011 HiRes model (Husser et al. 2013) over a
    wavelength range, as plain arrays.

    The model is read from the local cache, TEMPLATE_FILES_PATH (the
    package's ``template_files/``), which ships (teff, logg, feh) =
    (4900, 4.5, 0.0), (5500, 4.5, 0.0) and (5800, 3.5, +0.5) and the
    wavelength grid. Any other model needs ``allow_download=True`` once. The
    flux is neither continuum-normalized nor broadened.

    Parameters
    ----------
    teff : float
        Effective temperature (K), a grid value.
    logg : float
        Log surface gravity (cgs), a grid value.
    feh : float
        Metallicity [Fe/H] (dex), a grid value.
    alpha : float
        Alpha enhancement [alpha/Fe] (dex), a grid value.
    wave_range : (float, float)
        (lo, hi) wavelengths (Angstrom, in the frame chosen by ``air``); the
        model keeps the grid points from the first above ``lo`` up to, not
        including, the first above ``hi``.
    air : bool
        PHOENIX wavelengths are in vacuum. True converts them to air (Morton
        2000, ApJS 130, 403) before ``wave_range`` is applied; use it for an
        observation in air (see the product's ``wavelength_frame`` in
        docs/instruments.md).
    allow_download : bool
        Fetch files missing from the cache, or replace cached ones that do not
        open as FITS with data (e.g. truncated), from the Goettingen server
        over HTTPS. Default False: a missing or corrupt file raises.

    Returns
    -------
    wave : ndarray
        1-D wavelengths (Angstrom; vacuum, or air if ``air``).
    flux : ndarray
        The PHOENIX flux density (erg/s/cm^2/cm) divided by 1e13, in the
        file's precision.

    Raises
    ------
    FileNotFoundError
        For a file missing from the cache with downloads disabled.
    OSError
        For a corrupt cached file with downloads disabled (it is not deleted),
        or a download that is not a FITS file with data.
    ValueError
        If ``wave_range`` is not inside the model grid, or a model parameter
        is not finite (or teff <= 0, logg < 0).
    '''
    flux_sub_path, flux_file = _phoenix_filename(teff, logg, feh, alpha)
    wave_path = _cached_phoenix_file(_PHOENIX_WAVE_SUB_PATH, PHOENIX_WAVE_FILE,
                                     allow_download)
    flux_path = _cached_phoenix_file(flux_sub_path, flux_file, allow_download)

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
    return w[cut], s[cut] / (10 ** 13)


def _phoenix_filename(teff, logg, feh, alpha):
    '''
    Server sub-path and file name of a PHOENIX-ACES HiRes model.

    PHOENIX names carry the metallicity sign explicitly ("Z-1.0", "Z+0.5"),
    so its magnitude is formatted; the effective temperature is an integer.

    Parameters
    ----------
    teff, logg, feh, alpha : float
        See ``Template.from_phoenix``.

    Returns
    -------
    (str, str)
        The server sub-path and the file name.

    Raises
    ------
    ValueError
        If a parameter is not a finite number, or teff <= 0 or logg < 0.
    '''
    try:
        teff, logg, feh, alpha = (float(v) for v in (teff, logg, feh, alpha))
    except (TypeError, ValueError) as err:
        raise ValueError("PHOENIX teff, logg, feh and alpha must be numbers; got "
                         f"{(teff, logg, feh, alpha)!r}") from err
    if not np.all(np.isfinite([teff, logg, feh, alpha])) or teff <= 0 or logg < 0:
        raise ValueError("PHOENIX teff, logg, feh and alpha must be finite, with teff > 0 K "
                         f"and logg >= 0; got {(teff, logg, feh, alpha)}")
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
    '''The HTTPS URL (str) of a PHOENIX file, from ``_phoenix_filename``'s output.'''
    return PHOENIX_URL + sub_path.strip('/') + '/' + filename


def _cache_path(filename):
    '''
    Path of a PHOENIX file directly inside TEMPLATE_FILES_PATH.

    Only the directory is resolved (``os.path.realpath``): a cached file may
    be a symlink to a shared model store, and ``os.replace`` onto it replaces
    the link, never its target.

    Parameters
    ----------
    filename : str
        A bare file name, as from ``_phoenix_filename``.

    Returns
    -------
    str
        The path in the resolved TEMPLATE_FILES_PATH.

    Raises
    ------
    ValueError
        If the path would not lie directly inside TEMPLATE_FILES_PATH.
    '''
    root = os.path.realpath(TEMPLATE_FILES_PATH)
    path = os.path.join(root, filename)
    if (os.path.realpath(os.path.dirname(path)) != root
            or os.path.basename(path) in ('', os.curdir, os.pardir)):
        raise ValueError(f"PHOENIX file name {filename!r} does not resolve to a file "
                         f"directly inside {root}")
    return path


def _is_readable_fits(path):
    '''
    Whether ``path`` opens as FITS with a non-empty primary data array.

    A cheap guard against a truncated or foreign file: the data are memory-
    mapped, not read, so a file shorter than its header declares fails here.

    Parameters
    ----------
    path : str

    Returns
    -------
    bool
    '''
    try:
        with warnings.catch_warnings():
            # astropy warns "File may have been truncated"; the caller raises instead.
            warnings.simplefilter('ignore', AstropyUserWarning)
            with fits.open(path) as hdul:
                return hdul[0].data is not None and hdul[0].data.size > 0
    except (OSError, ValueError, TypeError):
        return False


def _cached_phoenix_file(sub_path, filename, allow_download):
    '''
    Local path of a PHOENIX file, downloading it only if allowed.

    A cached file that does not open as FITS with data (e.g. truncated) is
    treated as missing: re-downloaded and replaced if allowed, otherwise
    reported -- never deleted.

    Parameters
    ----------
    sub_path, filename : str
        See ``_phoenix_filename``.
    allow_download : bool
        When False, a file missing from (or corrupt in) TEMPLATE_FILES_PATH
        raises.

    Returns
    -------
    str
        The path in TEMPLATE_FILES_PATH.

    Raises
    ------
    FileNotFoundError
        If the file is missing and downloads are disabled.
    OSError
        If the cached file is corrupt and downloads are disabled.
    ValueError
        If ``filename`` does not resolve to a file inside TEMPLATE_FILES_PATH.
    '''
    path = _cache_path(filename)
    if os.path.exists(path):
        if _is_readable_fits(path):
            return path
        if not allow_download:
            raise OSError(
                f"PHOENIX file {path} is corrupt or truncated (it does not open as FITS "
                "with data). Delete it, or pass allow_download=True to replace it.")
    elif not allow_download:
        raise FileNotFoundError(
            f"PHOENIX file '{filename}' is not in {TEMPLATE_FILES_PATH} "
            "and downloads are disabled (allow_download=False); pass "
            "allow_download=True to fetch it.")
    return _download_phoenix_file(sub_path, filename)


def _download_phoenix_file(sub_path, filename, timeout=120):
    '''
    Download one PHOENIX file from the Goettingen server into
    TEMPLATE_FILES_PATH, over HTTPS with astropy's default TLS verification.

    astropy.utils.data.download_file fetches into a system temporary file
    (possibly on another filesystem). Its content is copied into a temporary
    file in TEMPLATE_FILES_PATH, fsynced, checked to open as FITS with data,
    and only then renamed onto the final name with ``os.replace`` (atomic on
    one filesystem). An interrupted or failed run therefore leaves the final
    path untouched: absent, or the previous file. Both temporary files are
    removed in every case. Any download error propagates to the caller.

    Parameters
    ----------
    sub_path : str
        Server sub-path (see ``_phoenix_filename``).
    filename : str
        File name to fetch.
    timeout : float
        Per-request timeout (s).

    Returns
    -------
    str
        The local path of the downloaded file.

    Raises
    ------
    OSError
        If the downloaded content does not open as FITS with data.
    ValueError
        If ``filename`` does not resolve to a file inside TEMPLATE_FILES_PATH.
    '''
    target = _cache_path(filename)
    cache_dir = os.path.dirname(target)
    os.makedirs(cache_dir, exist_ok=True)

    url = _phoenix_url(sub_path, filename)
    fetched = download_file(url, cache=False, timeout=timeout, show_progress=False)
    partial = None
    try:
        with open(fetched, 'rb') as src, tempfile.NamedTemporaryFile(
                dir=cache_dir, prefix='.' + filename + '.', suffix='.part',
                delete=False) as dst:
            partial = dst.name
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        if not _is_readable_fits(partial):
            raise OSError(f"the file downloaded from {url} does not open as FITS with data; "
                          "nothing was written to the cache")
        os.replace(partial, target)
        partial = None
    finally:
        for leftover in (partial, fetched):
            if leftover is not None and os.path.exists(leftover):
                os.remove(leftover)
    return target


def _wave_slice(wave, wave_range):
    '''
    Index range of a PHOENIX grid: from the first point above ``lo`` up to,
    not including, the first point above ``hi``.

    Parameters
    ----------
    wave : ndarray
        Increasing wavelength grid (Angstrom).
    wave_range : (float, float)
        (lo, hi), Angstrom.

    Returns
    -------
    slice

    Raises
    ------
    ValueError
        Unless lo < hi < wave[-1].
    '''
    lo, hi = wave_range
    if not lo < hi < wave[-1]:
        raise ValueError(f"wave_range={tuple(wave_range)} must satisfy lo < hi < "
                         f"{wave[-1]:.1f} A (the end of the PHOENIX grid)")
    return slice(int(np.argmax(wave > lo)), int(np.argmax(wave > hi)))
