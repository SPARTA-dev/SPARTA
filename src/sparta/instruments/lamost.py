"""LAMOST (Guoshoujing telescope) low-resolution survey spectra."""
import numpy as np
from astropy.io import fits
from astropy.time import Time

from sparta.instruments._common import _wavelength_window
from sparta.instruments.base import Instrument, LoadedSpectrum, Product, Recommended

LAMOST = Instrument(
    name="LAMOST",
    resolving_power=1800.0,
    native_step_kms=None,
    references=("https://www.lamost.org/dr5/v3/doc/data-production-description",),
    notes="Low-resolution mode; native pixel size not looked up.",
)


def load_lrs(path, *, min_wv, max_wv):
    '''
    Read a LAMOST low-resolution spectrum (DR5 data model,
    https://www.lamost.org/dr5/v3/doc/data-production-description; same
    layout in DR9 v2.0): primary array row 0 is the flux; the log-linear
    dispersion is log10(lambda_i) = COEFF0 + COEFF1 * i for 0-based pixel i
    (CRPIX1 = 1, "central wavelength (log10) of first pixel").

    VACUUM=T marks vacuum wavelengths; HELIO=T marks spectra already shifted
    to the heliocentric frame (HELIO_RV is the velocity used). DATE-OBS is the
    median UTC of the exposures.

    Parameters
    ----------
    path : str or path-like
        LAMOST FITS file.
    min_wv, max_wv : float
        Window (Angstrom); pixels with ``min_wv < w <= max_wv`` are kept.

    Returns
    -------
    LoadedSpectrum
        One order, heliocentric (``bcv`` = 0 km/s); ``time`` is a UTC JD (days).

    Raises
    ------
    ValueError
        If the primary array is not 2-D or HELIO is missing.
    NotImplementedError
        For a spectrum that is not heliocentric-corrected (HELIO=F).
    '''
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        data = hdul[0].data
        if data is None or data.ndim != 2:
            raise ValueError("LAMOST loader expects the primary data array "
                             "(flux, inverse variance, wavelength, andmask, ormask)")
        s = np.array(data[0], dtype=float)
    w = 10 ** (float(header['COEFF0']) + float(header['COEFF1']) * np.arange(s.size))

    if 'HELIO' not in header:
        raise ValueError("LAMOST header has no HELIO keyword: cannot tell whether the "
                         "spectrum is heliocentric-corrected")
    if not header['HELIO']:
        raise NotImplementedError("LAMOST spectrum is not heliocentric-corrected (HELIO=F); "
                                  "the sign convention of HELIO_RV is undocumented")

    window = _wavelength_window(w, min_wv, max_wv)
    return LoadedSpectrum(
        wave=[w[window]], flux=[s[window]],
        time=Time(header['DATE-OBS'], format='isot', scale='utc').jd, time_system='JD-UTC',
        bcv=0.0, specsys='HELIOCEN',
        wavelength_frame={True: 'vacuum', False: 'air'}.get(header.get('VACUUM')),
        survey="LAMOST", path=str(path))


PRODUCTS = (
    Product(
        survey="LAMOST",
        instrument=LAMOST,
        description="LAMOST low-resolution spectrum: primary array (flux, inverse variance, "
                    "wavelength, andmask, ormask) with a log-linear COEFF0 / COEFF1 dispersion.",
        resampled=True,
        grid="log-linear, COEFF1 dex per pixel (1e-4 in DR5)",
        wavelength_frame="vacuum",
        specsys="HELIOCEN",
        time_system="JD-UTC",
        multi_order=False,
        recommended=Recommended(),
        verified=False,
        load=load_lrs,
    ),
)
