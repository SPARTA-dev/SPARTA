"""APOGEE (SDSS, H band): apVisit spectra."""
import numpy as np
from astropy.io import fits

from sparta.instruments._common import _ascending
from sparta.instruments.base import Instrument, LoadedSpectrum, Product, Recommended

APOGEE = Instrument(
    name="APOGEE",
    resolving_power=22500.0,
    native_step_kms=None,
    references=("https://www.sdss4.org/dr17/irspec/spectra/",),
    notes="Native pixel size not looked up.",
)

# APOGEE_PIXMASK bits 12 (SIG_SKYLINE) and 13 (SIG_TELLURIC),
# https://www.sdss.org/dr17/algorithms/bitmasks/#APOGEE_PIXMASK
_SKY_OR_TELLURIC = (1 << 12) | (1 << 13)


def APOGEE_masking(flux, pixmask):
    '''
    Fill APOGEE pixels flagged as sky line or telluric (and NaN pixels) by
    linear interpolation in pixel index.

    Masked runs at the edges take the nearest good value instead of zero
    flux, which a CCF would read as a deep line.

    Parameters
    ----------
    flux : array_like
        1-D flux.
    pixmask : array_like of int
        APOGEE_PIXMASK value per pixel.

    Returns
    -------
    ndarray
        float64 copy of ``flux`` with the masked pixels filled.

    Raises
    ------
    ValueError
        If every pixel is masked.
    '''
    flux = np.array(flux, dtype=float)
    bad = (np.asarray(pixmask) & _SKY_OR_TELLURIC) != 0
    bad |= ~np.isfinite(flux)
    if bad.all():
        raise ValueError("every APOGEE pixel is masked; nothing to interpolate from")
    idx = np.arange(len(flux))
    flux[bad] = np.interp(idx[bad], idx[~bad], flux[~bad])
    return flux


def load_apvisit(path, *, min_wv, max_wv, segment=1):
    '''
    Read one chip of an APOGEE apVisit file (SDSS DR17 data model,
    https://data.sdss.org/datamodel/files/APOGEE_REDUX/APRED_VERS/visit/TELESCOPE/FIELD/PLATE_ID/MJD5/apVisit.html):
    HDU1 flux, HDU3 APOGEE_PIXMASK, HDU4 wavelength, one row per chip.

    Wavelengths are vacuum (https://www.sdss4.org/dr17/irspec/spectra/) in
    the observer frame: the header has VHELIO = VRAD + BC, so BC (km/s) is
    the bcv. Rows are reordered to ascending wavelength and sky / telluric
    pixels are filled by `APOGEE_masking`.

    Parameters
    ----------
    path : str or path-like
        apVisit FITS file.
    min_wv, max_wv : float
        Ignored (the whole chip is returned).
    segment : int
        Chip row to read (0, 1 or 2).

    Returns
    -------
    LoadedSpectrum
        One order; ``time`` is JD-MID (days), ``vrad`` is VRAD (km/s);
        ``extras['pixmask']`` holds the APOGEE_PIXMASK of the returned pixels.
    '''
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        w, s, pixmask = _ascending(hdul[4].data[segment],
                                   np.array(hdul[1].data[segment], dtype=float),
                                   np.array(hdul[3].data[segment]))
    return LoadedSpectrum(
        wave=[w], flux=[APOGEE_masking(s, pixmask)],
        # UNVERIFIED: the time scale of JD-MID (assumed UTC).
        time=float(header['JD-MID']), time_system='JD-UTC',
        bcv=float(header['BC']), specsys='TOPOCENT', wavelength_frame='vacuum',
        vrad=float(header['VRAD']), survey="APOGEE", path=str(path),
        extras={"pixmask": pixmask})


PRODUCTS = (
    Product(
        survey="APOGEE",
        instrument=APOGEE,
        description="APOGEE apVisit file: flux (HDU1), APOGEE_PIXMASK (HDU3) and "
                    "wavelength (HDU4), one row per chip; one chip is read.",
        resampled=False,
        grid="native pixels of one detector chip",
        wavelength_frame="vacuum",
        specsys="TOPOCENT",
        time_system="JD-UTC",
        multi_order=False,
        recommended=Recommended(),
        verified=False,
        load=load_apvisit,
    ),
)
