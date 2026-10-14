"""SRG: 1-D spectra with a linear FITS WCS (product undocumented)."""
import numpy as np
from astropy.io import fits

from sparta.instruments._common import _wavelength_window
from sparta.instruments.base import Instrument, LoadedSpectrum, Product, Recommended

SRG = Instrument(
    name="SRG",
    resolving_power=None,
    native_step_kms=None,
    notes="No documentation found for this product; the instrument is not identified.",
)


def load_linear(path, *, min_wv, max_wv):
    '''
    Read an SRG 1-D image with a linear FITS WCS,
    lambda = CRVAL1 + CDELT1 * (p - CRPIX1) for 1-based pixel p.

    UNVERIFIED: no documentation found for this product; frame, time scale
    and whether the spectra are barycentric are unknown.

    Parameters
    ----------
    path : str or path-like
        FITS file.
    min_wv, max_wv : float
        Window (Angstrom); pixels with ``min_wv < w <= max_wv`` are kept.

    Returns
    -------
    LoadedSpectrum
        One order; ``time`` is BJD (days), ``bcv`` = 0 km/s, ``air_mass`` is
        AIRMASS (NaN if absent).
    '''
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        s = np.array(hdul[0].data, dtype=float)
    pix = np.arange(1, s.size + 1)
    w = float(header["CRVAL1"]) + float(header["CDELT1"]) * (pix - float(header.get('CRPIX1', 1.0)))
    window = _wavelength_window(w, min_wv, max_wv)
    return LoadedSpectrum(
        wave=[w[window]], flux=[s[window]],
        time=float(header["BJD"]), time_system='BJD (scale unverified)',
        bcv=0.0, specsys=None, wavelength_frame=None,
        air_mass=float(header.get("AIRMASS", np.nan)),
        survey="SRG", path=str(path))


PRODUCTS = (
    Product(
        survey="SRG",
        instrument=SRG,
        description="1-D image in the primary HDU with a linear WCS (CRVAL1 / CDELT1 / CRPIX1).",
        resampled=True,
        grid="linear, CDELT1 Angstrom per pixel",
        wavelength_frame=None,
        specsys=None,
        time_system="BJD (scale unverified)",
        multi_order=False,
        recommended=Recommended(),
        verified=False,
        load=load_linear,
    ),
)
