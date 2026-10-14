"""CORALIE (1.2 m Euler telescope, La Silla): Geneva DRS 1-D spectra."""
import numpy as np
from astropy.io import fits

from sparta.instruments._common import _wavelength_window
from sparta.instruments.base import Instrument, LoadedSpectrum, Product, Recommended

CORALIE = Instrument(
    name="CORALIE",
    resolving_power=None,
    native_step_kms=None,
    notes="Resolving power and pixel size not looked up.",
)


def load_s1d(path, *, min_wv, max_wv):
    '''
    Read a CORALIE Geneva-DRS s1d image with a linear FITS WCS,
    lambda = CRVAL1 + CDELT1 * (p - CRPIX1) for 1-based pixel p.

    UNVERIFIED: barycentric, air wavelengths -- assumed from the HARPS s1d
    product of the same DRS family (whose SDP header says so).

    Parameters
    ----------
    path : str or path-like
        s1d FITS file.
    min_wv, max_wv : float
        Window (Angstrom); pixels with ``min_wv < w <= max_wv`` are kept.

    Returns
    -------
    LoadedSpectrum
        One order; ``time`` is HIERARCH ESO DRS BJD (days), ``bcv`` = 0 km/s.
    '''
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        s = np.array(hdul[0].data, dtype=float)
    pix = np.arange(1, s.size + 1)
    w = float(header["CRVAL1"]) + float(header["CDELT1"]) * (pix - float(header.get('CRPIX1', 1.0)))
    window = _wavelength_window(w, min_wv, max_wv)
    return LoadedSpectrum(
        wave=[w[window]], flux=[s[window]],
        time=float(header["HIERARCH ESO DRS BJD"]),
        time_system='BJD (Geneva DRS; scale unverified)',
        bcv=0.0, specsys='BARYCENT', wavelength_frame=None,
        survey="CORALIE", path=str(path))


PRODUCTS = (
    Product(
        survey="CORALIE",
        instrument=CORALIE,
        description="Geneva DRS s1d image in the primary HDU with a linear WCS "
                    "(CRVAL1 / CDELT1 / CRPIX1).",
        resampled=True,
        grid="linear, CDELT1 Angstrom per pixel",
        wavelength_frame=None,
        specsys="BARYCENT",
        time_system="BJD (Geneva DRS; scale unverified)",
        multi_order=False,
        recommended=Recommended(),
        verified=False,
        load=load_s1d,
    ),
)
