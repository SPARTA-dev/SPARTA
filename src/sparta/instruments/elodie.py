"""ELODIE (1.93 m, Observatoire de Haute-Provence): archive 'spec' product."""
import warnings

import numpy as np
from astropy.io import fits
from astropy.time import Time

from sparta.instruments._common import _wavelength_window
from sparta.instruments.base import Instrument, LoadedSpectrum, Product, Recommended

ELODIE = Instrument(
    name="ELODIE",
    resolving_power=42000.0,
    native_step_kms=None,
    references=("https://ui.adsabs.harvard.edu/abs/1996A%26AS..119..373B",),
    notes="Native pixel size not looked up.",
)


def load_spec(path, *, min_wv, max_wv):
    '''
    Read an ELODIE archive 'spec' file: one reconnected spectrum resampled to
    a constant step (0.05 Angstrom, 4000-6800 Angstrom) with a linear FITS
    WCS, lambda = CRVAL1 + CDELT1 * (p - CRPIX1) for 1-based pixel p.

    The archive documents these spectra as topocentric, "not corrected for
    the Earth's motion"
    (https://web.archive.org/web/2015/http://www.obs-hp.fr/archive/elodie-for-dummies.shtml);
    no BCV keyword is known, so none is applied and a warning is issued.

    Parameters
    ----------
    path : str or path-like
        ELODIE FITS file.
    min_wv, max_wv : float
        Window (Angstrom); pixels with ``min_wv < w <= max_wv`` are kept.

    Returns
    -------
    LoadedSpectrum
        One order; ``time`` is DATE-OBS as a UTC JD (days), ``bcv`` = 0 km/s.
    '''
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        s = np.array(hdul[0].data, dtype=float)
    pix = np.arange(1, s.size + 1)
    w = float(header['CRVAL1']) + float(header['CDELT1']) * (pix - float(header.get('CRPIX1', 1.0)))
    window = _wavelength_window(w, min_wv, max_wv)
    warnings.warn("ELODIE archive spectra are topocentric and SPARTA knows no BCV keyword "
                  "for them: no barycentric correction is applied.", UserWarning, stacklevel=3)
    return LoadedSpectrum(
        wave=[w[window]], flux=[s[window]],
        # UNVERIFIED: DATE-OBS taken as exposure start, UTC.
        time=Time(header['DATE-OBS'], format='isot', scale='utc').jd, time_system='JD-UTC',
        bcv=0.0, specsys='TOPOCENT',
        # UNVERIFIED: air wavelengths (the ELODIE library is in air; the archive
        # spec product is not documented either way).
        wavelength_frame=None,
        snr=float(header["SN"]), survey="ELODIE", path=str(path))


PRODUCTS = (
    Product(
        survey="ELODIE",
        instrument=ELODIE,
        description="ELODIE archive 'spec' file: one reconnected 1-D spectrum in the "
                    "primary HDU with a linear WCS (CRVAL1 / CDELT1 / CRPIX1).",
        resampled=True,
        grid="linear, 0.05 Angstrom per pixel, 4000-6800 Angstrom",
        wavelength_frame=None,
        specsys="TOPOCENT",
        time_system="JD-UTC",
        multi_order=False,
        recommended=Recommended(),
        verified=False,
        load=load_spec,
    ),
)
