"""Reader for CERES pipeline cubes (Brahm et al. 2017), shared by HARPS and FEROS."""
import numpy as np
from astropy.io import fits

from sparta.instruments._common import _best_order
from sparta.instruments.base import LoadedSpectrum

TIME_SYSTEM = "MBJD-UTC"


def load_ceres(path, *, min_wv, max_wv, survey, wavelength_frame):
    '''
    Read a CERES cube (https://github.com/rabrahm/ceres): data[0] wavelength
    and data[3] blaze-corrected flux, per order.

    CERES multiplies the wavelength solution by lambda_bary/lambda_topo, so
    the spectra are barycentric; HIERARCH MBJD is BJD - 2400000.5 computed
    from the UTC MJD.

    Parameters
    ----------
    path : str or path-like
        CERES FITS cube.
    min_wv, max_wv : float
        Window (Angstrom); the order with the most pixels in
        ``min_wv < w <= max_wv`` is returned, trimmed to it.
    survey : str
        Registry key to record.
    wavelength_frame : str or None
        Air/vacuum frame of this instrument's CERES solution.

    Returns
    -------
    LoadedSpectrum
        One order, barycentric (``bcv`` = 0 km/s); ``time`` is MBJD (days).

    Raises
    ------
    ValueError
        If no order overlaps the window.
    '''
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        data = np.array(hdul[0].data, dtype=float)
    w, s = _best_order(data[0], data[3], min_wv, max_wv)
    return LoadedSpectrum(
        wave=[w], flux=[s],
        time=float(header['HIERARCH MBJD']), time_system=TIME_SYSTEM,
        bcv=0.0, specsys='BARYCENT', wavelength_frame=wavelength_frame,
        snr=float(header["SNR"]), vrad=float(header["RV"]),
        survey=survey, path=str(path))
