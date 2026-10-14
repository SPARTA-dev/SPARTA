"""FEROS (MPG/ESO 2.2 m, La Silla): ESO Phase-3 1-D spectra and CERES reductions."""
import numpy as np
from astropy.io import fits

from sparta.instruments import _ceres
from sparta.instruments._common import _wavelength_window
from sparta.instruments._eso import _bcv_for_specsys, _eso_sdp_frames
from sparta.instruments.base import Instrument, LoadedSpectrum, Product, Recommended

FEROS = Instrument(
    name="FEROS",
    resolving_power=48000.0,
    native_step_kms=None,
    references=("https://www.eso.org/sci/facilities/lasilla/instruments/feros.html",),
    notes="Native pixel size not looked up.",
)


def load_sdp(path, *, min_wv, max_wv):
    '''
    Read a FEROS ESO Phase-3 SDP 1-D spectrum, trimmed to a window.

    The FEROS release description
    (http://www.eso.org/rm/api/v1/public/releaseDescriptions/94) states that
    the HIERARCH ESO DRS BARYCORR correction is already applied to the spectra
    and that wavelengths are in air, so the BCV is applied only if SPECSYS
    says the product is topocentric.

    Parameters
    ----------
    path : str or path-like
        ADP FITS file.
    min_wv, max_wv : float
        Window (Angstrom); pixels with ``min_wv < w <= max_wv`` are kept.

    Returns
    -------
    LoadedSpectrum
        One order; ``time`` is the exposure midpoint (MJD-OBS + MJD-END) / 2
        (days); ``bcv`` in km/s.

    Raises
    ------
    ValueError
        If SPECSYS is missing or the window does not overlap the spectrum.
    '''
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        table = hdul[1].data
        w = np.array(table['WAVE'][0], dtype=float)
        s = np.array(table['FLUX'][0], dtype=float)
        specsys, wavelength_frame = _eso_sdp_frames(hdul)
    bcv = _bcv_for_specsys(header, specsys, 'HIERARCH ESO DRS BARYCORR')
    window = _wavelength_window(w, min_wv, max_wv)
    return LoadedSpectrum(
        wave=[w[window]], flux=[s[window]],
        time=float(header["MJD-OBS"] + header["MJD-END"]) / 2, time_system="MJD-UTC",
        bcv=bcv, specsys=specsys, wavelength_frame=wavelength_frame,
        snr=float(header["SNR"]), survey="FEROS", path=str(path))


def load_ceres(path, *, min_wv, max_wv):
    '''
    Read FEROS data reduced with CERES (see `sparta.instruments._ceres`).

    UNVERIFIED: CERES' FEROS solution is not passed through ToVacuum
    (feros/ferospipe.py) but its CCF masks are, so the air/vacuum frame is
    unclear; identification of these files as CERES output rests on their
    header keywords (MBJD, BARYCENTRIC CORRECTION).

    Parameters
    ----------
    path : str or path-like
        CERES FITS cube.
    min_wv, max_wv : float
        Window (Angstrom) selecting the best-covering order.

    Returns
    -------
    LoadedSpectrum
    '''
    return _ceres.load_ceres(path, min_wv=min_wv, max_wv=max_wv,
                             survey="FEROS-CERES", wavelength_frame=None)


PRODUCTS = (
    Product(
        survey="FEROS",
        instrument=FEROS,
        description="ESO Phase-3 SDP 1-D spectrum (ADP.*.fits): one-row binary table in "
                    "HDU1 with WAVE / FLUX / ERR columns.",
        resampled=True,
        grid="1-D merged spectrum as delivered by the ESO pipeline",
        wavelength_frame="air",
        specsys="BARYCENT",
        time_system="MJD-UTC",
        multi_order=False,
        recommended=Recommended(),
        verified=False,
        load=load_sdp,
    ),
    Product(
        survey="FEROS-CERES",
        instrument=FEROS,
        description="FEROS spectrum reduced with the CERES pipeline: a cube whose data[0] / "
                    "data[3] hold the per-order wavelength / blaze-corrected flux; the order "
                    "that best covers the window is returned.",
        resampled=False,
        grid="native echelle pixels (one order)",
        wavelength_frame=None,
        specsys="BARYCENT",
        time_system=_ceres.TIME_SYSTEM,
        multi_order=False,
        recommended=Recommended(),
        verified=False,
        load=load_ceres,
    ),
)
