"""UVES (ESO VLT): ESO Phase-3 1-D echelle spectra."""
import numpy as np
from astropy.io import fits

from sparta.instruments._common import _wavelength_window
from sparta.instruments._eso import _bcv_for_specsys, _eso_sdp_frames
from sparta.instruments.base import Instrument, LoadedSpectrum, Product, Recommended

UVES = Instrument(
    name="UVES",
    resolving_power=None,
    native_step_kms=None,
    references=("https://www.eso.org/sci/facilities/paranal/instruments/uves.html",),
    notes="Resolving power depends on the slit width and is not recorded here.",
)


def load_sdp(path, *, min_wv, max_wv):
    '''
    Read a UVES ESO Phase-3 SDP 1-D spectrum, trimmed to a window.

    Per the UVES echelle release description
    (https://www.eso.org/sci/observing/phase3/data_releases/IDP_UVES_ECH_release_description_v1.3.pdf)
    no barycentric correction is applied to the wavelengths; it is stored in
    HIERARCH ESO QC VRAD BARYCOR (km/s). Uses the flux-calibrated FLUX column
    when present, else FLUX_REDUCED.

    Parameters
    ----------
    path : str or path-like
        ADP FITS file.
    min_wv, max_wv : float
        Window (Angstrom); pixels with ``min_wv < w <= max_wv`` are kept.

    Returns
    -------
    LoadedSpectrum
        One order; ``time`` is MJD-OBS, the exposure start (days); ``bcv``
        in km/s.

    Raises
    ------
    ValueError
        If SPECSYS is missing or the window does not overlap the spectrum.
    '''
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        table = hdul[1].data
        flux_column = 'FLUX' if 'FLUX' in table.columns.names else 'FLUX_REDUCED'
        w = np.array(table['WAVE'][0], dtype=float)
        s = np.array(table[flux_column][0], dtype=float)
        specsys, wavelength_frame = _eso_sdp_frames(hdul)
    bcv = _bcv_for_specsys(header, specsys, 'HIERARCH ESO QC VRAD BARYCOR')
    window = _wavelength_window(w, min_wv, max_wv)
    return LoadedSpectrum(
        wave=[w[window]], flux=[s[window]],
        time=float(header["MJD-OBS"]), time_system='MJD-UTC',
        bcv=bcv, specsys=specsys, wavelength_frame=wavelength_frame,
        snr=float(header["SNR"]), survey="UVES", path=str(path))


PRODUCTS = (
    Product(
        survey="UVES",
        instrument=UVES,
        description="ESO Phase-3 SDP 1-D echelle spectrum (ADP.*.fits): one-row binary "
                    "table in HDU1 with WAVE and FLUX (or FLUX_REDUCED) columns.",
        resampled=True,
        grid="1-D merged spectrum as delivered by the ESO pipeline",
        wavelength_frame="air",
        specsys="TOPOCENT",
        time_system="MJD-UTC",
        multi_order=False,
        recommended=Recommended(),
        verified=False,
        load=load_sdp,
    ),
)
