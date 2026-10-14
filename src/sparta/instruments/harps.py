"""HARPS (ESO 3.6 m, La Silla): ESO Phase-3 1-D spectra and CERES reductions."""
import numpy as np
from astropy.io import fits

from sparta.instruments import _ceres
from sparta.instruments._common import _wavelength_window
from sparta.instruments._eso import _bcv_for_specsys, _eso_sdp_frames
from sparta.instruments.base import Instrument, LoadedSpectrum, Product, Recommended

HARPS = Instrument(
    name="HARPS",
    # ESO instrument description: "RS=120,000 (measured)", sampled at
    # "4.1 px per FWHM", so one detector pixel spans c / (R * 4.1) km/s.
    resolving_power=120000.0,
    native_step_kms=299792.458 / (120000.0 * 4.1),
    references=(
        "https://www.eso.org/sci/facilities/lasilla/instruments/harps/inst/description.html",
        "https://www.eso.org/sci/facilities/lasilla/instruments/harps/overview.html",
        "https://www.eso.org/sci/observing/phase3/p3sdpstd.pdf",
    ),
    notes="R and the native step follow the ESO instrument description: R_S = 120,000 "
          "(measured) and 4.1 px per FWHM, i.e. ~0.609 km/s per detector pixel. The ESO "
          "overview page and the ADP SPEC_RES header quote the nominal R = 115,000. "
          "Recommended settings are those of examples/TIRAVEL_Quickstart.ipynb, not "
          "instrument documentation.",
)

# UNVERIFIED: whether the HARPS DRS BJD is on the UTC or TDB scale.
_SDP_TIME_SYSTEM = "BJD (HARPS DRS; scale unverified)"


def load_sdp(path, *, min_wv, max_wv):
    '''
    Read a HARPS ESO Phase-3 SDP 1-D spectrum (HDU1 binary table with WAVE /
    FLUX / ERR columns), trimmed to a wavelength window.

    The SDP s1d product is barycentric (SPECSYS) in air (TUCD1).

    Parameters
    ----------
    path : str or path-like
        ADP FITS file.
    min_wv, max_wv : float
        Window (Angstrom); pixels with ``min_wv < w <= max_wv`` are kept.

    Returns
    -------
    LoadedSpectrum
        One order; ``time`` is HIERARCH ESO DRS BJD (days), ``bcv`` (km/s) is
        0 for a barycentric product, else HIERARCH ESO DRS BERV.

    Raises
    ------
    ValueError
        If the window does not overlap the spectrum or SPECSYS is missing.
    '''
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        wave = np.asarray(hdul[1].data['WAVE'][0], float)
        flux = np.asarray(hdul[1].data['FLUX'][0], float)
        window = _wavelength_window(wave, min_wv, max_wv)

        specsys, wavelength_frame = _eso_sdp_frames(hdul)
        bcv = _bcv_for_specsys(header, specsys, 'HIERARCH ESO DRS BERV')
        try:
            air_mass = (header['HIERARCH ESO TEL AIRM START']
                        + header['HIERARCH ESO TEL AIRM END']) / 2
        except KeyError:
            air_mass = np.nan
        return LoadedSpectrum(
            wave=[wave[window]], flux=[flux[window]],
            time=float(header['HIERARCH ESO DRS BJD']), time_system=_SDP_TIME_SYSTEM,
            bcv=bcv, specsys=specsys, wavelength_frame=wavelength_frame,
            snr=float(header['SNR']), air_mass=float(air_mass),
            survey="HARPS", path=str(path))


def load_ceres(path, *, min_wv, max_wv):
    '''
    Read HARPS data reduced with CERES (see `sparta.instruments._ceres`).

    CERES converts the HARPS wavelengths to vacuum (harps/harpspipe.py,
    ToVacuum). UNVERIFIED: identification of these files as CERES output
    rests on their header keywords (MBJD, BARYCENTRIC CORRECTION).

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
                             survey="HARPS-CERES", wavelength_frame="vacuum")


PRODUCTS = (
    Product(
        survey="HARPS",
        instrument=HARPS,
        description="ESO Phase-3 SDP 1-D spectrum (ADP.*.fits, HARPS DRS s1d): one-row "
                    "binary table in HDU1 with WAVE / FLUX / ERR columns.",
        resampled=True,
        grid="linear, 0.01 Angstrom per pixel (SPEC_BIN = 0.001 nm)",
        wavelength_frame="air",
        specsys="BARYCENT",
        time_system=_SDP_TIME_SYSTEM,
        multi_order=False,
        recommended=Recommended(
            dv_kms=0.1, window_A=(4990.0, 5090.0),
            preprocess=dict(n_trim=10, clean_margins=True, remove_nans=True, delta=0.5,
                            cosmic_sigma=3, filter_lowcut=3, filter_highcut=0.15, alpha=0.3)),
        verified=True,
        load=load_sdp,
        grid_step_A=0.01,
    ),
    Product(
        survey="HARPS-CERES",
        instrument=HARPS,
        description="HARPS spectrum reduced with the CERES pipeline: a cube whose data[0] / "
                    "data[3] hold the per-order wavelength / blaze-corrected flux; the order "
                    "that best covers the window is returned.",
        resampled=False,
        grid="native echelle pixels (one order)",
        wavelength_frame="vacuum",
        specsys="BARYCENT",
        time_system=_ceres.TIME_SYSTEM,
        multi_order=False,
        recommended=Recommended(),
        verified=False,
        load=load_ceres,
    ),
)
