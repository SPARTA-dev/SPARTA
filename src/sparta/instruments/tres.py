"""TRES (FLWO 1.5 m Tillinghast Reflector Echelle Spectrograph)."""
import numpy as np
from astropy.io import fits

from sparta.instruments._common import _best_order
from sparta.instruments._iraf import _iraf_multispec_wavelengths
from sparta.instruments.base import Instrument, LoadedSpectrum, Product, Recommended

TRES = Instrument(
    name="TRES",
    resolving_power=44000.0,
    native_step_kms=None,
    references=("http://tdc-www.harvard.edu/instruments/tres/",),
    notes="R for the medium fiber; native pixel size not looked up.",
)


def load_iraf(path, *, min_wv, max_wv):
    '''
    Read a TRES IRAF echelle spectrum: one order per image line, with a
    non-linear multispec dispersion function per order in the header
    (http://tdc-www.harvard.edu/instruments/tres/reduce.html).

    BCV (km/s) is "the barycentric velocity correction" added by the CfA
    reduction (http://tdc-www.harvard.edu/instruments/tres/process/process2018.html).
    UNVERIFIED: WS_BJD and BCV are not in the public TRES documentation
    (which lists HJDN/GJDN/MJD-MID); BCV assumed in km/s, WS_BJD a BJD of
    unknown scale, and ThAr (IRAF) solutions in air.

    Parameters
    ----------
    path : str or path-like
        TRES FITS file.
    min_wv, max_wv : float
        Window (Angstrom); the order with the most pixels in
        ``min_wv < w <= max_wv`` is returned, trimmed to it.

    Returns
    -------
    LoadedSpectrum
        One order; ``time`` is WS_BJD (days), ``bcv`` is BCV (km/s).

    Raises
    ------
    ValueError
        If the image is not 2-D, the apertures do not match the image lines,
        or no order overlaps the window.
    '''
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        data = np.array(hdul[0].data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"TRES loader expects a 2-D (order, pixel) image, got shape {data.shape}")
    waves = _iraf_multispec_wavelengths(header)
    if sorted(waves) != list(range(1, data.shape[0] + 1)):
        raise ValueError(f"multispec apertures {sorted(waves)} do not match the "
                         f"{data.shape[0]} image lines")
    w, s = _best_order([waves[n] for n in range(1, data.shape[0] + 1)],
                       [data[n - 1, :waves[n].size] for n in range(1, data.shape[0] + 1)],
                       min_wv, max_wv)
    return LoadedSpectrum(
        wave=[w], flux=[s],
        time=float(header["WS_BJD"]), time_system='BJD (scale unverified)',
        bcv=float(header["BCV"]), specsys='TOPOCENT', wavelength_frame=None,
        survey="TRES", path=str(path))


PRODUCTS = (
    Product(
        survey="TRES",
        instrument=TRES,
        description="IRAF echelle image, one order per line, with a multispec WCS "
                    "(WAT2 cards); the order that best covers the window is returned.",
        resampled=False,
        grid="native echelle pixels (one order)",
        wavelength_frame=None,
        specsys="TOPOCENT",
        time_system="BJD (scale unverified)",
        multi_order=False,
        recommended=Recommended(),
        verified=False,
        load=load_iraf,
    ),
)
