"""Records shared by every instrument module: what an instrument is, what one
of its archive products is, and what loading a file of that product returns.

Each instrument module exports ``PRODUCTS``, a tuple of `Product`; the package
registry (``sparta.instruments.REGISTRY``) is keyed by `Product.survey`.

Conventions: wavelengths in Angstrom, velocities in km/s, times in days.
"""
import dataclasses
import functools
import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

WAVELENGTH_FRAMES = ("air", "vacuum", None)
SPECSYS_VALUES = ("BARYCENT", "HELIOCEN", "TOPOCENT", "SOURCE", None)


@dataclass(frozen=True)
class Instrument:
    """
    A spectrograph.

    Parameters
    ----------
    name : str
        Instrument name.
    resolving_power : float or None
        Spectral resolving power R = lambda / delta-lambda (dimensionless).
    native_step_kms : float or None
        Detector pixel size in velocity (km/s per pixel), i.e. the sampling
        before any archive resampling.
    references : tuple of str
        URLs / bibcodes the values above come from.
    notes : str
        Caveats, in particular any value that is not documented.
    """
    name: str
    resolving_power: Optional[float]
    native_step_kms: Optional[float]
    references: tuple = ()
    notes: str = ""


@dataclass(frozen=True)
class Recommended:
    """
    Starting-point analysis settings for a product.

    Parameters
    ----------
    dv_kms : float or None
        CCF velocity step (km/s), as passed to ``calc_rv_against_template``.
    window_A : tuple of float or None
        (min_wv, max_wv) wavelength window (Angstrom) for single-order
        products; None for multi-order ones.
    preprocess : dict
        Keyword arguments for ``Spectrum.SpecPreProccess``.
    drop_orders : tuple of int
        Order indices to drop (``SelectOrders(..., remove=True)``).
    """
    dv_kms: Optional[float] = None
    window_A: Optional[tuple] = None
    preprocess: dict = field(default_factory=dict)
    drop_orders: tuple = ()


@dataclass(frozen=True)
class LoadedSpectrum:
    """
    One file (or simulated epoch) as returned by a `Product` loader.

    Parameters
    ----------
    wave : list of ndarray
        Per-order float64 wavelengths (Angstrom), ascending within an order.
    flux : list of ndarray
        Per-order float64 flux, matching ``wave``.
    time : float
        Observation time (days) on ``time_system``.
    time_system : str
        What ``time`` is, e.g. 'BJD-TDB', 'MJD-UTC'.
    bcv : float
        Barycentric velocity correction (km/s) still to be applied to ``wave``.
    specsys : str or None
        Frame of ``wave`` before ``bcv`` is applied (FITS SPECSYS:
        'BARYCENT', 'HELIOCEN', 'TOPOCENT', 'SOURCE'), None when unknown.
    wavelength_frame : str or None
        'air', 'vacuum', or None when not documented.
    snr : float
        Signal-to-noise ratio from the file; NaN when unknown.
    air_mass : float
        Air mass; NaN when unknown.
    vrad : float or None
        Radial velocity (km/s) recorded in the file, if any.
    native_step_kms : list of float or None
        Per-order native (detector) sampling (km/s per pixel) when ``wave`` is
        a resampled grid -- for an archive product, the instrument's
        ``native_step_kms`` (see `Product`); None when ``wave`` is the native
        grid, or the native sampling is unknown.
    survey : str
        Registry key of the product that produced this record.
    path : str or None
        Source file; None for a record that is not read from a file.
    extras : dict
        Product-specific additions (e.g. APOGEE pixel mask bits).
    """
    wave: list
    flux: list
    time: float
    time_system: str
    bcv: float
    specsys: Optional[str]
    wavelength_frame: Optional[str]
    snr: float = math.nan
    air_mass: float = math.nan
    vrad: Optional[float] = None
    native_step_kms: Optional[list] = None
    survey: str = ""
    path: Optional[str] = None
    extras: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Product:
    """
    One archive data product of an instrument, and its loader.

    Parameters
    ----------
    survey : str
        Registry key, e.g. 'HARPS', 'NRES-BANZAI'.
    instrument : Instrument
    description : str
        What the files are (pipeline, file type, layout).
    resampled : bool
        True if the archive grid is not the detector pixel grid. Then, if the
        instrument declares ``native_step_kms``, ``load`` reports it per order
        in ``LoadedSpectrum.native_step_kms`` (unless the loader already set
        it), so the error estimates count native pixels, not archive ones.
    grid : str
        The wavelength grid of the product, in words.
    wavelength_frame : str or None
        'air', 'vacuum', or None when not documented.
    specsys : str or None
        Documented SPECSYS of the product; loaders that read SPECSYS from the
        file report what the file says.
    time_system : str
        What the loaded ``time`` is.
    multi_order : bool
        True if a load returns several echelle orders.
    recommended : Recommended
    verified : bool
        True if the loader has been checked against real archive files. An
        unverified product's ``load`` warns (UserWarning) on every call.
    load : callable
        ``load(path, *, min_wv, max_wv, **options) -> LoadedSpectrum``;
        ``min_wv``/``max_wv`` (Angstrom) select the window of single-order
        products and are ignored by multi-order ones.
    """
    survey: str
    instrument: Instrument
    description: str
    resampled: bool
    grid: str
    wavelength_frame: Optional[str]
    specsys: Optional[str]
    time_system: str
    multi_order: bool
    recommended: Recommended
    verified: bool
    load: Callable[..., LoadedSpectrum]

    def __post_init__(self):
        if self.wavelength_frame not in WAVELENGTH_FRAMES:
            raise ValueError(f"{self.survey}: wavelength_frame {self.wavelength_frame!r} "
                             f"not in {WAVELENGTH_FRAMES}")
        if self.specsys not in SPECSYS_VALUES:
            raise ValueError(f"{self.survey}: specsys {self.specsys!r} not in {SPECSYS_VALUES}")
        if self.resampled and self.instrument.native_step_kms is not None:
            object.__setattr__(self, "load",
                               _seed_native_step(self.instrument.native_step_kms, self.load))
        if not self.verified:
            object.__setattr__(self, "load", _warn_unverified(self.survey, self.load))


def _seed_native_step(native_step_kms, load):
    @functools.wraps(load)
    def wrapped(*args, **kwargs):
        record = load(*args, **kwargs)
        if record.native_step_kms is not None:
            return record
        return dataclasses.replace(record, native_step_kms=[native_step_kms] * len(record.wave))
    return wrapped


def _warn_unverified(survey, load):
    @functools.wraps(load)
    def wrapped(*args, **kwargs):
        warnings.warn(f"loader not verified against archive files: {survey}",
                      UserWarning, stacklevel=2)
        return load(*args, **kwargs)
    return wrapped

