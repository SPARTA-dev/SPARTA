"""NRES (Las Cumbres Observatory network echelle spectrographs)."""
import numpy as np
from astropy.io import fits

from sparta.instruments.base import Instrument, LoadedSpectrum, Product, Recommended

NRES = Instrument(
    name="NRES",
    resolving_power=53000.0,
    native_step_kms=1.45,
    references=(
        "https://lco.global/observatory/instruments/nres/",
        "https://banzai-nres.readthedocs.io",
    ),
    notes="LCO quotes R ~ 53,000 in its summary and a 'typical spectral resolution of "
          "48,000' (4.47 pixels per fiber width) in the optical-design section. "
          "native_step_kms is measured, not documented: the median c * dlambda / lambda "
          "of the bundled TOI677 e91 files (1.44-1.52 km/s over the 67 orders). "
          "Recommended settings are the project's empirical defaults (examples/02_radial_velocity_unicor.ipynb), not "
          "instrument documentation.",
)


def load_e91(path, *, min_wv, max_wv):
    '''
    Read an NRES commissioning-pipeline spectrum (RLEVEL 91, '*-e91.fits'):
    one image row per echelle order (67), flux in HDU2 and wavelength (nm)
    in HDU6.

    Wavelengths are topocentric; the BCV is recovered from the
    barycentric-corrected (RVCC, km/s) and raw (RCC, z) cross-correlation RVs.

    Parameters
    ----------
    path : str or path-like
        e91 FITS file.
    min_wv, max_wv : float
        Ignored (all orders are returned).

    Returns
    -------
    LoadedSpectrum
        One entry per order, wavelengths in Angstrom; ``time`` is the BJD_TDB of the
        exposure centre (days), ``bcv`` in km/s.
    '''
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        flux = np.asarray(hdul[2].data, float)
        wave = 10.0 * np.asarray(hdul[6].data, float)       # nm -> Angstrom
        if flux.shape != wave.shape:
            raise ValueError(f"NRES flux {flux.shape} and wavelength {wave.shape} "
                             "images differ in shape")
        return LoadedSpectrum(
            wave=list(wave), flux=list(flux),
            time=float(header['BJD']),        # header: "[BJD_TDB] Exposure center"
            time_system="BJD-TDB",
            bcv=float(header['RVCC'] - header['RCC'] * 299792.458),
            specsys="TOPOCENT",
            # UNVERIFIED: air or vacuum for the commissioning (IDL) pipeline;
            # BANZAI-NRES documents vacuum, the older products do not say.
            wavelength_frame=None,
            survey="NRES", path=str(path))


def load_banzai(path, *, min_wv, max_wv):
    '''
    Read a BANZAI-NRES 1-D product (https://banzai-nres.readthedocs.io):
    per-order 'wavelength' (Angstrom, vacuum) and 'normflux' columns of the
    science fiber (SCIFIBER) in HDU1.

    Parameters
    ----------
    path : str or path-like
        BANZAI-NRES FITS file.
    min_wv, max_wv : float
        Ignored (all science-fiber orders are returned).

    Returns
    -------
    LoadedSpectrum
        One entry per science-fiber row; ``time`` is TCORR, the mid-exposure
        BJD_TDB (days); ``bcv`` is BARYCORR converted from m/s to km/s.
    '''
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        rows = hdul[1].data[hdul[1].data['fiber'] == header['SCIFIBER']]
        return LoadedSpectrum(
            wave=list(np.array(rows['wavelength'], dtype=float)),
            flux=list(np.array(rows['normflux'], dtype=float)),
            time=float(header["TCORR"]), time_system="BJD-TDB",
            bcv=float(header['BARYCORR']) / 1000,      # m/s -> km/s
            specsys="TOPOCENT", wavelength_frame="vacuum",
            survey="NRES-BANZAI", path=str(path))


PRODUCTS = (
    Product(
        survey="NRES",
        instrument=NRES,
        description="NRES commissioning (IDL) pipeline extracted spectrum, RLEVEL 91 "
                    "('*-e91.fits'): HDU2 SPECFLAT flux and HDU6 WAVESPEC wavelengths (nm), "
                    "67 orders x 4096 pixels.",
        resampled=False,
        grid="native echelle pixels",
        wavelength_frame=None,
        specsys="TOPOCENT",
        time_system="BJD-TDB",
        multi_order=True,
        recommended=Recommended(
            dv_kms=0.05,
            preprocess=dict(n_trim=100, clean_margins=True, remove_nans=True, delta=0.5,
                            cosmic_sigma=3, filter_lowcut=4, filter_highcut=0.15, alpha=0.3),
            drop_orders=tuple(range(10)) + tuple(range(50, 67))),
        verified=True,
        load=load_e91,
    ),
    Product(
        survey="NRES-BANZAI",
        instrument=NRES,
        description="BANZAI-NRES 1-D product: HDU1 table, one row per (fiber, order) with "
                    "'wavelength' (Angstrom) and 'normflux' columns; the SCIFIBER rows are read.",
        resampled=False,
        grid="native echelle pixels",
        wavelength_frame="vacuum",
        specsys="TOPOCENT",
        time_system="BJD-TDB",
        multi_order=True,
        recommended=Recommended(),
        verified=False,
        load=load_banzai,
    ),
)
