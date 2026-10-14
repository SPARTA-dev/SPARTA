"""PEPSI (LBT): stellar-rest-frame 1-D spectra as distributed by the NASA Exoplanet Archive."""
import numpy as np
from astropy.io import fits

from sparta.instruments._common import _wavelength_window
from sparta.instruments.base import Instrument, LoadedSpectrum, Product, Recommended

PEPSI = Instrument(
    name="PEPSI",
    resolving_power=None,
    native_step_kms=None,
    references=("https://exoplanetarchive.ipac.caltech.edu/docs/PEPSIMission.html",),
    notes="Resolving power depends on the fiber mode and is not recorded here.",
)


def load_table(path, *, min_wv, max_wv):
    '''
    Read a PEPSI binary table: wavelength (column 0) and normalized flux
    (column 1).

    Per the NASA Exoplanet Archive PEPSI notes
    (https://exoplanetarchive.ipac.caltech.edu/docs/PEPSIMission.html) the
    wavelengths are in air and already shifted to the stellar rest frame
    using the barycentric (OBSVEL) and catalogue stellar (RADVEL) velocities
    -- so RVs measured from them are relative to RADVEL. UNVERIFIED: the
    wavelength unit (Angstrom assumed; the Exoplanet Archive notes say
    microns) and the time scale of JD-OBS (mid-exposure).

    Parameters
    ----------
    path : str or path-like
        PEPSI FITS file.
    min_wv, max_wv : float
        Window (Angstrom); pixels with ``min_wv < w <= max_wv`` are kept.

    Returns
    -------
    LoadedSpectrum
        One order in the stellar rest frame (``bcv`` = 0 km/s); ``time`` is
        JD-OBS (days).
    '''
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        table = hdul[1].data
        w = np.array(table.field(0), dtype=float)
        s = np.array(table.field(1), dtype=float)
    window = _wavelength_window(w, min_wv, max_wv)
    return LoadedSpectrum(
        wave=[w[window]], flux=[s[window]],
        time=float(header["JD-OBS"]), time_system='JD-UTC',
        bcv=0.0, specsys='SOURCE', wavelength_frame='air',
        survey="PEPSI", path=str(path))


PRODUCTS = (
    Product(
        survey="PEPSI",
        instrument=PEPSI,
        description="NASA Exoplanet Archive PEPSI spectrum: HDU1 table with wavelength "
                    "(column 0) and normalized flux (column 1), in the stellar rest frame.",
        resampled=True,
        grid="as tabulated",
        wavelength_frame="air",
        specsys="SOURCE",
        time_system="JD-UTC",
        multi_order=False,
        recommended=Recommended(),
        verified=False,
        load=load_table,
    ),
)
