"""UVES (ESO VLT): ESO Phase-3 1-D echelle spectra."""
import numpy as np
from astropy.io import fits

from sparta.instruments._eso import _bcv_for_specsys, _eso_sdp_frames
from sparta.instruments.base import Instrument, LoadedSpectrum, Product, Recommended

UVES = Instrument(
    name="UVES",
    resolving_power=None,
    native_step_kms=None,
    references=("https://www.eso.org/sci/facilities/paranal/instruments/uves.html",),
    notes="Resolving power depends on the slit width and is not recorded here.",
)

# A wavelength step this many times the median step of the table marks the
# CCD-chip gap of the merged spectrum. The two red-arm chips are sampled at
# 0.0285 and 0.0341 A/pixel (ratio 1.2); the gap is a ~70 A step.
_GAP_STEP_FACTOR = 1.5


def _mid_exposure_mjd(primary, spectrum):
    '''
    Mid-exposure time of an SDP spectrum.

    Parameters
    ----------
    primary, spectrum : astropy.io.fits.Header
        Primary header and SPECTRUM-extension header.

    Returns
    -------
    float
        TMID (SPECTRUM extension) when present, else the midpoint of MJD-OBS
        and MJD-END (primary header); MJD in days. The files carry no TIMESYS,
        so the FITS default, UTC, applies.

    Raises
    ------
    ValueError
        If neither TMID nor both of MJD-OBS and MJD-END are present.
    '''
    if 'TMID' in spectrum:
        return float(spectrum['TMID'])
    if 'MJD-OBS' in primary and 'MJD-END' in primary:
        return 0.5 * (float(primary['MJD-OBS']) + float(primary['MJD-END']))
    raise ValueError("UVES SDP file has neither TMID nor MJD-OBS and MJD-END; "
                     "cannot tell the mid-exposure time")


def _chips(w, valid):
    '''
    Pixel ranges of the CCD chips in the merged spectrum, without the no-data
    pixels at their ends.

    Parameters
    ----------
    w : ndarray
        Strictly ascending wavelengths of the full table (Angstrom).
    valid : ndarray of bool
        Pixels that carry data.

    Returns
    -------
    list of (int, int)
        ``(start, stop)`` index ranges, in wavelength order; chips with no
        valid pixel are left out.
    '''
    step = np.diff(w)
    edges = np.flatnonzero(step > _GAP_STEP_FACTOR * np.median(step)) + 1
    chips = []
    for start, stop in zip(np.r_[0, edges], np.r_[edges, w.size]):
        good = np.flatnonzero(valid[start:stop])
        if good.size:
            chips.append((start + good[0], start + good[-1] + 1))
    return chips


def load_sdp(path, *, min_wv, max_wv):
    '''
    Read a UVES ESO Phase-3 SDP 1-D spectrum, trimmed to a window.

    Per the UVES echelle release description
    (https://www.eso.org/sci/observing/phase3/data_releases/IDP_UVES_ECH_release_description_v1.3.pdf)
    no barycentric correction is applied to the wavelengths; it is stored in
    HIERARCH ESO QC VRAD BARYCOR (km/s). On a 2026 red-arm file that value
    differs from astropy's correction at mid-exposure by -46 m/s, and matches
    it at no time within the exposure, so it is not good to the m/s level.
    Uses the flux-calibrated FLUX column (with ERR) when present, else
    FLUX_REDUCED (with ERR_REDUCED); the columns are never mixed.

    The merged spectrum spans two CCD chips with a gap between them. The
    window must lie within one chip. Pixels without data -- flux exactly 0
    (the pipeline's end padding, and where the flux calibration does not
    reach), error <= 0, or non-finite -- are trimmed from the chip ends; a
    window with such pixels inside a chip is refused, so the returned order is
    always one uniform linear grid.

    Parameters
    ----------
    path : str or path-like
        ADP FITS file of product category SCIENCE.SPECTRUM.
    min_wv, max_wv : float
        Window (Angstrom); pixels with ``min_wv < w <= max_wv`` are kept.

    Returns
    -------
    LoadedSpectrum
        One order; ``time`` is the mid-exposure MJD (TMID, UTC, days); ``bcv``
        in km/s; ``extras['err']`` holds the flux error of the order (a
        one-element list, flux units) and ``extras['flux_column']`` the column
        read.

    Raises
    ------
    ValueError
        If the file is not a UVES SCIENCE.SPECTRUM product, WAVE is not finite
        and strictly ascending, SPECSYS or the mid-exposure time is missing,
        or the window holds no valid pixel, spans the chip gap, or has pixels
        without data inside a chip.
    '''
    if not min_wv < max_wv:
        raise ValueError(f"empty wavelength window: min_wv={min_wv} >= max_wv={max_wv} (Angstrom)")
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        product_category = header.get('PRODCATG')
        if product_category != 'SCIENCE.SPECTRUM':
            raise ValueError(f"{path}: PRODCATG is {product_category!r}; the UVES loader reads "
                             "only 1-D SCIENCE.SPECTRUM products, not the ancillary 2-D images")
        if header.get('INSTRUME', '').strip() != 'UVES':
            raise ValueError(f"{path}: INSTRUME is {header.get('INSTRUME')!r}, not 'UVES'")
        table = hdul[1].data
        flux_column, err_column = (('FLUX', 'ERR') if 'FLUX' in table.columns.names
                                   else ('FLUX_REDUCED', 'ERR_REDUCED'))
        w = np.array(table['WAVE'][0], dtype=float)
        s = np.array(table[flux_column][0], dtype=float)
        e = np.array(table[err_column][0], dtype=float)
        specsys, wavelength_frame = _eso_sdp_frames(hdul)
        time = _mid_exposure_mjd(header, hdul[1].header)
    if not (np.all(np.isfinite(w)) and np.all(np.diff(w) > 0)):
        raise ValueError(f"{path}: WAVE is not finite and strictly ascending")
    bcv = _bcv_for_specsys(header, specsys, 'HIERARCH ESO QC VRAD BARYCOR')

    valid = np.isfinite(s) & np.isfinite(e) & (s != 0) & (e > 0)
    chips = _chips(w, valid)
    coverage = ", ".join(f"{w[a]:.3f}-{w[b - 1]:.3f}" for a, b in chips) or "none"
    in_window = (w > min_wv) & (w <= max_wv) & valid
    hit = [(a, b) for a, b in chips if in_window[a:b].any()]
    if not hit:
        raise ValueError(f"wavelength window ({min_wv}, {max_wv}] Angstrom holds no valid pixel "
                         f"of the spectrum (chips with data: {coverage} Angstrom)")
    if len(hit) > 1:
        raise ValueError(f"wavelength window ({min_wv}, {max_wv}] Angstrom spans the UVES "
                         f"CCD-chip gap; choose a window within one chip ({coverage} Angstrom)")
    a, b = hit[0]
    keep = np.flatnonzero(in_window[a:b]) + a
    # Later resampling would interpolate across a hole as if it were data.
    holes = np.flatnonzero(np.diff(keep) > 1)
    if holes.size:
        raise ValueError(f"wavelength window ({min_wv}, {max_wv}] Angstrom contains pixels "
                         f"without data inside a chip, first after {w[keep[holes[0]]]:.3f} "
                         f"Angstrom; choose a window that avoids them")
    return LoadedSpectrum(
        wave=[w[keep]], flux=[s[keep]],
        time=time, time_system='MJD-UTC',
        bcv=bcv, specsys=specsys, wavelength_frame=wavelength_frame,
        snr=float(header["SNR"]), survey="UVES", path=str(path),
        extras={'err': [e[keep]], 'flux_column': flux_column})


PRODUCTS = (
    Product(
        survey="UVES",
        instrument=UVES,
        description="ESO Phase-3 SDP 1-D echelle spectrum (ADP.*.fits, PRODCATG "
                    "SCIENCE.SPECTRUM): one-row binary table in HDU1 with WAVE and FLUX "
                    "(or FLUX_REDUCED) columns. The window must lie within one CCD chip; "
                    "no-data pixels (zero flux, non-positive error) are trimmed from the "
                    "chip ends, and a window with such pixels inside a chip is refused. Errors "
                    "are in extras['err']. The ancillary 2-D images "
                    "(ANCILLARY.2DSPECTRUM) are not read.",
        resampled=True,
        grid="1-D merged spectrum as delivered by the ESO pipeline, linear in "
             "wavelength per CCD chip",
        wavelength_frame="air",
        specsys="TOPOCENT",
        time_system="MJD-UTC",
        multi_order=False,
        recommended=Recommended(),
        verified=False,
        load=load_sdp,
    ),
)
