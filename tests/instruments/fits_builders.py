"""Shared helpers for the synthetic-FITS instrument tests.

Each test module writes minimal files following the instrument's documented
header conventions (data-model URLs are cited in the loader docstrings).
"""
import numpy as np
import pytest
from astropy.io import fits

from sparta import instruments


def write(tmp_path, hdus, name="spec.fits"):
    path = tmp_path / name
    fits.HDUList(hdus).writeto(path)
    return str(path)


def load(survey, path, *, min_wv=4900.0, max_wv=5150.0, **options):
    """Load through the registry, expecting the unverified-loader warning when due."""
    product = instruments.get(survey)
    if product.verified:
        return product.load(path, min_wv=min_wv, max_wv=max_wv, **options)
    with pytest.warns(UserWarning, match=f"loader not verified against archive files: {survey}"):
        return product.load(path, min_wv=min_wv, max_wv=max_wv, **options)


def check_single_order(spec, min_wv, max_wv):
    assert len(spec.wave) == len(spec.flux) == 1
    w, s = spec.wave[0], spec.flux[0]
    assert w.ndim == 1 and w.shape == s.shape and w.size > 0
    assert w.dtype == np.float64 and s.dtype == np.float64
    assert np.all(np.diff(w) > 0)
    assert min_wv < w[0] and w[-1] <= max_wv
    assert isinstance(spec.time, float)


def sdp_table(columns, n):
    """One-row ESO-SDP-style binary table of array columns."""
    return fits.BinTableHDU.from_columns(
        [fits.Column(name=name, format=f"{n}D", array=np.asarray(values, float)[None, :])
         for name, values in columns.items()], name="SPECTRUM")


def sdp(tmp_path, columns, header, tucd="em.wl;obs.atmos", name="sdp.fits"):
    """ESO Phase-3 SDP file: primary header + SPECTRUM table, TUCD1 on the WAVE column."""
    n = len(next(iter(columns.values())))
    table = sdp_table(columns, n)
    if tucd is not None:
        table.header["TUCD1"] = tucd
    return write(tmp_path, [fits.PrimaryHDU(header=fits.Header(header)), table], name)


SDP_WAVE = np.linspace(4800.0, 5300.0, 5001)


def ceres(tmp_path, name="ceres.fits"):
    """CERES cube with three orders (5000-5120, 5100-5220, 5200-5320 A); flux = 10 * order + ramp."""
    n = 2048
    cube = np.zeros((11, 3, n))
    for k, start in enumerate([5000.0, 5100.0, 5200.0]):
        cube[0, k] = np.linspace(start, start + 120.0, n)
        cube[3, k] = 10.0 * k + np.linspace(0.0, 1.0, n)
    hdr = fits.Header()
    hdr.update({"HIERARCH MBJD": 58000.5, "HIERARCH BARYCENTRIC CORRECTION (KM/S)": 3.2,
                "SNR": 80.0, "RV": 11.1})
    return write(tmp_path, [fits.PrimaryHDU(cube, hdr)], name)


def check_ceres(survey, frame, tmp_path):
    spec = load(survey, ceres(tmp_path), min_wv=5150.0, max_wv=5215.0)
    check_single_order(spec, 5150.0, 5215.0)
    assert np.all((spec.flux[0] >= 10.0) & (spec.flux[0] < 20.0))    # order 1 (5100-5220 A)
    assert (spec.time, spec.bcv, spec.vrad, spec.snr) == (58000.5, 0.0, 11.1, 80.0)
    assert (spec.time_system, spec.specsys, spec.wavelength_frame) == ("MBJD-UTC", "BARYCENT", frame)
    with pytest.raises(ValueError, match="no echelle order"):
        load(survey, ceres(tmp_path, "b.fits"), min_wv=6000.0, max_wv=6100.0)


def linear_image(tmp_path, extra, n=20001):
    """1-D image with a linear WCS starting at 4000 A, 0.1 A per pixel; flux = 0-based pixel."""
    hdr = fits.Header()
    hdr.update({"CRVAL1": 4000.0, "CDELT1": 0.1, "CRPIX1": 1.0, **extra})
    return write(tmp_path, [fits.PrimaryHDU(np.arange(n, dtype=np.float32), hdr)])
