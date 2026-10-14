"""Tests for the specutils-based per-instrument ReadSpec loaders (S3).

These read the bundled example FITS (HARPS TOI193, NRES TOI677) and are skipped
if that data is absent. Bit-for-bit agreement with the legacy branches is
covered separately; here we pin the loader contract.
"""
import pathlib

import numpy as np
import pytest

from sparta.Auxil.ReadSpec import ReadSpec

ROOT = pathlib.Path(__file__).resolve().parent.parent
HARPS = sorted((ROOT / "examples" / "data" / "TOI193").glob("*.fits"))
NRES = sorted((ROOT / "examples" / "data" / "TOI677").glob("*.fits"))


@pytest.mark.skipif(not HARPS, reason="bundled HARPS data not present")
def test_harps_loader():
    rs = ReadSpec("HARPS")
    rs.load_spectrum_from_fits(path=str(HARPS[0]), min_wv=4990, max_wv=5090)
    assert rs.n_orders == 1
    assert rs.w[0].shape == rs.s[0].shape and rs.w[0].size > 0
    assert np.isfinite(rs.s[0]).all()
    assert 4990 <= rs.w[0].min() and rs.w[0].max() <= 5091   # trimmed to window
    # ESO SDP s1d: barycentric (SPECSYS) air (TUCD1) wavelengths -> no BCV.
    assert (rs.specsys, rs.wavelength_frame, rs.bcv) == ("BARYCENT", "air", 0.0)
    assert isinstance(rs.DATE_OBS, float)


@pytest.mark.skipif(not HARPS, reason="bundled HARPS data not present")
def test_harps_window_off_range_raises_valueerror():
    with pytest.raises(ValueError, match="does not overlap"):
        ReadSpec("HARPS").load_spectrum_from_fits(path=str(HARPS[0]), min_wv=8000, max_wv=8100)


@pytest.mark.skipif(not NRES, reason="bundled NRES data not present")
def test_nres_loader():
    rs = ReadSpec("NRES")
    rs.load_spectrum_from_fits(path=str(NRES[0]))
    assert rs.n_orders == 67 and len(rs.w) == 67 and len(rs.s) == 67
    assert np.isfinite(rs.s[0]).all()
    assert rs.w[0].min() > 3000   # nm -> Angstrom conversion applied
    assert (rs.time_system, rs.specsys) == ("BJD-TDB", "TOPOCENT")
