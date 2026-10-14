"""Real-file tests for the verified loaders (sparta.instruments) on the bundled data.

These read the example FITS (HARPS TOI193, NRES TOI677) and are skipped if that
data is absent. Bit-for-bit agreement with the legacy reader is pinned in
test_golden_instruments.py; here we pin the loader contract.
"""
import pathlib

import numpy as np
import pytest

from sparta import instruments
from sparta.Observations import Observations

ROOT = pathlib.Path(__file__).resolve().parent.parent
HARPS = sorted((ROOT / "examples" / "data" / "TOI193").glob("*.fits"))
NRES = sorted((ROOT / "examples" / "data" / "TOI677").glob("*.fits"))


@pytest.mark.skipif(not HARPS, reason="bundled HARPS data not present")
def test_harps_loader():
    spec = instruments.get("HARPS").load(HARPS[0], min_wv=4990, max_wv=5090)
    assert len(spec.wave) == len(spec.flux) == 1
    w, s = spec.wave[0], spec.flux[0]
    assert w.shape == s.shape and w.size > 0
    assert np.isfinite(s).all()
    assert 4990 <= w.min() and w.max() <= 5091   # trimmed to window
    # ESO SDP s1d: barycentric (SPECSYS) air (TUCD1) wavelengths -> no BCV.
    assert (spec.specsys, spec.wavelength_frame, spec.bcv) == ("BARYCENT", "air", 0.0)
    assert isinstance(spec.time, float)
    # The 0.01 A archive grid is resampled; the detector step (ESO: R 120,000
    # at 4.1 px per FWHM) is what sets the noise.
    assert spec.native_step_kms == [pytest.approx(299792.458 / (120000.0 * 4.1))]
    assert spec.native_step_kms[0] == pytest.approx(0.6093, abs=1e-4)


@pytest.mark.skipif(not HARPS, reason="bundled HARPS data not present")
def test_harps_window_off_range_raises_valueerror():
    with pytest.raises(ValueError, match="does not overlap"):
        instruments.get("HARPS").load(HARPS[0], min_wv=8000, max_wv=8100)


@pytest.mark.skipif(not NRES, reason="bundled NRES data not present")
def test_nres_loader():
    spec = instruments.get("NRES").load(NRES[0], min_wv=4900, max_wv=5150)
    assert len(spec.wave) == len(spec.flux) == 67
    assert np.isfinite(spec.flux[0]).all()
    assert spec.wave[0].min() > 3000   # nm -> Angstrom conversion applied
    assert (spec.time_system, spec.specsys) == ("BJD-TDB", "TOPOCENT")
    assert np.isnan(spec.snr) and np.isnan(spec.air_mass) and spec.vrad is None
    assert spec.native_step_kms is None       # native echelle pixels


@pytest.mark.skipif(not (HARPS and NRES), reason="bundled example data not present")
def test_observations_seed_the_native_step_of_resampled_products_only():
    harps = Observations(survey="HARPS", target_visits_lib=str(HARPS[0].parent),
                         min_wv=4990, max_wv=5090, sample_rate=8)
    native = instruments.get("HARPS").instrument.native_step_kms
    assert harps.sample_size == 4
    for sp in harps.spec_list:
        assert sp.Info["NativeStep"] == [native]      # the archive grid is finer here
    nres = Observations(survey="NRES", target_visits_lib=str(NRES[0].parent), sample_rate=6)
    assert all("NativeStep" not in sp.Info for sp in nres.spec_list)
