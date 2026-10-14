"""FEROS loaders (ESO Phase-3 SDP and CERES cubes) on synthetic FITS."""
import numpy as np
import pytest

from fits_builders import SDP_WAVE, check_ceres, check_single_order, load, sdp


@pytest.mark.parametrize("specsys, bcv", [("BARYCENT", 0.0), ("TOPOCENT", -7.5)])
def test_feros_applies_barycorr_only_when_topocentric(tmp_path, specsys, bcv):
    w = SDP_WAVE
    cols = {"WAVE": w, "FLUX": np.ones_like(w), "ERR": np.full_like(w, np.nan)}
    hdr = {"SPECSYS": specsys, "HIERARCH ESO DRS BARYCORR": -7.5,
           "MJD-OBS": 58000.0, "MJD-END": 58000.02, "SNR": 30.0}
    spec = load("FEROS", sdp(tmp_path, cols, hdr), min_wv=4900.0, max_wv=5150.0)
    check_single_order(spec, 4900.0, 5150.0)
    assert spec.bcv == bcv
    assert spec.time == pytest.approx(58000.01)
    assert (spec.time_system, spec.wavelength_frame, spec.snr) == ("MJD-UTC", "air", 30.0)


def test_eso_sdp_without_specsys_is_refused(tmp_path):
    w = SDP_WAVE
    cols = {"WAVE": w, "FLUX": np.ones_like(w), "ERR": np.ones_like(w)}
    hdr = {"HIERARCH ESO DRS BARYCORR": -7.5, "MJD-OBS": 58000.0, "MJD-END": 58000.02, "SNR": 30.0}
    with pytest.raises(ValueError, match="SPECSYS"):
        load("FEROS", sdp(tmp_path, cols, hdr), min_wv=4900.0, max_wv=5150.0)


def test_feros_ceres_picks_order_covering_window(tmp_path):
    check_ceres("FEROS-CERES", None, tmp_path)
