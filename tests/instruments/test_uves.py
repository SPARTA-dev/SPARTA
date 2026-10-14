"""UVES ESO Phase-3 SDP loader on synthetic FITS."""
import numpy as np

from fits_builders import SDP_WAVE, check_single_order, load, sdp


def test_uves_topocentric_applies_qc_barycor(tmp_path):
    w = SDP_WAVE
    cols = {"WAVE": w, "FLUX_REDUCED": np.ones_like(w), "ERR_REDUCED": np.ones_like(w),
            "BGFLUX_REDUCED": np.zeros_like(w), "FLUX": 2 * np.ones_like(w),
            "ERR": np.ones_like(w)}
    hdr = {"SPECSYS": "TOPOCENT", "HIERARCH ESO QC VRAD BARYCOR": 12.3,
           "MJD-OBS": 58000.25, "SNR": 40.0}
    spec = load("UVES", sdp(tmp_path, cols, hdr), min_wv=4900.0, max_wv=5150.0)
    check_single_order(spec, 4900.0, 5150.0)
    assert np.all(spec.flux[0] == 2.0)             # flux-calibrated FLUX column
    assert (spec.bcv, spec.specsys, spec.wavelength_frame, spec.time_system, spec.time) == \
        (12.3, "TOPOCENT", "air", "MJD-UTC", 58000.25)


def test_uves_reduced_product_without_flux_column(tmp_path):
    w = SDP_WAVE
    cols = {"WAVE": w, "FLUX_REDUCED": 3 * np.ones_like(w), "ERR_REDUCED": np.ones_like(w),
            "BGFLUX_REDUCED": np.zeros_like(w)}
    hdr = {"SPECSYS": "BARYCENT", "MJD-OBS": 58000.25, "SNR": 40.0}
    spec = load("UVES", sdp(tmp_path, cols, hdr, tucd="em.wl"), min_wv=4900.0, max_wv=5150.0)
    assert np.all(spec.flux[0] == 3.0)
    assert (spec.bcv, spec.wavelength_frame) == (0.0, "vacuum")
