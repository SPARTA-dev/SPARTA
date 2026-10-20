"""UVES ESO Phase-3 SDP loader on synthetic FITS.

The synthetic files mimic what a real UVES red-arm SDP file shows: zero-flux
padding at the ends of the merged spectrum, a CCD-chip gap with a step change
across it, and a stretch at a chip end where the flux-calibrated FLUX is 0
(ERR = 0) while FLUX_REDUCED still has data.
"""
import numpy as np
import pytest
from astropy.io import fits

from fits_builders import SDP_WAVE, check_single_order, load, sdp, write

HEADER = {"PRODCATG": "SCIENCE.SPECTRUM", "INSTRUME": "UVES", "SPECSYS": "TOPOCENT",
          "HIERARCH ESO QC VRAD BARYCOR": 12.3, "MJD-OBS": 58000.25, "MJD-END": 58000.27,
          "SNR": 40.0}
TMID = 58000.2601


def columns(w, flux=2.0, err=0.5, flux_reduced=3.0, err_reduced=1.0):
    def full(v):
        return np.broadcast_to(np.asarray(v, float), w.shape).copy()
    return {"WAVE": w, "FLUX_REDUCED": full(flux_reduced), "ERR_REDUCED": full(err_reduced),
            "BGFLUX_REDUCED": np.zeros_like(w), "FLUX": full(flux), "ERR": full(err)}


def uves_sdp(tmp_path, cols, header=HEADER, tmid=TMID, name="sdp.fits", **kw):
    path = sdp(tmp_path, cols, header, name=name, **kw)
    if tmid is not None:
        with fits.open(path, mode="update") as hdul:
            hdul[1].header["TMID"] = tmid
    return path


def two_chip_wave():
    """Lower chip 4800-5000 A at 0.02 A, upper chip 5050-5300 A at 0.025 A: a 50 A gap."""
    return np.concatenate([np.arange(4800.0, 5000.0 + 1e-9, 0.02),
                           np.arange(5050.0, 5300.0 + 1e-9, 0.025)])


def test_uves_topocentric_applies_qc_barycor(tmp_path):
    spec = load("UVES", uves_sdp(tmp_path, columns(SDP_WAVE)), min_wv=4900.0, max_wv=5150.0)
    check_single_order(spec, 4900.0, 5150.0)
    assert np.all(spec.flux[0] == 2.0)             # flux-calibrated FLUX column
    assert (spec.bcv, spec.specsys, spec.wavelength_frame, spec.time_system) == \
        (12.3, "TOPOCENT", "air", "MJD-UTC")


def test_uves_reduced_product_without_flux_column(tmp_path):
    cols = columns(SDP_WAVE)
    del cols["FLUX"], cols["ERR"]
    hdr = {**HEADER, "SPECSYS": "BARYCENT"}
    spec = load("UVES", uves_sdp(tmp_path, cols, hdr, tucd="em.wl"), min_wv=4900.0, max_wv=5150.0)
    assert np.all(spec.flux[0] == 3.0)
    assert np.all(spec.extras["err"][0] == 1.0)    # ERR_REDUCED goes with FLUX_REDUCED
    assert (spec.bcv, spec.wavelength_frame) == (0.0, "vacuum")


def test_uves_time_is_mid_exposure_tmid(tmp_path):
    spec = load("UVES", uves_sdp(tmp_path, columns(SDP_WAVE)), min_wv=4900.0, max_wv=5150.0)
    assert spec.time == TMID and isinstance(spec.time, float)


def test_uves_time_falls_back_to_mid_of_mjd_obs_and_mjd_end(tmp_path):
    spec = load("UVES", uves_sdp(tmp_path, columns(SDP_WAVE), tmid=None),
                min_wv=4900.0, max_wv=5150.0)
    assert spec.time == pytest.approx(58000.26, abs=1e-9)


def test_uves_without_any_mid_exposure_time_raises(tmp_path):
    hdr = {k: v for k, v in HEADER.items() if k != "MJD-END"}
    with pytest.raises(ValueError, match="TMID"):
        load("UVES", uves_sdp(tmp_path, columns(SDP_WAVE), hdr, tmid=None),
             min_wv=4900.0, max_wv=5150.0)


def test_uves_errors_returned_in_extras_aligned_with_flux(tmp_path):
    w = SDP_WAVE
    err = 0.1 + 1e-4 * np.arange(w.size)           # a ramp catches any mis-slicing
    spec = load("UVES", uves_sdp(tmp_path, columns(w, err=err)), min_wv=4900.0, max_wv=5150.0)
    assert len(spec.extras["err"]) == len(spec.flux) == 1
    e = spec.extras["err"][0]
    assert e.shape == spec.flux[0].shape and e.dtype == np.float64
    np.testing.assert_array_equal(e, err[np.searchsorted(w, spec.wave[0])])
    assert spec.extras["flux_column"] == "FLUX"


def test_uves_zero_flux_padding_is_trimmed(tmp_path):
    w = SDP_WAVE
    flux = np.where((w < 4850.0) | (w > 5250.0), 0.0, 2.0)   # real files: FLUX = 0, ERR > 0
    spec = load("UVES", uves_sdp(tmp_path, columns(w, flux=flux)), min_wv=4800.0, max_wv=5300.0)
    check_single_order(spec, 4800.0, 5300.0)
    assert spec.wave[0][0] >= 4850.0 and spec.wave[0][-1] <= 5250.0
    assert np.all(spec.flux[0] == 2.0)


def test_uves_reduced_product_zero_padding_is_trimmed(tmp_path):
    w = SDP_WAVE
    cols = columns(w, flux_reduced=np.where(w < 4850.0, 0.0, 3.0))
    del cols["FLUX"], cols["ERR"]
    spec = load("UVES", uves_sdp(tmp_path, cols), min_wv=4800.0, max_wv=4900.0)
    assert spec.wave[0][0] >= 4850.0 and np.all(spec.flux[0] == 3.0)


def test_uves_non_positive_or_nan_pixels_at_the_end_are_trimmed(tmp_path):
    w = SDP_WAVE
    err = np.where(w > 5250.0, 0.0, 0.5)
    flux = np.where(w < 4810.0, np.nan, 2.0)
    spec = load("UVES", uves_sdp(tmp_path, columns(w, flux=flux, err=err)),
                min_wv=4800.0, max_wv=5300.0)
    assert spec.wave[0][0] >= 4810.0 and spec.wave[0][-1] <= 5250.0
    assert np.all(np.isfinite(spec.flux[0])) and np.all(spec.extras["err"][0] > 0)


def test_uves_flux_missing_at_chip_end_is_trimmed_not_filled(tmp_path):
    # Real red-arm files: FLUX = 0, ERR = 0 over the last ~8 A of the lower chip while
    # FLUX_REDUCED still has data. The loader trims it rather than mixing columns.
    w = two_chip_wave()
    hole = (w > 4990.0) & (w <= 5000.0 + 1e-6)
    spec = load("UVES", uves_sdp(tmp_path, columns(w, flux=np.where(hole, 0.0, 2.0),
                                                   err=np.where(hole, 0.0, 0.5))),
                min_wv=4950.0, max_wv=5000.0)
    assert spec.wave[0][-1] <= 4990.0 and np.all(spec.flux[0] == 2.0)


def test_uves_window_with_no_data_inside_a_chip_raises(tmp_path):
    w = SDP_WAVE
    err = np.full_like(w, 0.5)
    err[np.argmin(np.abs(w - 5000.0))] = 0.0       # one bad pixel inside the chip
    path = uves_sdp(tmp_path, columns(w, err=err))
    with pytest.raises(ValueError, match="without data inside a chip"):
        load("UVES", path, min_wv=4900.0, max_wv=5100.0)
    check_single_order(load("UVES", path, min_wv=5010.0, max_wv=5100.0), 5010.0, 5100.0)


def test_uves_window_spanning_chip_gap_raises(tmp_path):
    with pytest.raises(ValueError, match="spans the UVES CCD-chip gap"):
        load("UVES", uves_sdp(tmp_path, columns(two_chip_wave())), min_wv=4900.0, max_wv=5150.0)


def test_uves_window_inside_one_chip_returns_one_uniform_order(tmp_path):
    spec = load("UVES", uves_sdp(tmp_path, columns(two_chip_wave())),
                min_wv=5100.0, max_wv=5200.0)
    check_single_order(spec, 5100.0, 5200.0)
    assert np.ptp(np.diff(spec.wave[0])) < 1e-6


def test_uves_window_inside_the_gap_raises(tmp_path):
    with pytest.raises(ValueError, match="no valid pixel"):
        load("UVES", uves_sdp(tmp_path, columns(two_chip_wave())), min_wv=5010.0, max_wv=5040.0)


def test_uves_window_only_on_padding_raises(tmp_path):
    w = SDP_WAVE
    flux = np.where(w < 4850.0, 0.0, 2.0)
    with pytest.raises(ValueError, match="no valid pixel"):
        load("UVES", uves_sdp(tmp_path, columns(w, flux=flux)), min_wv=4800.0, max_wv=4840.0)


def test_uves_empty_window_raises(tmp_path):
    with pytest.raises(ValueError, match="empty wavelength window"):
        load("UVES", uves_sdp(tmp_path, columns(SDP_WAVE)), min_wv=5000.0, max_wv=5000.0)


@pytest.mark.parametrize("bad", ["nan", "descending"])
def test_uves_bad_wave_axis_raises(tmp_path, bad):
    w = SDP_WAVE.copy()
    if bad == "nan":
        w[100] = np.nan
    else:
        w[100], w[101] = w[101], w[100]
    with pytest.raises(ValueError, match="strictly ascending"):
        load("UVES", uves_sdp(tmp_path, columns(w)), min_wv=4900.0, max_wv=5150.0)


def test_uves_rejects_ancillary_2d_product(tmp_path):
    hdr = fits.Header({"PRODCATG": "ANCILLARY.2DSPECTRUM", "INSTRUME": "UVES",
                       "CTYPE1": "AWAV", "CRVAL1": 4800.0, "CDELT1": 0.02, "CRPIX1": 1.0})
    path = write(tmp_path, [fits.PrimaryHDU(np.ones((30, 100), np.float32), hdr)])
    with pytest.raises(ValueError, match="ANCILLARY.2DSPECTRUM"):
        load("UVES", path)


def test_uves_rejects_other_instruments_sdp(tmp_path):
    with pytest.raises(ValueError, match="FEROS"):
        load("UVES", uves_sdp(tmp_path, columns(SDP_WAVE), {**HEADER, "INSTRUME": "FEROS"}))
