"""Correctness tests for the reimplemented CCF / RV extraction.

The legacy CrossCorrelateSpec could not recover non-negative radial velocities
(see KNOWN_TRAPS.md): it truncated the template to a shifted sub-grid and used
a constant (not per-lag) normalization, so only RV <~ 0 was recoverable. The
rewrite over unicor.ccf_vs_velocity resamples template and signal onto a
common, equal-length log-lambda grid and scans symmetric velocity lags, so the
whole axis is valid. The golden master is physical: recovered RV must equal the
injected RV, for both signs.
"""
import numpy as np
import pytest

import unicor as so
from sparta.UNICOR.Spectrum import Spectrum
from sparta.UNICOR.Template import Template
from sparta.UNICOR.CCF1d import CCF1d

C = 299792.458
WAVE = np.linspace(5480.0, 5520.0, 4000)


def _line(center):
    return 1.0 - 0.6 * np.exp(-0.5 * ((WAVE - center) / 0.12) ** 2)


@pytest.fixture(scope="module")
def template():
    return Template(template=Spectrum(wv=[WAVE.copy()], sp=[_line(5500.0)], name="t"))


@pytest.mark.parametrize("inj", [-10.0, -5.0, 0.0, 5.0, 10.0])
def test_rv_recovers_injected(template, inj):
    obs = Spectrum(wv=[WAVE.copy()], sp=[_line(5500.0 * (1 + inj / C))], name="o")
    ccf = CCF1d().CrossCorrelateSpec(spec_in=obs, template_in=template,
                                     dv=0.5, VelBound=50).CombineCCFs()
    rv = ccf.CorrCombined["RV"]
    rv = float(rv.value if hasattr(rv, "value") else rv)
    assert rv == pytest.approx(inj, abs=0.05)
    assert float(ccf.CorrCombined["peakCorr"]) > 0.99


@pytest.mark.parametrize("inj", [-8.0, 0.0, 8.0])
def test_ccf_vs_velocity_recovers_shift(inj):
    dv, n = 0.5, 2000
    u = np.arange(n) * dv

    def line(cv):
        return 1.0 - 0.6 * np.exp(-0.5 * ((u - n // 2 * dv - cv) / 3.0) ** 2)

    v, ccf = so.ccf_vs_velocity(line(0.0), line(inj), dv, vmax=40)
    assert v[np.argmax(ccf)] == pytest.approx(inj, abs=dv)
    assert ccf.max() > 0.999   # ~1; shifted signals lose a little edge overlap


def test_ccf_vs_velocity_rejects_mismatched_grids():
    with pytest.raises(ValueError):
        so.ccf_vs_velocity(np.ones(10), np.ones(12), 0.5, 5)


def test_rv_beyond_the_scan_is_nan_not_a_wrong_sign(template):
    # -70 km/s with a +-50 km/s scan: the legacy parabola window wrapped around
    # the array and reported about +2.8 km/s, silently.
    obs = Spectrum(wv=[WAVE.copy()], sp=[_line(5500.0 * (1 - 70.0 / C))], name="o")
    ccf = CCF1d().CrossCorrelateSpec(spec_in=obs, template_in=template,
                                     dv=0.5, VelBound=50).CombineCCFs()
    assert np.isnan(float(ccf.Corr["RV"][0, 0]))
    assert np.isnan(float(np.asarray(ccf.CorrCombined["RV"])))


def test_unitless_velbound_is_km_per_s_whatever_the_unit_of_dv(template):
    from astropy import units as u
    obs = Spectrum(wv=[WAVE.copy()], sp=[_line(5500.0 * (1 + 5.0 / C))], name="o")
    ccf = CCF1d().CrossCorrelateSpec(spec_in=obs, template_in=template,
                                     dv=500 * u.m / u.s, VelBound=50).CombineCCFs()
    v = ccf.Corr["vel"].to_value(u.km / u.s)
    assert v.max() == pytest.approx(50.0, abs=0.5)
    assert float(ccf.CorrCombined["RV"]) == pytest.approx(5.0, abs=0.05)


def test_a_grid_that_is_not_velocity_even_is_refused(template):
    obs = Spectrum(wv=[WAVE.copy()], sp=[_line(5500.0)], name="o")
    obs.InterpolateSpectrum(delta=0.5, InterpMethod="log")
    # Dropping pixels leaves GridType == 'log' while breaking the uniform step.
    obs.wv[0] = np.delete(obs.wv[0], np.arange(100, 110))
    obs.sp[0] = np.delete(obs.sp[0], np.arange(100, 110))
    with pytest.raises(ValueError, match="uniform"):
        CCF1d().CrossCorrelateSpec(spec_in=obs, template_in=template, VelBound=50)


def test_an_order_the_template_does_not_cover_is_nan_not_fatal():
    # Order 0 lies outside the template's span, so the resampled template is a
    # constant there; that order is NaN and order 1 still yields its RV.
    tmpl = Template(template=Spectrum(wv=[WAVE.copy()], sp=[_line(5500.0)], name="t"))
    far = np.linspace(6000.0, 6040.0, 4000)
    obs = Spectrum(wv=[far, WAVE.copy()],
                   sp=[np.ones_like(far), _line(5500.0 * (1 + 4.0 / C))], name="o")
    with pytest.warns(UserWarning, match="order 0"):
        ccf = CCF1d().CrossCorrelateSpec(spec_in=obs, template_in=tmpl, dv=0.5, VelBound=50)
    assert np.isnan(ccf.Corr["RV"][0, 0])
    assert float(ccf.Corr["RV"][1, 0]) == pytest.approx(4.0, abs=0.05)


# --- ML uncertainty calibration (Zucker 2003, MNRAS 342, 1291, §2.3) --------
# sigma^2 = -[N C'' C / (1 - C^2)]^-1 with N the number of independent spectral
# bins. Resampling onto a finer log grid adds correlated samples, not bins, so N
# must be counted on the native grid -- and never from the number of velocity
# lags, which made eRV scale with the scan half-width.

NATIVE_DV = 1.0     # km/s, the "detector" sampling the noise is drawn on
MC_WAVE, _ = so.log_lambda_grid(5480.0, 5520.0, NATIVE_DV)
_LINES = np.random.default_rng(1)
MC_CENTERS = _LINES.uniform(5483.0, 5517.0, 30)
MC_DEPTHS = _LINES.uniform(0.2, 0.7, 30)


def _forest(wave, v_kms):
    """Thirty Gaussian absorption lines (sigma 0.08 A), shifted by ``v_kms``."""
    centers = MC_CENTERS * (1.0 + v_kms / C)
    return 1.0 - (MC_DEPTHS * np.exp(
        -0.5 * ((wave[:, None] - centers) / 0.08) ** 2)).sum(axis=1)


@pytest.fixture(scope="module")
def forest_template():
    twave, _ = so.log_lambda_grid(5470.0, 5530.0, 0.25)
    return Template(template=Spectrum(wv=[twave], sp=[_forest(twave, 0.0)], name="t"))


def _noisy_rvs(template, n_real, dv, vel_bound, seed=42):
    rng = np.random.default_rng(seed)
    clean = _forest(MC_WAVE, 3.0)
    rv, erv = np.empty(n_real), np.empty(n_real)
    for k in range(n_real):
        # SNR 50 white noise on the NATIVE grid; CCF1d then resamples to dv.
        obs = Spectrum(wv=[MC_WAVE.copy()],
                       sp=[clean + rng.normal(0.0, 0.02, MC_WAVE.size)], name="o")
        ccf = CCF1d().CrossCorrelateSpec(spec_in=obs, template_in=template,
                                         dv=dv, VelBound=vel_bound).CombineCCFs()
        rv[k] = float(ccf.CorrCombined["RV"])
        erv[k] = float(ccf.CorrCombined["eRV"])
    return rv, erv


def test_rv_uncertainty_is_calibrated_by_monte_carlo(forest_template):
    rv, erv = _noisy_rvs(forest_template, n_real=200, dv=NATIVE_DV / 4, vel_bound=30)
    ratio = np.median(erv) / np.std(rv, ddof=1)
    # With 200 realizations the scatter itself is known to ~5% (1/sqrt(2*199)).
    # The cubic-spline resampling smooths the white noise a little, so the
    # resampled CCF peak is slightly higher than on the native grid and the
    # formula leans optimistic: measured ratios 0.93-1.02 over four seeds and
    # dv = 0.1-0.5 km/s. The band is that bias plus ~4 sigma of MC noise; the
    # old len(v)-based error gave 2.8 here (3.4 at VelBound=20).
    assert 0.8 < ratio < 1.2
    assert np.mean(rv) == pytest.approx(3.0, abs=4 * np.std(rv) / np.sqrt(rv.size))


def test_rv_uncertainty_does_not_depend_on_the_velocity_scan(forest_template):
    _, erv_narrow = _noisy_rvs(forest_template, n_real=5, dv=0.25, vel_bound=20)
    _, erv_wide = _noisy_rvs(forest_template, n_real=5, dv=0.25, vel_bound=200)
    np.testing.assert_allclose(erv_wide, erv_narrow, rtol=0.01)


def test_effective_bins_count_native_pixels_not_resampled_ones(forest_template):
    obs = Spectrum(wv=[MC_WAVE.copy()], sp=[_forest(MC_WAVE, 0.0)], name="o")
    fine = CCF1d().CrossCorrelateSpec(spec_in=obs, template_in=forest_template,
                                      dv=0.25, VelBound=20)
    coarse = CCF1d().CrossCorrelateSpec(spec_in=obs, template_in=forest_template,
                                        dv=NATIVE_DV, VelBound=20)
    # Both count the ~MC_WAVE.size native pixels (minus the resampling margin).
    for ccf in (fine, coarse):
        assert ccf.Corr["n_eff"][0] == pytest.approx(MC_WAVE.size, rel=0.01)


def test_per_order_and_bisector_errors_use_the_same_bin_count(forest_template):
    obs = Spectrum(wv=[MC_WAVE.copy(), MC_WAVE.copy()],
                   sp=[_forest(MC_WAVE, 3.0), _forest(MC_WAVE, 3.0)], name="o")
    ccf = CCF1d().CrossCorrelateSpec(spec_in=obs, template_in=forest_template,
                                     dv=0.25, VelBound=30, err_per_ord=True).CombineCCFs()
    # Two identical orders carry twice the bins of one: sqrt(2) smaller error.
    assert ccf.CorrCombined["n_eff"] == pytest.approx(ccf.Corr["n_eff"].sum())
    assert float(ccf.Corr["eRV"][0, 0]) == pytest.approx(
        np.sqrt(2) * float(ccf.CorrCombined["eRV"]), rel=1e-3)
    _, ebis, _ = ccf.calcBIS(ccf.CorrCombined["vel"], ccf.CorrCombined["corr"])
    assert ebis == pytest.approx(np.sqrt(2) * float(ccf.CorrCombined["eRV"]), rel=1e-3)
    # A single order of a two-order spectrum: which order's bins is ambiguous.
    with pytest.raises(ValueError, match="subset"):
        ccf.extract_RV(ccf.Corr["vel"], ccf.Corr["corr"][0], n_ord=1)
