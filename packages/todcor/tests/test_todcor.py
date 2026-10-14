"""Tests for the standalone TODCOR estimator (:mod:`todcor`).

The pure estimator (``todcor.estimator`` + ``todcor.simulate``) on synthetic
composites -- noiseless/noisy recovery, Parseval, the exact-shift chi2, the CRLB
against a Monte Carlo, sub-grid refinement, and window combination -- plus the
log-lambda primitives it re-exports from ``unicor``. The SPARTA-object adapter
``CCF2d`` is tested in SPARTA's ``tests/test_ccf2d.py``.

Self-contained: no external data, numpy only.
"""

import inspect

import numpy as np
import pytest

import todcor as ops
from todcor import (
    chi2_from_stats, combine_stats, composite_flux, crlb_from_map, model_flux,
    profile_from_stats, shift_template_flux, todcor_map, window_stats,
)


def _gauss(wave, center, depth, fwhm):
    """A Gaussian line of fractional ``depth`` (negative = absorption)."""
    sig = fwhm / 2.3548200450309493
    return 1.0 + depth * np.exp(-0.5 * ((wave - center) / sig) ** 2)


def _synthetic_templates():
    """An absorption-line 'primary' and an emission-line 'secondary'."""
    wave = np.linspace(4000.0, 5000.0, 8000)
    t1 = (
        _gauss(wave, 4101.7, -0.5, 1.5)
        * _gauss(wave, 4340.5, -0.45, 1.5)
        * _gauss(wave, 4471.5, -0.25, 1.2)
    )
    t2 = _gauss(wave, 4058.0, +0.6, 2.0) * _gauss(wave, 4685.7, +0.8, 3.0)
    return (wave, t1), (wave, t2)


def _build_case(dv=2.0, alpha=0.1, pad=12.0):
    """A shared log-lambda grid plus both apodised template residuals."""
    (w1, f1), (w2, f2) = _synthetic_templates()
    lo = max(w1.min(), w2.min()) + pad
    hi = min(w1.max(), w2.max()) - pad
    wave_grid, _ = ops.log_lambda_grid(lo, hi, dv)
    win = ops.tukey_window(len(wave_grid), alpha=alpha)
    g1 = (ops.resample(w1, f1, wave_grid, fill=1.0) - 1.0) * win
    g2 = (ops.resample(w2, f2, wave_grid, fill=1.0) - 1.0) * win
    return wave_grid, win, (w1, f1), (w2, f2), g1, g2, dv


# =============================================================================
# The DSP primitives shared with UNICOR
# =============================================================================
def test_log_lambda_grid_is_velocity_even():
    wave, u = ops.log_lambda_grid(4000.0, 4500.0, 2.5)
    steps = np.diff(np.log(wave)) * ops.C_KMS
    assert np.allclose(steps, 2.5, rtol=1e-12)
    assert np.allclose(np.diff(u), 2.5)


def test_parseval_weights_match_real_space():
    rng = np.random.default_rng(0)
    for n in (256, 257):
        x = rng.standard_normal(n)
        x -= x.mean()  # DC removed, as in the fit
        X = np.fft.rfft(x)[1:]
        w = ops.parseval_weights(n, sigma=1.0)
        assert np.sum(w * np.abs(X) ** 2) == pytest.approx(np.sum(x ** 2), rel=1e-9)


def test_shift_kms_matches_log_lambda_translation():
    # A template shifted by v must move by exactly shift_kms(v) along the
    # velocity coordinate of a log-lambda grid.
    wave, u = ops.log_lambda_grid(4000.0, 4200.0, 1.0)
    v = 137.0
    moved = ops.C_KMS * np.log(wave * (1.0 + v / ops.C_KMS) / wave[0])
    assert np.allclose(moved - u, ops.shift_kms(v), rtol=1e-12)


def test_resample_fill_leaves_continuum_outside_coverage():
    wave = np.linspace(4000.0, 4100.0, 100)
    flux = np.full_like(wave, 0.5)
    grid = np.linspace(3900.0, 4200.0, 50)
    filled = ops.resample(wave, flux, grid, fill=1.0)
    clamped = ops.resample(wave, flux, grid)
    assert filled[0] == 1.0 and filled[-1] == 1.0
    assert clamped[0] == 0.5 and clamped[-1] == 0.5


# =============================================================================
# The estimator
# =============================================================================
def test_noiseless_recovery():
    wave_grid, win, t1, t2, g1, g2, dv = _build_case()
    v1_true, v2_true, a1_true, a2_true = 120.0, -180.0, 0.7, 0.5
    flux = composite_flux(wave_grid, [t1, t2], [v1_true, v2_true],
                          [a1_true, a2_true])
    data_g = (flux - 1.0) * win

    v1 = np.arange(v1_true - 60, v1_true + 60 + 1e-6, 2.0)
    v2 = np.arange(v2_true - 60, v2_true + 60 + 1e-6, 2.0)
    res = todcor_map(data_g, g1, g2, dv, v1, v2, sigma=1.0)

    assert abs(res["best"]["v1"] - v1_true) <= 4.0
    assert abs(res["best"]["v2"] - v2_true) <= 4.0
    assert res["best"]["a1"] == pytest.approx(a1_true, abs=0.05)
    assert res["best"]["a2"] == pytest.approx(a2_true, abs=0.05)


def test_noisy_recovery_unbiased():
    wave_grid, win, t1, t2, g1, g2, dv = _build_case()
    v1_true, v2_true, a1_true, a2_true = 80.0, -140.0, 0.6, 0.4
    flux0 = composite_flux(wave_grid, [t1, t2], [v1_true, v2_true],
                           [a1_true, a2_true])
    v1 = np.arange(v1_true - 80, v1_true + 80 + 1e-6, 2.0)
    v2 = np.arange(v2_true - 80, v2_true + 80 + 1e-6, 2.0)

    rng = np.random.default_rng(42)
    sigma = 0.02
    rec1, rec2 = [], []
    for _ in range(12):
        flux = flux0 + rng.normal(0.0, sigma, size=flux0.shape)
        data_g = (flux - 1.0) * win
        res = todcor_map(data_g, g1, g2, dv, v1, v2, sigma=sigma)
        rec1.append(res["best"]["v1"])
        rec2.append(res["best"]["v2"])
    # Mean recovered velocity should sit within a pixel of truth.
    assert abs(np.mean(rec1) - v1_true) <= dv
    assert abs(np.mean(rec2) - v2_true) <= dv


def test_exact_shift_chi2_zero_even_and_odd_n():
    # A composite built with model_flux at exact (v1,v2,a1,a2) must be fit with
    # chi2 ~ 0 -- including even-N windows (Nyquist mode handled consistently).
    for n in (88, 256, 400, 401):
        grid = np.exp(np.linspace(np.log(4400), np.log(4520), n))
        dv = np.log(grid[1] / grid[0]) * ops.C_KMS
        g1 = -0.5 * np.exp(-0.5 * ((grid - 4450) / 2.0) ** 2)
        g2 = +0.6 * np.exp(-0.5 * ((grid - 4485) / 3.0) ** 2)
        v1, v2, a1, a2 = 33.0, -51.0, 0.7, 0.5
        data = model_flux(grid, dv, g1, g2, v1, v2, a1, a2) - 1.0
        st = window_stats(data, g1, g2, dv, np.array([v1]), np.array([v2]),
                          sigma=0.01)
        r = profile_from_stats(st, np.array([v1]), np.array([v2]), nonneg=False)
        assert abs(r["best"]["chi2"]) < 1e-6, f"N={n}: chi2={r['best']['chi2']:.2e}"
        assert r["a1"][0, 0] == pytest.approx(a1, abs=1e-6)
        assert r["a2"][0, 0] == pytest.approx(a2, abs=1e-6)


def test_crlb_matches_monte_carlo():
    # The CRLB from the noiseless-map curvature should predict the Monte-Carlo
    # scatter of the recovered velocities (known noise sigma). The slowest test
    # in the suite (~7 s): 120 map evaluations.
    wave_grid, win, t1, t2, g1, g2, dv = _build_case(dv=1.0)
    v1_true, v2_true, sigma = 100.0, -150.0, 0.02
    flux0 = composite_flux(wave_grid, [t1, t2], [v1_true, v2_true], [0.7, 0.5])
    v1 = np.arange(v1_true - 25, v1_true + 25 + 1e-6, 1.0)
    v2 = np.arange(v2_true - 25, v2_true + 25 + 1e-6, 1.0)

    res0 = todcor_map((flux0 - 1.0) * win, g1, g2, dv, v1, v2, sigma=sigma)
    cr = crlb_from_map(res0["logL_profile"], v1, v2, half=5)

    rng = np.random.default_rng(3)
    r1, r2 = [], []
    for _ in range(120):
        flux = flux0 + rng.normal(0.0, sigma, size=flux0.shape)
        r = todcor_map((flux - 1.0) * win, g1, g2, dv, v1, v2, sigma=sigma)
        r1.append(r["best"]["v1"])
        r2.append(r["best"]["v2"])
    # CRLB within a factor ~1.6 of the empirical scatter (both directions).
    assert 0.6 < cr["sigma_v1"] / np.std(r1) < 1.7
    assert 0.6 < cr["sigma_v2"] / np.std(r2) < 1.7


def test_subgrid_refinement_recovers_offgrid_truth():
    # Inject velocities that fall BETWEEN grid nodes; the quadratic-peak
    # refinement must beat the +/- step/2 quantization of the argmax node.
    wave_grid, win, t1, t2, g1, g2, dv = _build_case()
    v1_true, v2_true = 120.7, -180.6      # off the 2 km/s grid
    flux = composite_flux(wave_grid, [t1, t2], [v1_true, v2_true], [0.7, 0.5])
    data_g = (flux - 1.0) * win
    v1 = np.round(np.arange(v1_true - 60, v1_true + 60 + 1e-6, 2.0)) * 1.0
    v2 = np.round(np.arange(v2_true - 60, v2_true + 60 + 1e-6, 2.0))

    res = todcor_map(data_g, g1, g2, dv, v1, v2, sigma=0.01)
    cr = crlb_from_map(res["logL_profile"], v1, v2)

    # The refined peak lands well inside a grid step of the truth...
    ref_v1, ref_v2 = cr["peak_refined"]
    assert abs(ref_v1 - v1_true) < 0.6
    assert abs(ref_v2 - v2_true) < 0.6
    # ...and is strictly better than the raw grid node.
    assert abs(ref_v1 - v1_true) < abs(cr["peak"][0] - v1_true) + 1e-9
    assert abs(ref_v2 - v2_true) < abs(cr["peak"][1] - v2_true) + 1e-9
    # Refinement never leaves the neighbouring-node cell.
    assert abs(ref_v1 - cr["peak"][0]) <= 2.0
    assert abs(ref_v2 - cr["peak"][1]) <= 2.0


def test_marginal_map_peaks_near_truth():
    wave_grid, win, t1, t2, g1, g2, dv = _build_case()
    v1_true, v2_true = 100.0, -100.0
    flux = composite_flux(wave_grid, [t1, t2], [v1_true, v2_true], [0.7, 0.6])
    data_g = (flux - 1.0) * win
    v1 = np.arange(v1_true - 60, v1_true + 60 + 1e-6, 2.0)
    v2 = np.arange(v2_true - 60, v2_true + 60 + 1e-6, 2.0)
    res = todcor_map(data_g, g1, g2, dv, v1, v2, sigma=0.02)
    i, j = np.unravel_index(np.argmax(res["logL_marginal"]),
                            res["logL_marginal"].shape)
    assert abs(v1[i] - v1_true) <= 4.0
    assert abs(v2[j] - v2_true) <= 4.0


def test_multiwindow_combination_recovers():
    # Two disjoint windows, one carrying mostly the primary, one the blend,
    # combined into a single joint fit with shared velocities/amplitudes.
    dv = 2.0
    wave = np.linspace(4000.0, 5000.0, 8000)
    prim = _gauss(wave, 4101.7, -0.5, 1.5) * _gauss(wave, 4340.5, -0.45, 1.5)
    sec = _gauss(wave, 4058.0, +0.6, 2.0) * _gauss(wave, 4685.7, +0.8, 3.0)
    t_prim, t_sec = (wave, prim), (wave, sec)
    v1_true, v2_true, a1_true, a2_true = 90.0, -150.0, 0.7, 0.5

    v1 = np.arange(v1_true - 60, v1_true + 60 + 1e-6, 2.0)
    v2 = np.arange(v2_true - 60, v2_true + 60 + 1e-6, 2.0)

    stats = []
    for lo, hi in [(4030, 4380), (4640, 4740)]:
        grid, _ = ops.log_lambda_grid(lo, hi, dv)
        win = ops.tukey_window(len(grid), alpha=0.2)
        g1 = (ops.resample(*t_prim, grid, fill=1.0) - 1.0) * win
        g2 = (ops.resample(*t_sec, grid, fill=1.0) - 1.0) * win
        flux = composite_flux(grid, [t_prim, t_sec], [v1_true, v2_true],
                              [a1_true, a2_true])
        data_g = (flux - 1.0) * win
        stats.append(window_stats(data_g, g1, g2, dv, v1, v2, sigma=1.0))

    res = profile_from_stats(combine_stats(stats), v1, v2)
    assert abs(res["best"]["v1"] - v1_true) <= 4.0
    assert abs(res["best"]["v2"] - v2_true) <= 4.0
    assert res["best"]["a1"] == pytest.approx(a1_true, abs=0.08)
    assert res["best"]["a2"] == pytest.approx(a2_true, abs=0.08)


def test_combine_stats_rejects_mismatched_velocity_grids():
    wave_grid, win, t1, t2, g1, g2, dv = _build_case()
    flux = composite_flux(wave_grid, [t1, t2], [0.0, 0.0], [0.7, 0.5])
    data_g = (flux - 1.0) * win
    sa = window_stats(data_g, g1, g2, dv, np.arange(-10, 11, 2.0),
                      np.arange(-10, 11, 2.0))
    sb = window_stats(data_g, g1, g2, dv, np.arange(-10, 11, 5.0),
                      np.arange(-10, 11, 5.0))
    with pytest.raises(ValueError, match="same velocity grid"):
        combine_stats([sa, sb])


def test_combine_stats_weights_rescale_each_windows_noise():
    # The statistics are linear in the weights 1/sigma^2, so weighting a
    # unit-noise window by 1/sigma^2 must equal computing it at sigma -- this
    # is what lets CCF2d re-weight orders without new FFTs. nu, the expected
    # chi2 at the window's own noise, does not change with that noise.
    wave_grid, win, t1, t2, g1, g2, dv = _build_case()
    v = np.arange(-10, 11, 2.0)
    rng = np.random.default_rng(11)
    sigmas = (0.02, 0.005)
    flux = composite_flux(wave_grid, [t1, t2], [3.0, -4.0], [0.7, 0.5])
    unit, ref = [], []
    for s in sigmas:
        data_g = (flux - 1.0 + rng.normal(0.0, s, flux.size)) * win
        unit.append(window_stats(data_g, g1, g2, dv, v, v))
        ref.append(window_stats(data_g, g1, g2, dv, v, v, sigma=s))
    weighted = combine_stats(unit, weights=[1.0 / s ** 2 for s in sigmas])
    expected = combine_stats(ref)
    for k in ('S11', 'S22', 'SD', 'b1', 'b2', 'A12', 'nu'):
        assert np.allclose(weighted[k], expected[k], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize('weights', [[1.0], [1.0, 0.0], [1.0, np.nan], [1.0, -2.0]])
def test_combine_stats_rejects_bad_weights(weights):
    wave_grid, win, t1, t2, g1, g2, dv = _build_case()
    data_g = (composite_flux(wave_grid, [t1, t2], [0.0, 0.0], [0.7, 0.5]) - 1.0) * win
    st = window_stats(data_g, g1, g2, dv, np.arange(-4, 5, 2.0), np.arange(-4, 5, 2.0))
    with pytest.raises(ValueError, match="weights"):
        combine_stats([st, st], weights=weights)


def test_chi2_from_stats_splits_the_joint_chi2_by_window():
    # Per-window chi2 at the joint amplitudes sums to the joint chi2: the
    # per-window noise estimate of CCF2d rests on it.
    t_prim, t_sec = _synthetic_templates()
    dv, v = 2.0, np.arange(-20.0, 21.0, 2.0)
    rng = np.random.default_rng(12)
    stats = []
    for lo, hi in [(4030, 4380), (4640, 4740)]:
        grid, _ = ops.log_lambda_grid(lo, hi, dv)
        win = ops.tukey_window(len(grid), alpha=0.2)
        g1 = (ops.resample(*t_prim, grid, fill=1.0) - 1.0) * win
        g2 = (ops.resample(*t_sec, grid, fill=1.0) - 1.0) * win
        flux = composite_flux(grid, [t_prim, t_sec], [4.0, -6.0], [0.7, 0.4])
        stats.append(window_stats((flux - 1.0 + rng.normal(0.0, 0.01, flux.size)) * win,
                                  g1, g2, dv, v, v))
    res = profile_from_stats(combine_stats(stats), v, v)
    parts = [chi2_from_stats(st, res['a1'], res['a2']) for st in stats]
    assert np.allclose(sum(parts), res['chi2'], rtol=1e-12, atol=0.0)
    assert np.all(parts[0] > 0) and np.all(parts[1] > 0)


def test_window_stats_rejects_grid_length_mismatch():
    wave_grid, win, t1, t2, g1, g2, dv = _build_case()
    with pytest.raises(ValueError, match="share one grid"):
        window_stats(np.zeros(len(g1) - 1), g1, g2, dv, np.array([0.0]),
                     np.array([0.0]))


def test_crlb_on_a_grid_edge_is_nan_and_not_extrapolated():
    v1 = np.arange(-10.0, 11.0)
    v2 = np.arange(-10.0, 11.0)
    V1, V2 = np.meshgrid(v1, v2, indexing="ij")
    logL = -0.5 * ((V1 - 14.0) ** 2 + (V2 - 2.0) ** 2)   # true peak beyond v1 = +10
    cr = crlb_from_map(logL, v1, v2)
    assert cr["on_edge"]
    assert np.isnan(cr["sigma_v1"]) and np.isnan(cr["sigma_v2"])
    assert cr["peak_refined"] == cr["peak"] == (10.0, 2.0)


def test_crlb_interior_peak_reports_on_edge_false():
    v1 = np.arange(-10.0, 11.0)
    V1, V2 = np.meshgrid(v1, v1, indexing="ij")
    cr = crlb_from_map(-0.5 * ((V1 - 0.3) ** 2 + (V2 + 0.2) ** 2 / 4.0), v1, v1)
    assert not cr["on_edge"]
    assert cr["sigma_v1"] == pytest.approx(1.0, rel=1e-6)
    assert cr["sigma_v2"] == pytest.approx(2.0, rel=1e-6)


# =============================================================================
# Oversampled analysis grids: Fisher weights and noise degrees of freedom
# =============================================================================
def _narrow_line_flux(wave, centers, depths, fwhm=8.0, v=0.0):
    """Analytic narrow-line spectrum (FWHM in km/s), Doppler-shifted by ``v``."""
    rest = np.asarray(wave, float)[None, :] / (1.0 + v / ops.C_KMS)
    sig = centers[:, None] * fwhm / 2.3548200450309493 / ops.C_KMS
    return 1.0 - np.sum(depths[:, None]
                        * np.exp(-0.5 * ((rest - centers[:, None]) / sig) ** 2), axis=0)


def _impulse_responses(wave_native, wave_grid, n_imp=6, spacing=30):
    """Analysis-grid responses of `resample` to single native-pixel impulses.

    The impulses sit well inside the analysis span, so no response is clipped.
    """
    first = np.searchsorted(wave_native, wave_grid[0]) + spacing // 2
    out = []
    for j in first + spacing * np.arange(n_imp):
        x = np.zeros(len(wave_native))
        x[j] = 1.0
        y = ops.resample(wave_native, x, wave_grid)
        out.append(y[y != 0.0])
    return out


def test_oversampling_divides_the_weights():
    wave_grid, win, t1, t2, g1, g2, dv = _build_case()
    flux = composite_flux(wave_grid, [t1, t2], [10.0, -20.0], [0.7, 0.5])
    data_g = (flux - 1.0) * win
    v = np.arange(-30.0, 31.0, 5.0)
    st1 = window_stats(data_g, g1, g2, dv, v, v, sigma=0.02)
    st8 = window_stats(data_g, g1, g2, dv, v, v, sigma=0.02, oversampling=8.0)
    for k in ("S11", "S22", "SD", "b1", "b2", "A12", "nu"):
        assert np.allclose(st8[k], st1[k] / 8.0, rtol=1e-12), k
    # White noise on the analysis grid: one degree of freedom per retained
    # Fourier coefficient (DC and the even-N Nyquist mode are dropped).
    n = len(data_g)
    assert st1["nu"] == pytest.approx(n - 2 if n % 2 == 0 else n - 1)
    with pytest.raises(ValueError, match="oversampling"):
        window_stats(data_g, g1, g2, dv, v, v, oversampling=0.5)
    with pytest.raises(ValueError, match="noise_psd"):
        window_stats(data_g, g1, g2, dv, v, v, noise_psd=np.ones(n))


def test_resampled_noise_psd_counts_native_pixels():
    # Linear interpolation of white native noise onto an M-times finer grid:
    # the mean noise power per analysis pixel is (2 + 1/M^2)/3 sigma^2 (-> 2/3),
    # and the noise occupies only n/M independent samples.
    # The closed form assumes analysis nodes aligned with native nodes.
    M = 10
    wave_native, _ = ops.log_lambda_grid(4450.0, 4520.0, 1.0)
    wave_grid, _ = ops.log_lambda_grid(wave_native[2], wave_native[-3], 1.0 / M)
    n = len(wave_grid)
    psd = ops.resampled_noise_psd(_impulse_responses(wave_native, wave_grid), n, M)
    assert psd.shape == (n // 2,)
    assert 2.0 * psd.sum() / n == pytest.approx((2.0 + 1.0 / M ** 2) / 3.0, rel=1e-3)
    g = np.zeros(n)
    st = window_stats(g, g + 1e-3, g + 2e-3, 1.0 / M, np.array([0.0]),
                      np.array([0.0]), oversampling=M, noise_psd=psd)
    assert st["nu"] == pytest.approx((n / M) * (2.0 + 1.0 / M ** 2) / 3.0, rel=2e-3)
    with pytest.raises(ValueError, match="longer than the window"):
        ops.resampled_noise_psd([np.ones(n + 1)], n, M)



def test_resampled_noise_psd_weights_average_the_responses():
    # Responses at different sub-pixel phases are averaged with the number of
    # native pixels at each phase; a phase no native pixel has drops out.
    n, M = 400, 0.5
    h1, h2 = np.array([1.0]), np.array([0.6, 0.6, -0.1])
    one = ops.resampled_noise_psd([h1], n, M)
    two = ops.resampled_noise_psd([h2], n, M)
    np.testing.assert_allclose(ops.resampled_noise_psd([h1, h2], n, M, weights=[3, 1]),
                               (3 * one + two) / 4, rtol=1e-12)
    np.testing.assert_allclose(ops.resampled_noise_psd([h1, h2], n, M, weights=[1, 0]),
                               one, rtol=1e-12)
    # A unit response per native pixel: white, with n / M native pixels' power.
    np.testing.assert_allclose(one, 1.0 / M, rtol=1e-12)
    with pytest.raises(ValueError, match="weights"):
        ops.resampled_noise_psd([h1, h2], n, M, weights=[0, 0])
    with pytest.raises(ValueError, match="weights"):
        ops.resampled_noise_psd([h1, h2], n, M, weights=[1])

def test_crlb_and_chi2_red_honest_on_an_oversampled_grid():
    # Noise drawn on a ~1 km/s native grid, then interpolated onto a 0.1 km/s
    # analysis grid: ~10 analysis pixels share each native noise value. With
    # the weights divided by the oversampling and the dof taken from the
    # resampled-noise PSD, the CRLB must predict the Monte-Carlo scatter and
    # chi2_red must be ~1. (Before the fix the CRLB was ~sqrt(10) too small.)
    M, sigma = 10, 0.1
    rng = np.random.default_rng(11)
    c1, d1 = rng.uniform(4455.0, 4515.0, 20), rng.uniform(0.1, 0.6, 20)
    c2, d2 = rng.uniform(4455.0, 4515.0, 20), rng.uniform(0.1, 0.6, 20)
    v1_true, v2_true, a1, a2 = 3.3, -2.6, 0.7, 0.4

    wave_native, _ = ops.log_lambda_grid(4450.0, 4520.0, 1.0)
    wave_grid, _ = ops.log_lambda_grid(4452.0, 4518.0, 1.0 / M)
    n, dv = len(wave_grid), 1.0 / M
    win = ops.tukey_window(n, alpha=0.1)
    g1 = (_narrow_line_flux(wave_grid, c1, d1) - 1.0) * win
    g2 = (_narrow_line_flux(wave_grid, c2, d2) - 1.0) * win
    flux0 = 1.0 + a1 * (_narrow_line_flux(wave_native, c1, d1, v=v1_true) - 1.0) \
                + a2 * (_narrow_line_flux(wave_native, c2, d2, v=v2_true) - 1.0)
    psd = ops.resampled_noise_psd(_impulse_responses(wave_native, wave_grid), n, M)
    psd = psd * np.mean(win ** 2)       # the taper scales the noise too

    v1 = v1_true + np.arange(-0.6, 0.6 + 1e-9, 0.05)
    v2 = v2_true + np.arange(-0.6, 0.6 + 1e-9, 0.05)
    kw = dict(sigma=sigma, oversampling=M, noise_psd=psd)

    def fit(flux_native):
        data_g = (ops.resample(wave_native, flux_native, wave_grid) - 1.0) * win
        res = todcor_map(data_g, g1, g2, dv, v1, v2, **kw)
        return res, crlb_from_map(res["logL_profile"], v1, v2)

    _, cr0 = fit(flux0)
    r1, r2, chi2_red = [], [], []
    for _ in range(60):
        res, cr = fit(flux0 + rng.normal(0.0, sigma, size=flux0.shape))
        r1.append(cr["peak_refined"][0])
        r2.append(cr["peak_refined"][1])
        chi2_red.append(res["best"]["chi2_red"])
    assert 0.75 < np.std(r1) / cr0["sigma_v1"] < 1.3
    assert 0.75 < np.std(r2) / cr0["sigma_v2"] < 1.3
    assert np.mean(chi2_red) == pytest.approx(1.0, abs=0.04)


# =============================================================================
# Velocity conventions
# =============================================================================
# A component shifted by v moves by s = c ln(lambda_obs / lambda_rest) along the
# log-lambda axis: c ln(1 + v/c) under 'cz' (IAU 2000; IRAF fxcor), c artanh(v/c)
# under 'relativistic' (factor sqrt((1+b)/(1-b))). The fits below are noise-free
# and analytic (no interpolation of the data), so the maps peak exactly at the
# injected velocities of the matching convention.
_C = 299792.458
CONVENTIONS = ('cz', 'relativistic')
SB2_PAIRS = [(150.0, -150.0), (-150.0, 150.0), (100.0, -100.0), (-100.0, 100.0),
             (50.0, -50.0), (-50.0, 50.0), (10.0, -10.0), (-10.0, 10.0)]


def _factor(v, convention):
    """lambda_obs / lambda_rest, written out from the two definitions."""
    b = np.asarray(v, float) / _C
    return 1.0 + b if convention == 'cz' else np.sqrt((1.0 + b) / (1.0 - b))


def _forest(seed, n_lines, sigma_kms, lam_lo=5000.0, lam_hi=5060.0):
    """Rest-frame Gaussian-line forest (in velocity) as a function of lambda,
    its lines 400 km/s clear of the window edges (no wrap-around)."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(400.0, _C * np.log(lam_hi / lam_lo) - 400.0, n_lines)
    depths = rng.uniform(0.05, 0.6, n_lines)

    def flux(wave):
        u = _C * np.log(np.asarray(wave, float) / lam_lo)
        return np.prod(1.0 - depths * np.exp(-0.5 * ((u[:, None] - centres) / sigma_kms) ** 2),
                       axis=1)
    return flux


_T1, _T2 = _forest(11, 60, 4.0), _forest(12, 40, 6.0)


def _sb2_window(v1, v2, convention, a1=0.7, a2=0.4, dv=1.0):
    """Analytic noise-free composite residual and rest-frame template residuals."""
    wave, _ = ops.log_lambda_grid(5000.0, 5060.0, dv)
    data_g = (a1 * (_T1(wave / _factor(v1, convention)) - 1.0)
              + a2 * (_T2(wave / _factor(v2, convention)) - 1.0))
    return wave, data_g, _T1(wave) - 1.0, _T2(wave) - 1.0, dv


def _refined_peak(v1, v2, fit_convention, inject_convention=None, half=20, step=0.005):
    """Sub-grid TODCOR peak (km/s) on a fine map (``step`` km/s) around the truth.

    The map nodes are deliberately offset from the truth, so the answer comes
    from the quadratic refinement, not from a node that happens to sit on it.
    """
    wave, data_g, g1, g2, dv = _sb2_window(v1, v2, inject_convention or fit_convention)
    f1 = v1 + 0.0013 + np.arange(-half, half + 1) * step
    f2 = v2 - 0.0021 + np.arange(-half, half + 1) * step
    res = todcor_map(data_g, g1, g2, dv, f1, f2, velocity_convention=fit_convention)
    crlb = crlb_from_map(res['logL_profile'], f1, f2)
    assert not crlb['on_edge']
    return np.array(crlb['peak_refined']), res['best']


@pytest.mark.parametrize('conv', CONVENTIONS)
@pytest.mark.parametrize('v1, v2', SB2_PAIRS)
def test_sb2_injection_is_recovered_to_the_cm_per_s(v1, v2, conv):
    peak, best = _refined_peak(v1, v2, conv)
    np.testing.assert_array_less(np.abs(peak - [v1, v2]) * 1e3, 0.05)   # m/s
    assert best['a1'] == pytest.approx(0.7, abs=1e-3)
    assert best['a2'] == pytest.approx(0.4, abs=1e-3)


@pytest.mark.parametrize('v1, v2', [(150.0, -150.0), (50.0, -100.0)])
def test_cz_and_relativistic_fits_differ_by_v2_over_2c(v1, v2):
    # The same data fitted in both conventions: the shift s is the same, and
    # c (expm1(s/c) - tanh(s/c)) ~ v^2 / 2c for either sign of v.
    cz, _ = _refined_peak(v1, v2, 'cz', inject_convention='cz')
    rel, _ = _refined_peak(v1, v2, 'relativistic', inject_convention='cz')
    np.testing.assert_allclose(cz - rel, np.array([v1, v2]) ** 2 / (2 * _C), rtol=3e-3)


@pytest.mark.parametrize('conv', CONVENTIONS)
def test_forward_model_follows_the_convention(conv):
    v1, v2, a1, a2 = 150.0, -100.0, 0.7, 0.4
    wave, data_g, g1, g2, dv = _sb2_window(v1, v2, conv, a1, a2)
    t_wave, _ = ops.log_lambda_grid(4990.0, 5070.0, 0.01)
    t1, t2 = (t_wave, _T1(t_wave)), (t_wave, _T2(t_wave))
    # Physical shift (resampling a finely sampled template) ...
    np.testing.assert_allclose(
        shift_template_flux(*t1, wave, v1, velocity_convention=conv),
        _T1(wave / _factor(v1, conv)), atol=1e-5)
    np.testing.assert_allclose(
        composite_flux(wave, [t1, t2], [v1, v2], [a1, a2], velocity_convention=conv),
        1.0 + data_g, atol=1e-5)
    # ... and the Fourier phase shift the fit uses.
    np.testing.assert_allclose(
        model_flux(wave, dv, g1, g2, v1, v2, a1, a2, velocity_convention=conv),
        1.0 + data_g, atol=1e-9)


def test_the_default_convention_is_cz():
    for fn in (window_stats, todcor_map, shift_template_flux, composite_flux, model_flux):
        assert inspect.signature(fn).parameters['velocity_convention'].default == 'cz'
    wave, data_g, g1, g2, dv = _sb2_window(50.0, -50.0, 'cz')
    grid = np.array([49.0, 50.0, 51.0])
    default = window_stats(data_g, g1, g2, dv, grid, -grid)
    cz = window_stats(data_g, g1, g2, dv, grid, -grid, velocity_convention='cz')
    for key in ('b1', 'b2', 'A12'):
        np.testing.assert_array_equal(default[key], cz[key])


@pytest.mark.parametrize('bad', ['classical', 'CZ', None])
def test_an_unknown_convention_is_refused(bad):
    wave, data_g, g1, g2, dv = _sb2_window(10.0, -10.0, 'cz')
    t = (wave, _T1(wave))
    calls = (lambda: window_stats(data_g, g1, g2, dv, [0.0], [0.0], velocity_convention=bad),
             lambda: todcor_map(data_g, g1, g2, dv, [0.0], [0.0], velocity_convention=bad),
             lambda: shift_template_flux(*t, wave, 10.0, velocity_convention=bad),
             lambda: composite_flux(wave, [t, t], [10.0, -10.0], [0.5, 0.5],
                                    velocity_convention=bad),
             lambda: model_flux(wave, dv, g1, g2, 10.0, -10.0, 0.5, 0.5,
                                velocity_convention=bad))
    for call in calls:
        with pytest.raises(ValueError, match='convention'):
            call()
