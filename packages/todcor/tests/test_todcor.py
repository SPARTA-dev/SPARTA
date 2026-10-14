"""Tests for the standalone TODCOR estimator (:mod:`todcor`).

The pure estimator (``todcor.estimator`` + ``todcor.simulate``) on synthetic
composites -- noiseless/noisy recovery, Parseval, the exact-shift chi2, the CRLB
against a Monte Carlo, sub-grid refinement, and window combination -- plus the
log-lambda primitives it re-exports from ``unicor``. The SPARTA-object adapter
``CCF2d`` is tested in SPARTA's ``tests/test_ccf2d.py``.

Self-contained: no external data, numpy only.
"""

import numpy as np
import pytest

import todcor as ops
from todcor import (
    combine_stats, composite_flux, crlb_from_map, model_flux,
    profile_from_stats, todcor_map, window_stats,
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


def test_window_stats_rejects_grid_length_mismatch():
    wave_grid, win, t1, t2, g1, g2, dv = _build_case()
    with pytest.raises(ValueError, match="share one grid"):
        window_stats(np.zeros(len(g1) - 1), g1, g2, dv, np.array([0.0]),
                     np.array([0.0]))
