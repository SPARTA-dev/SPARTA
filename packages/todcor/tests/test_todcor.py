"""Tests for the standalone TODCOR estimator (:mod:`todcor`).

The pure estimator (``todcor.estimator`` + ``todcor.simulate``) on synthetic
composites -- noiseless/noisy recovery, the exact-shift chi2, the CRLB against
a Monte Carlo, sub-grid refinement, window combination, and the
extended-template window (no wrap-around, a free offset, the one-template
reduction to UNICOR's CCF, cm/s recovery with lines across the window edges,
the degree-6 peak) -- plus the log-lambda primitives it re-exports from
``unicor``. The SPARTA-object adapter
``CCF2d`` is tested in SPARTA's ``tests/test_ccf2d.py``.

Self-contained: no external data, numpy only.
"""

import inspect

import numpy as np
import pytest

import todcor as ops
from unicor.dsp import tukey_window
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
    win = tukey_window(len(wave_grid), alpha=alpha)
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


@pytest.mark.parametrize("e", [0, 7, 20])
def test_exact_shift_chi2_zero_even_and_odd_n(e):
    # A composite built with model_flux at exact (v1,v2,a1,a2) must be fit with
    # chi2 ~ 0 -- for even and odd window and extended lengths (the Nyquist
    # mode of the shift handled consistently), with or without an extension,
    # and whatever constant offset the data carry (it is profiled out).
    for n in (88, 256, 400, 401):
        grid = np.exp(np.linspace(np.log(4400), np.log(4520), n + 2 * e))
        dv = np.log(grid[1] / grid[0]) * ops.C_KMS
        g1 = -0.5 * np.exp(-0.5 * ((grid - 4450) / 2.0) ** 2)
        g2 = +0.6 * np.exp(-0.5 * ((grid - 4485) / 3.0) ** 2)
        v1, v2, a1, a2 = 33.0, -51.0, 0.7, 0.5
        data = model_flux(g1, g2, n, dv, v1, v2, a1, a2) + 0.37
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
        win = tukey_window(len(grid), alpha=0.2)
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
        win = tukey_window(len(grid), alpha=0.2)
        g1 = (ops.resample(*t_prim, grid, fill=1.0) - 1.0) * win
        g2 = (ops.resample(*t_sec, grid, fill=1.0) - 1.0) * win
        flux = composite_flux(grid, [t_prim, t_sec], [4.0, -6.0], [0.7, 0.4])
        stats.append(window_stats((flux - 1.0 + rng.normal(0.0, 0.01, flux.size)) * win,
                                  g1, g2, dv, v, v))
    res = profile_from_stats(combine_stats(stats), v, v)
    parts = [chi2_from_stats(st, res['a1'], res['a2']) for st in stats]
    assert np.allclose(sum(parts), res['chi2'], rtol=1e-12, atol=0.0)
    assert np.all(parts[0] > 0) and np.all(parts[1] > 0)


@pytest.mark.parametrize("n_data", [+1, +2])
def test_window_stats_rejects_grid_length_mismatch(n_data):
    # The templates must be the window's grid extended on each side (one more
    # pixel on the right allowed, for an odd length): data longer than the
    # templates, or two templates of different length, are refused.
    wave_grid, win, t1, t2, g1, g2, dv = _build_case()
    with pytest.raises(ValueError, match="g1 must be the window's grid extended"):
        window_stats(np.zeros(len(g1) + n_data), g1, g2, dv, np.array([0.0]),
                     np.array([0.0]))
    with pytest.raises(ValueError, match="share one extended grid"):
        window_stats(np.zeros(len(g1) - 2), g1, g2[1:], dv, np.array([0.0]),
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
# The two-dimensional correlation R (output='ccf') and the output modes
# =============================================================================
def _noisy_case(seed=7, sigma=0.01):
    wave_grid, win, t1, t2, g1, g2, dv = _build_case()
    flux = composite_flux(wave_grid, [t1, t2], [100.0, -150.0], [0.7, 0.5])
    flux = flux + np.random.default_rng(seed).normal(0.0, sigma, flux.shape)
    v1 = np.arange(40.0, 161.0, 2.0)
    v2 = np.arange(-210.0, -89.0, 2.0)
    return window_stats((flux - 1.0) * win, g1, g2, dv, v1, v2, sigma=sigma), v1, v2


@pytest.mark.parametrize("nonneg", [False, True])
def test_r_is_the_explained_power_fraction_and_shares_the_logl_argmax(nonneg):
    st, v1, v2 = _noisy_case()
    res = profile_from_stats(st, v1, v2, nonneg=nonneg)
    R = res["R"]
    assert np.all((R >= 0) & (R <= 1))
    # R^2 = 1 - chi2_min / SD, so logL_profile = -(SD/2)(1 - R^2) node by node.
    np.testing.assert_allclose(res["logL_profile"], -0.5 * st["SD"] * (1 - R ** 2),
                               rtol=1e-10, atol=1e-8 * st["SD"])
    assert np.argmax(R) == np.argmax(res["logL_profile"])
    i, j = np.unravel_index(np.argmax(R), R.shape)
    assert res["best"]["R"] == R[i, j]
    assert 0.9 < res["best"]["R"] < 1.0


def test_r_is_the_zucker_mazeh_two_dimensional_correlation():
    # Zucker & Mazeh (1994): with the light ratio fitted at every node,
    # R^2 = (c1^2 - 2 c1 c2 c12 + c2^2) / (1 - c12^2), with c1, c2 the
    # normalised data-template correlations and c12 the template-template one.
    st, v1, v2 = _noisy_case()
    c1 = st["b1"][:, None] / np.sqrt(st["S11"] * st["SD"])
    c2 = st["b2"][None, :] / np.sqrt(st["S22"] * st["SD"])
    c12 = st["A12"] / np.sqrt(st["S11"] * st["S22"])
    zm94 = np.sqrt((c1 ** 2 - 2 * c1 * c2 * c12 + c2 ** 2) / (1 - c12 ** 2))
    np.testing.assert_allclose(profile_from_stats(st, v1, v2)["R"], zm94, rtol=1e-9)


@pytest.mark.parametrize("n", [256, 257])
def test_r_is_one_for_a_noiseless_composite_at_the_truth(n):
    grid = np.exp(np.linspace(np.log(4400), np.log(4520), n))
    dv = np.log(grid[1] / grid[0]) * ops.C_KMS
    g1 = -0.5 * np.exp(-0.5 * ((grid - 4450) / 2.0) ** 2)
    g2 = +0.6 * np.exp(-0.5 * ((grid - 4485) / 3.0) ** 2)
    # Templates on the window extended by 15 pixels (~470 km/s) per side.
    data = model_flux(g1, g2, n - 30, dv, 33.0, -51.0, 0.7, 0.5)
    # The lines are ~130 km/s wide (sigma): scan well beyond that.
    v1, v2 = 33.0 + 25.0 * np.arange(-12, 13), -51.0 + 25.0 * np.arange(-12, 13)
    res = todcor_map(data, g1, g2, dv, v1, v2, sigma=0.01)
    assert (res["best"]["v1"], res["best"]["v2"]) == (33.0, -51.0)
    assert res["best"]["R"] == pytest.approx(1.0, abs=1e-12)
    assert res["R"].min() < 0.9


def _periodic_case(n=1201, k_max=60, n_lags=40, seed=11):
    """Data and template 1 on one odd-N log grid, sharing the low Fourier modes
    1..k_max; template 2 only in the modes above, so it is orthogonal to both at
    every shift. Template 1 is periodic over the window -- extended by repeating
    it, unicor's lag k is exactly TODCOR's circular shift by k pixels."""
    rng = np.random.default_rng(seed)
    wave, _ = ops.log_lambda_grid(5000.0, 5000.0 * np.exp(n * 0.5 / ops.C_KMS), 0.5)
    wave = wave[:n]
    def band(lo, hi):
        spec = np.zeros(n // 2 + 1, complex)
        spec[lo:hi + 1] = rng.normal(size=hi + 1 - lo) + 1j * rng.normal(size=hi + 1 - lo)
        return np.fft.irfft(spec, n)
    g1 = band(1, k_max)
    shifted = np.roll(g1, 7)                             # the data: g1 moved by 7 pixels
    data = 0.8 * shifted + 0.3 * band(1, k_max)          # plus 'noise' in the same band
    g2 = band(k_max + 1, n // 2)
    template_ext = np.concatenate([g1[-n_lags:], g1, g1[:n_lags]])
    return wave, data, g1, g2, template_ext, n_lags


@pytest.mark.parametrize("nonneg", [False, True])
def test_one_template_r_reduces_to_unicors_ccf(nonneg):
    # With template 2 orthogonal to the data and to template 1 at every shift,
    # c2 = c12 = 0 and R(v1, v2) = |c1(v1)|: the 1-D normalised CCF of the data
    # against template 1 (clipped at 0 when the amplitudes must be >= 0).
    import unicor
    wave, data, g1, g2, template_ext, k = _periodic_case()
    v, ccf = unicor.ccf_vs_velocity(template_ext, data, wave)
    dv = ops.log_grid_step(wave)
    v2 = np.array([-3.0, 0.0, 12.5])
    res = todcor_map(data, g1, g2, dv, v, v2, sigma=1.0, nonneg=nonneg)
    want = np.maximum(ccf, 0.0) if nonneg else np.abs(ccf)
    for j in range(v2.size):
        np.testing.assert_allclose(res["R"][:, j], want, atol=1e-10)
    assert v[np.argmax(res["R"][:, 0])] == pytest.approx(v[k + 7])


def test_output_selects_the_primary_map_and_nothing_else():
    st, v1, v2 = _noisy_case()
    as_ccf = profile_from_stats(st, v1, v2, nonneg=True)
    as_lnl = profile_from_stats(st, v1, v2, nonneg=True, output='loglike')
    assert as_ccf["output"] == 'ccf' and as_ccf["map"] is as_ccf["R"]
    assert as_lnl["output"] == 'loglike' and as_lnl["map"] is as_lnl["logL_profile"]
    assert as_ccf["best"] == as_lnl["best"]
    for key in ("R", "logL_profile", "logL_marginal", "a1", "a2", "chi2"):
        np.testing.assert_array_equal(as_ccf[key], as_lnl[key])


@pytest.mark.parametrize("bad", ['R', 'logL', None])
def test_an_unknown_output_mode_is_refused_by_the_estimator(bad):
    st, v1, v2 = _noisy_case()
    with pytest.raises(ValueError, match="output"):
        profile_from_stats(st, v1, v2, output=bad)


def test_r_is_undefined_without_data_power():
    st, v1, v2 = _noisy_case()
    flat = dict(st, SD=0.0, b1=0.0 * st["b1"], b2=0.0 * st["b2"])
    with pytest.raises(ValueError, match="no power"):
        profile_from_stats(flat, v1, v2)


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
    for k in ("S11", "S22", "SD", "b1", "b2", "A12"):
        assert np.allclose(st8[k], st1[k] / 8.0, rtol=1e-12), k
    # White noise on the analysis grid: one degree of freedom per pixel, less
    # one for the window's free offset -- n / M independent pixels when M
    # analysis pixels share each native one, for either parity of n.
    for n in (len(data_g), len(data_g) - 1):
        d, t1_, t2_ = data_g[:n], g1[:n], g2[:n]
        assert window_stats(d, t1_, t2_, dv, v, v)["nu"] == pytest.approx(n - 1.0)
        assert window_stats(d, t1_, t2_, dv, v, v, oversampling=8.0)["nu"] \
            == pytest.approx(n / 8.0 - 1.0)
    with pytest.raises(ValueError, match="oversampling"):
        window_stats(data_g, g1, g2, dv, v, v, oversampling=0.5)


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
    win = tukey_window(n, alpha=0.1)
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
    # ... and the Fourier phase shift the fit uses (no extension: a circular
    # shift, exact here since the lines are 400 km/s clear of the edges).
    np.testing.assert_allclose(
        model_flux(g1, g2, wave.size, dv, v1, v2, a1, a2, velocity_convention=conv),
        data_g, atol=1e-9)


def test_cubic_shift_of_a_coarse_template_beats_linear():
    # A template sampled at ~4 px per line sigma: linear interpolation of the
    # shifted template errs at the per-cent level, the cubic spline far less.
    t_wave, _ = ops.log_lambda_grid(4990.0, 5070.0, 0.6)
    wave, _ = ops.log_lambda_grid(5000.0, 5060.0, 0.1)
    v = 13.37
    exact = _T1(wave / _factor(v, 'cz'))
    err = {kind: np.max(np.abs(shift_template_flux(t_wave, _T1(t_wave), wave, v, kind=kind) - exact))
           for kind in ('linear', 'cubic')}
    assert err['cubic'] < 0.1 * err['linear']
    np.testing.assert_allclose(
        composite_flux(wave, [(t_wave, _T1(t_wave))], [v], [1.0], kind='cubic'),
        shift_template_flux(t_wave, _T1(t_wave), wave, v, kind='cubic'), atol=1e-15)
    with pytest.raises(ValueError, match="kind"):
        shift_template_flux(t_wave, _T1(t_wave), wave, v, kind='quadratic')


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
             lambda: model_flux(g1, g2, wave.size, dv, 10.0, -10.0, 0.5, 0.5,
                                velocity_convention=bad))
    for call in calls:
        with pytest.raises(ValueError, match='convention'):
            call()


# =============================================================================
# The extended-template window: no wrap-around, a free offset (acceptance)
# =============================================================================
def _line_forest(seed, lam_lo, lam_hi, lines_per_A, sigma_kms):
    """Rest-frame Gaussian-line forest (in velocity) as a function of lambda,
    its lines everywhere in [lam_lo, lam_hi] -- across any window's edges."""
    rng = np.random.default_rng(seed)
    n = int(lines_per_A * (lam_hi - lam_lo))
    centres = rng.uniform(0.0, _C * np.log(lam_hi / lam_lo), n)
    depths = rng.uniform(0.05, 0.6, n)

    def flux(wave):
        u = _C * np.log(np.asarray(wave, float) / lam_lo)
        out = np.ones(u.size)
        for c, d in zip(centres, depths):
            near = np.abs(u - c) < 12.0 * sigma_kms        # exp(-72): nothing beyond
            out[near] *= 1.0 - d * np.exp(-0.5 * ((u[near] - c) / sigma_kms) ** 2)
        return out
    return flux


def _shifted(flux, wave, v):
    """The forest Doppler-shifted by v ('cz'), evaluated exactly at ``wave``."""
    return flux(np.asarray(wave, float) / (1.0 + v / _C))


def test_window_stats_without_an_extension_is_a_circular_shift():
    # e = 0: the template is the window itself, and a shift of k whole pixels
    # reads it rolled by k -- the statistics are those of np.roll.
    rng = np.random.default_rng(4)
    n, dv = 301, 1.0
    data, g1, g2 = rng.normal(size=(3, n))
    k = np.array([-7, 0, 3, 12])
    v = ops.velocity_from_log_shift(k * dv / _C)
    st = window_stats(data, g1, g2, dv, v, v)
    rolled1 = np.array([np.roll(g1, kk) for kk in k])
    rolled2 = np.array([np.roll(g2, kk) for kk in k])
    proj = lambda x: x - x.mean(axis=-1, keepdims=True)     # the free offset
    np.testing.assert_allclose(st["b1"], proj(rolled1) @ proj(data), rtol=1e-10)
    np.testing.assert_allclose(st["S22"], np.sum(proj(rolled2) ** 2, axis=1), rtol=1e-10)
    np.testing.assert_allclose(st["A12"], proj(rolled1) @ proj(rolled2).T,
                               rtol=1e-10, atol=1e-10 * n)
    assert st["SD"] == pytest.approx(np.sum(proj(data) ** 2), rel=1e-12)


def test_a_constant_added_to_the_data_or_a_template_changes_nothing():
    # The window's continuum level is a free parameter: every statistic is
    # projected off a constant, so offsets in the data or the (extended)
    # templates leave the fit unchanged.
    wave, _ = ops.log_lambda_grid(5000.0, 5030.0, 1.0)
    t_wave, _ = ops.log_lambda_grid(4990.0, 5040.0, 0.1)
    T1, T2 = _line_forest(31, 4980.0, 5050.0, 1.0, 4.0), _line_forest(32, 4980.0, 5050.0, 1.0, 5.0)
    flux = 1.0 + 0.7 * (_shifted(T1, wave, 12.3) - 1.0) + 0.3 * (_shifted(T2, wave, -8.1) - 1.0)
    flux = flux + np.random.default_rng(5).normal(0.0, 0.01, wave.size)
    win = ops.prepare_window(wave, flux, t_wave, T1(t_wave), t_wave, T2(t_wave), v_max=30.0)
    v1, v2 = np.arange(5.0, 20.0), np.arange(-15.0, 0.0)
    ref = window_stats(win.data_g, win.g1, win.g2, win.dv, v1, v2)
    moved = window_stats(win.data_g + 0.3, win.g1 - 0.2, win.g2 + 0.45, win.dv, v1, v2)
    for k in ("S11", "S22", "SD", "b1", "b2", "A12", "nu"):
        scale = np.max(np.abs(ref[k]))
        np.testing.assert_allclose(moved[k], ref[k], rtol=0, atol=1e-9 * scale, err_msg=k)


@pytest.mark.parametrize("n_cut", [0, 1])
def test_one_template_correlation_is_unicors_ccf_lag_by_lag(n_cut):
    # With one template, the offset-projected normalised correlation
    # c1 = b1 / sqrt(S11 SD) over the window's observed pixels is UNICOR's
    # extended-template Pearson CCF: same pixels, same template segment at
    # every whole lag. (An even window's template carries one extra pixel on
    # the right, which no lag up to +-extension reads; UNICOR wants it off.)
    import unicor
    wave, _ = ops.log_lambda_grid(5000.0, 5040.0, 1.0)
    wave = wave[:wave.size - n_cut]
    t_wave, _ = ops.log_lambda_grid(4990.0, 5050.0, 0.1)
    T1, T2 = _line_forest(41, 4980.0, 5060.0, 1.0, 4.0), _line_forest(42, 4980.0, 5060.0, 1.0, 5.0)
    flux = 1.0 + 0.7 * (_shifted(T1, wave, 9.4) - 1.0) + 0.3 * (_shifted(T2, wave, -21.7) - 1.0)
    flux = flux + np.random.default_rng(6).normal(0.0, 0.01, wave.size)
    win = ops.prepare_window(wave, flux, t_wave, T1(t_wave), t_wave, T2(t_wave), v_max=40.0)
    n, e = win.wave.size, win.extension
    v, ccf = unicor.ccf_vs_velocity(win.g1[:n + 2 * e], win.data_g, win.wave)
    assert v.size == 2 * win.extension + 1                 # every lag of the extension
    st = window_stats(win.data_g, win.g1, win.g2, win.dv, v, np.array([0.0]))
    c1 = st["b1"] / np.sqrt(st["S11"] * st["SD"])
    np.testing.assert_allclose(c1, ccf, rtol=0, atol=1e-10)
    assert v[np.argmax(c1)] == pytest.approx(9.4, abs=1.0)


@pytest.mark.parametrize("n_cut", [0, 1])                    # either window parity
@pytest.mark.parametrize("width", [50.0, 200.0])
@pytest.mark.parametrize("v1, v2", [(37.31, -52.64), (-59.87, 58.43), (12.345, -3.21),
                                    (0.4, 0.0)])
def test_noise_free_sb2_is_recovered_to_the_cm_per_s_with_lines_across_the_edges(
        width, v1, v2, n_cut):
    # An independent forward model: analytic line forests (R ~ 60 000 lines)
    # evaluated exactly at the shifted wavelengths, with lines straddling both
    # window edges; templates sampled 10x finer than the 0.5 km/s grid. The
    # pipeline prepare_window -> window_stats -> joint_fit (coarse 1 km/s map,
    # 0.1 km/s fine map, degree-6 peak) must land on the truth. The old
    # circular, tapered window was off by up to ~12 m/s on 50 A.
    lo, hi = 5000.0, 5000.0 + width
    T1 = _line_forest(1, lo - 10.0, hi + 10.0, 1.0, 2.1)
    T2 = _line_forest(2, lo - 10.0, hi + 10.0, 0.7, 3.0)
    wave, _ = ops.log_lambda_grid(lo, hi, 0.5)
    wave = wave[:wave.size - n_cut]
    t_wave, _ = ops.log_lambda_grid(lo - 5.0, hi + 5.0, 0.05)
    a1, a2 = 0.7, 0.3
    flux = 1.0 + a1 * (_shifted(T1, wave, v1) - 1.0) + a2 * (_shifted(T2, wave, v2) - 1.0)
    g1 = np.arange(round(v1) - 6.0, round(v1) + 7.0)
    g2 = np.arange(round(v2) - 6.0, round(v2) + 7.0)
    v_max = max(np.abs(g1).max(), np.abs(g2).max()) + 2.0    # + the fine map's reach
    win = ops.prepare_window(wave, flux, t_wave, T1(t_wave), t_wave, T2(t_wave), v_max=v_max)
    assert win.wave.size == wave.size                       # the templates cover it all
    st = window_stats(win.data_g, win.g1, win.g2, win.dv, g1, g2)
    fit = ops.joint_fit([st], None, [(win, {})], g1, g2, nonneg=True, output='ccf')
    assert fit.fine is not None
    err = np.abs(np.array(fit.crlb["peak_refined"]) - [v1, v2]) * 1e3      # m/s
    np.testing.assert_array_less(err, 0.05)
    assert fit.best["a1"] == pytest.approx(a1, abs=1e-3)
    assert fit.best["a2"] == pytest.approx(a2, abs=1e-3)


def _non_quadratic_peak(v1, v2, truth, SD=2e5, r=(0.6, 0.35), s=(1.2, 1.7), c12=0.15,
                        mix=(0.3, -0.2)):
    """logL = -(SD/2)(1 - R^2), with Zucker & Mazeh's R^2 of Gaussian c1, c2
    (in mixed coordinates, so v1 and v2 correlate) and a constant c12.

    Both c's peak at ``truth``, and so does R^2 (dR^2/dc_i > 0 there). Returns
    the map and its analytic covariance ``inv(-Hessian)`` at the maximum.
    """
    V1, V2 = np.meshgrid(v1, v2, indexing="ij")
    J = np.array([[1.0, mix[0]], [mix[1], 1.0]])            # u = J (v - truth)
    u1 = (V1 - truth[0]) + mix[0] * (V2 - truth[1])
    u2 = (V2 - truth[1]) + mix[1] * (V1 - truth[0])
    c1 = r[0] * np.exp(-u1 ** 2 / (4 * s[0] ** 2))
    c2 = r[1] * np.exp(-u2 ** 2 / (4 * s[1] ** 2))
    R2 = (c1 ** 2 - 2 * c1 * c2 * c12 + c2 ** 2) / (1 - c12 ** 2)
    # At the maximum grad c = 0, so Hess R^2 = sum_i (dR^2/dc_i) Hess c_i.
    dR2 = np.array([2 * (r[0] - c12 * r[1]), 2 * (r[1] - c12 * r[0])]) / (1 - c12 ** 2)
    hess_u = np.diag(dR2 * -np.array(r) / (2 * np.square(s)))
    hessian = 0.5 * SD * J.T @ hess_u @ J
    return -0.5 * SD * (1 - R2), np.linalg.inv(-hessian)


def test_crlb_degree_six_peak_reads_a_non_quadratic_peak():
    step, truth = 0.1, (0.0373, -0.0521)                    # off the nodes
    v1 = np.arange(-20, 21) * step
    v2 = np.arange(-20, 21) * step - 0.3
    logL, cov = _non_quadratic_peak(v1, v2, truth)
    cr = crlb_from_map(logL, v1, v2)
    assert not cr["on_edge"]
    np.testing.assert_array_less(np.abs(np.subtract(cr["peak_refined"], truth)), 1e-4 * step)
    np.testing.assert_allclose(cr["cov"], cov, rtol=1e-3)
    assert cr["sigma_v1"] == pytest.approx(np.sqrt(cov[0, 0]), rel=1e-3)
    assert cr["sigma_v2"] == pytest.approx(np.sqrt(cov[1, 1]), rel=1e-3)
    assert cr["rho"] == pytest.approx(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]), rel=1e-3)


def test_crlb_degree_six_peak_beyond_the_map_edge_is_nan():
    step = 0.1
    v = np.arange(-20, 21) * step
    logL, _ = _non_quadratic_peak(v, v, (2.6, 0.0))         # v1 truth beyond +2.0
    cr = crlb_from_map(logL, v, v)
    assert cr["on_edge"]
    assert np.isnan(cr["sigma_v1"]) and np.isnan(cr["sigma_v2"]) and np.isnan(cr["cov"]).all()
    # Grid-limited, not extrapolated (v2 at the ridge's node along v1 = 2).
    assert cr["peak_refined"] == cr["peak"] and cr["peak"][0] == 2.0


def test_crlb_refuses_a_maximum_more_than_a_step_away_or_a_saddle():
    v = np.arange(-6.0, 7.0)
    V1, V2 = np.meshgrid(v, v, indexing="ij")
    # A one-node spike on a broad peak centred 3 steps away: the argmax is the
    # spike, the polynomial's maximum ~3 steps off -- extrapolation.
    spike = -0.5 * ((V1 - 3.0) ** 2 + V2 ** 2) / 100.0 + 0.06 * ((V1 == 0) & (V2 == 0))
    # A double well in v2 whose argmax node sits between the wells: the
    # fitted curvature there is positive along v2.
    saddle = -V1 ** 2 + 0.5 * V2 ** 2 - 0.1 * V2 ** 4 + 0.5 * (V2 == 0)
    for logL in (spike, saddle):
        cr = crlb_from_map(logL, v, v)
        assert cr["peak"] == (0.0, 0.0) and not cr["on_edge"]
        assert np.isnan(cr["sigma_v1"]) and np.isnan(cr["sigma_v2"])
        assert cr["peak_refined"] == cr["peak"]


def test_model_flux_is_the_composite_of_the_extended_templates():
    # The fit's own model: at whole-pixel velocities it is exactly the
    # extended templates read k pixels off centre; at any velocity it is a
    # composite the fit matches with chi2 = 0 and the true amplitudes.
    wave, _ = ops.log_lambda_grid(5000.0, 5030.0, 1.0)
    t_wave, _ = ops.log_lambda_grid(4990.0, 5040.0, 0.1)
    T1, T2 = _line_forest(51, 4980.0, 5050.0, 1.0, 4.0), _line_forest(52, 4980.0, 5050.0, 1.0, 5.0)
    win = ops.prepare_window(wave, np.ones(wave.size), t_wave, T1(t_wave), t_wave, T2(t_wave),
                             v_max=30.0)
    n, e = win.wave.size, win.extension
    k1, k2 = 17, -23
    v1, v2 = (ops.velocity_from_log_shift(k * win.dv / _C) for k in (k1, k2))
    want = 0.7 * win.g1[e - k1: e - k1 + n] + 0.3 * win.g2[e - k2: e - k2 + n]
    np.testing.assert_allclose(model_flux(win.g1, win.g2, n, win.dv, v1, v2, 0.7, 0.3), want,
                               rtol=0, atol=1e-12)
    for v1, v2 in ((v1, v2), (13.37, -21.05)):
        data = model_flux(win.g1, win.g2, n, win.dv, v1, v2, 0.7, 0.3) + 0.42
        st = window_stats(data, win.g1, win.g2, win.dv, np.array([v1]), np.array([v2]),
                          sigma=0.01)
        best = profile_from_stats(st, np.array([v1]), np.array([v2]))["best"]
        assert best["chi2"] == pytest.approx(0.0, abs=1e-12 * st["SD"])
        assert (best["a1"], best["a2"]) == (pytest.approx(0.7, abs=1e-10),
                                            pytest.approx(0.3, abs=1e-10))
