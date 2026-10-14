"""The per-window atoms of :mod:`todcor.estimator`.

``prepare_window`` (continuum, template resampling, taper), ``refine_grid``
(the fine local map), ``joint_fit`` (pool, coarse map, fine map, CRLB and the
evaluation at the refined peak) and ``window_noise`` (per-window noise from
the joint chi2) -- what SPARTA's ``CCF2d`` composes, on plain arrays. Each is
checked against the lower-level functions it chains, with ``np.array_equal``
where it is a pure composition.

Self-contained: no external data, numpy only. Velocities in km/s,
wavelengths in Angstrom.
"""

import dataclasses
import pathlib
import re
import warnings

import numpy as np
import pytest

import todcor
from todcor import (
    TodcorFit, TodcorWindow, chi2_from_stats, combine_stats, composite_flux, crlb_from_map,
    joint_fit, log_grid_step, log_lambda_grid, prepare_window, profile_from_stats,
    refine_grid, resample, tukey_window, window_noise, window_stats,
)
from todcor import estimator

V_TRUE, A_TRUE = (60.0, -95.0), (0.7, 0.4)
V1 = np.arange(30.0, 91.0, 2.0)
V2 = np.arange(-125.0, -64.0, 2.0)


def _gauss(wave, center, depth, fwhm):
    sig = fwhm / 2.3548200450309493
    return 1.0 + depth * np.exp(-0.5 * ((wave - center) / sig) ** 2)


def _templates():
    """An absorption-line primary and an emission-line secondary (Angstrom, flux)."""
    wave = np.linspace(4000.0, 5000.0, 8000)
    t1 = (_gauss(wave, 4101.7, -0.5, 1.5) * _gauss(wave, 4340.5, -0.45, 1.5)
          * _gauss(wave, 4471.5, -0.25, 1.2))
    t2 = _gauss(wave, 4058.0, +0.6, 2.0) * _gauss(wave, 4685.7, +0.8, 3.0)
    return (wave, t1), (wave, t2)


def _observed(noise=(0.01, 0.03), seed=3):
    """Two 1 km/s log-lambda orders of a noisy SB2 at V_TRUE."""
    tmpl = _templates()
    rng = np.random.default_rng(seed)
    orders = []
    for (lo, hi), s in zip(((4030.0, 4380.0), (4400.0, 4720.0)), noise):
        grid, _ = log_lambda_grid(lo, hi, 1.0)
        flux = composite_flux(grid, list(tmpl), list(V_TRUE), list(A_TRUE))
        orders.append((grid, flux + rng.normal(0.0, s, grid.size)))
    return orders, tmpl


def _prepared(sigma=(1.0, 1.0), noise=(0.01, 0.03)):
    """The prepared windows, their window_stats kwargs and coarse statistics."""
    orders, ((w1, f1), (w2, f2)) = _observed(noise)
    prepared, stats = [], []
    for (grid, flux), s in zip(orders, sigma):
        win = prepare_window(grid, flux, w1, f1, w2, f2)
        kw = dict(sigma=s, noise_psd=win.taper_power * np.ones(grid.size // 2))
        prepared.append((win, kw))
        stats.append(window_stats(win.data_g, win.g1, win.g2, win.dv, V1, V2, **kw))
    return prepared, stats


def _assert_same_dict(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], dict):
            _assert_same_dict(a[k], b[k])
        elif isinstance(a[k], str):
            assert a[k] == b[k]
        else:
            assert np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=True), k


# =============================================================================
# The namespace
# =============================================================================
def test_the_atoms_are_exported_and_the_version_is_the_packages():
    for name in ("TodcorWindow", "TodcorFit", "prepare_window", "refine_grid", "joint_fit",
                 "window_noise"):
        assert name in todcor.__all__ and getattr(todcor, name) is getattr(estimator, name)
    pyproject = (pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert re.search(r'^version = "(.+)"', pyproject, flags=re.M).group(1) \
        == todcor.__version__ == "0.2.0"


# =============================================================================
# prepare_window
# =============================================================================
def test_prepare_window_removes_the_median_resamples_and_tapers():
    [(grid, flux), _], ((w1, f1), (w2, f2)) = _observed()
    win = prepare_window(grid, flux, w1, f1, w2, f2)
    taper = tukey_window(grid.size, alpha=0.1)
    r1, r2 = resample(w1, f1, grid, fill=1.0), resample(w2, f2, grid, fill=1.0)
    assert isinstance(win, TodcorWindow)
    assert np.array_equal(win.wave, grid)
    assert win.dv == log_grid_step(grid)
    assert np.array_equal(win.data_g, (flux - np.median(flux)) * taper)
    assert np.array_equal(win.g1, (r1 - np.median(r1)) * taper)
    assert np.array_equal(win.g2, (r2 - np.median(r2)) * taper)
    assert win.taper_power == float(np.mean(taper ** 2))


def test_prepare_window_with_a_fixed_continuum_and_no_taper():
    [(grid, flux), _], ((w1, f1), (w2, f2)) = _observed()
    win = prepare_window(grid, flux, w1, f1, w2, f2, continuum=1.0, alpha=0)
    assert np.array_equal(win.data_g, flux - 1.0)
    assert np.array_equal(win.g1, resample(w1, f1, grid, fill=1.0) - 1.0)
    assert win.taper_power == 1.0


def test_prepare_window_fills_uncovered_template_pixels_with_the_continuum():
    # A template covering only the blue half contributes no lines (and no
    # clamped edge value) on the red half.
    [(grid, flux), _], ((w1, f1), (w2, f2)) = _observed()
    half = w1 < 4200.0
    win = prepare_window(grid, flux, w1[half], f1[half] * 0.9, w2, f2, continuum=1.0, alpha=0)
    assert np.all(win.g1[grid > w1[half][-1]] == 0.0)
    assert np.all(win.g1[grid < 4190.0] < 0.0)


@pytest.mark.parametrize("name", ["spectrum", "template1", "template2"])
def test_prepare_window_refuses_non_finite_flux(name):
    [(grid, flux), _], ((w1, f1), (w2, f2)) = _observed()
    arrays = {"spectrum": flux.copy(), "template1": f1.copy(), "template2": f2.copy()}
    arrays[name][len(arrays[name]) // 3] = np.nan
    with pytest.raises(ValueError, match=f"^{name} flux contains NaN/inf; mask or interpolate"):
        prepare_window(grid, arrays["spectrum"], w1, arrays["template1"], w2,
                       arrays["template2"])


def test_prepare_window_needs_one_log_lambda_grid():
    [(grid, flux), _], ((w1, f1), (w2, f2)) = _observed()
    with pytest.raises(ValueError, match="uniform"):
        prepare_window(np.linspace(grid[0], grid[-1], grid.size), flux, w1, f1, w2, f2)
    with pytest.raises(ValueError, match="one flux per wavelength"):
        prepare_window(grid, flux[:-1], w1, f1, w2, f2)


def test_a_prepared_window_is_frozen():
    [(grid, flux), _], ((w1, f1), (w2, f2)) = _observed()
    win = prepare_window(grid, flux, w1, f1, w2, f2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        win.dv = 2.0


# =============================================================================
# refine_grid
# =============================================================================
def test_refine_grid_spans_two_local_steps_at_a_tenth_of_the_step():
    v = np.arange(-10.0, 11.0, 2.0)
    f1, f2 = refine_grid(v, v - 50.0, 5, 3)
    nodes = np.arange(-20, 21) / 10
    assert np.array_equal(f1, v[5] + 2.0 * nodes)
    assert np.array_equal(f2, v[3] - 50.0 + 2.0 * nodes)
    assert (f1[0], f1[-1]) == (v[5] - 4.0, v[5] + 4.0)


def test_refine_grid_takes_the_local_step_of_a_non_uniform_grid():
    v1, v2 = np.array([0.0, 1.0, 3.0, 7.0]), np.arange(5.0)
    f1, f2 = refine_grid(v1, v2, 2, 1, span=1, factor=4)
    nodes = np.arange(-4, 5) / 4
    assert np.array_equal(f1, 3.0 + 3.0 * nodes)
    assert np.array_equal(f2, 1.0 + 1.0 * nodes)


@pytest.mark.parametrize("i, j", [(0, 3), (9, 3), (3, 0), (3, 9)])
def test_refine_grid_needs_an_interior_node(i, j):
    v = np.arange(10.0)
    with pytest.raises(ValueError, match="interior"):
        refine_grid(v, v, i, j)


# =============================================================================
# joint_fit
# =============================================================================
@pytest.mark.parametrize("output", ["ccf", "loglike"])
def test_joint_fit_is_the_pooled_coarse_fine_and_peak_composition(output):
    prepared, stats = _prepared()
    weights = np.array([2.0, 0.5])
    fit = joint_fit(stats, weights, prepared, V1, V2, nonneg=True, output=output)
    assert isinstance(fit, TodcorFit)

    pooled = combine_stats(stats, weights)
    _assert_same_dict(fit.pooled, pooled)
    profile = profile_from_stats(pooled, V1, V2, nonneg=True, output=output)
    _assert_same_dict(fit.profile, profile)
    assert fit.profile["map"] is fit.profile["R" if output == "ccf" else "logL_profile"]
    _assert_same_dict(fit.coarse, crlb_from_map(profile["logL_profile"], V1, V2, half=3))

    i, j = np.unravel_index(np.argmax(profile["logL_profile"]), profile["logL_profile"].shape)
    f1, f2 = refine_grid(V1, V2, i, j)
    fine = [window_stats(w.data_g, w.g1, w.g2, w.dv, f1, f2, **kw) for w, kw in prepared]
    fine = profile_from_stats(combine_stats(fine, weights), f1, f2, nonneg=True, output=output)
    _assert_same_dict(fit.fine, fine)
    crlb = crlb_from_map(fine["logL_profile"], f1, f2, half=3)
    _assert_same_dict(fit.crlb, crlb)

    p1, p2 = np.array([crlb["peak_refined"][0]]), np.array([crlb["peak_refined"][1]])
    at_peak = [window_stats(w.data_g, w.g1, w.g2, w.dv, p1, p2, **kw) for w, kw in prepared]
    for got, want in zip(fit.at_peak, at_peak):
        _assert_same_dict(got, want)
    best = profile_from_stats(combine_stats(at_peak, weights), p1, p2, nonneg=True)["best"]
    assert fit.best == best
    shares = [chi2_from_stats(st, best["a1"], best["a2"]).item() for st in at_peak]
    assert np.array_equal(fit.window_chi2, shares)
    assert np.sum(fit.window_chi2 * weights) == pytest.approx(best["chi2"], rel=1e-9)

    assert np.allclose(crlb["peak_refined"], V_TRUE, atol=0.5)
    assert best["a1"] == pytest.approx(A_TRUE[0], abs=0.05)


def test_joint_fit_output_modes_differ_only_in_the_primary_map():
    prepared, stats = _prepared()
    a = joint_fit(stats, None, prepared, V1, V2, nonneg=True, output="ccf")
    b = joint_fit(stats, None, prepared, V1, V2, nonneg=True, output="loglike")
    assert a.best == b.best and a.crlb["peak_refined"] == b.crlb["peak_refined"]
    assert np.array_equal(a.fine["logL_profile"], b.fine["logL_profile"])


def test_joint_fit_weights_are_the_statistics_at_that_noise():
    # The statistics are linear in 1/sigma^2: weighting unit-noise windows by
    # 1/sigma_i^2 is the fit at sigma_i, to round-off.
    sigma = (0.01, 0.03)
    unit, stats_unit = _prepared()
    at_sigma, stats_sigma = _prepared(sigma=sigma)
    a = joint_fit(stats_unit, 1.0 / np.square(sigma), unit, V1, V2, nonneg=True, output="ccf")
    b = joint_fit(stats_sigma, None, at_sigma, V1, V2, nonneg=True, output="ccf")
    np.testing.assert_allclose(a.crlb["peak_refined"], b.crlb["peak_refined"], rtol=1e-9)
    # The off-diagonal is round-off (~1e-9 of the diagonal: the components'
    # lines do not overlap), so it gets an absolute floor.
    np.testing.assert_allclose(a.crlb["cov"], b.crlb["cov"], rtol=1e-6,
                               atol=1e-6 * np.abs(b.crlb["cov"]).max())
    assert a.best["chi2"] == pytest.approx(b.best["chi2"], rel=1e-9)


def test_joint_fit_reuses_the_fine_statistics_it_cached(monkeypatch):
    prepared, stats = _prepared()
    calls = []
    real = estimator.window_stats

    def spy(*args, **kwargs):
        calls.append(len(args[4]))
        return real(*args, **kwargs)

    monkeypatch.setattr(estimator, "window_stats", spy)
    cache = {}
    first = joint_fit(stats, None, prepared, V1, V2, nonneg=True, output="ccf",
                      fine_cache=cache)
    assert calls == [41, 41, 1, 1] and len(cache) == 1
    calls.clear()
    second = joint_fit(stats, [1.0, 0.2], prepared, V1, V2, nonneg=True, output="ccf",
                       fine_cache=cache)
    assert calls == [1, 1] and len(cache) == 1
    assert second.fine["v1_grid"][20] == first.fine["v1_grid"][20]
    # Without a cache every call evaluates its own fine map.
    calls.clear()
    fresh = joint_fit(stats, [1.0, 0.2], prepared, V1, V2, nonneg=True, output="ccf")
    assert calls == [41, 41, 1, 1]
    _assert_same_dict(fresh.fine, second.fine)


def test_joint_fit_does_not_refine_a_peak_on_the_grid_edge():
    prepared, _ = _prepared()
    v1 = np.arange(30.0, 51.0, 2.0)                       # truth at 60 km/s
    stats = [window_stats(w.data_g, w.g1, w.g2, w.dv, v1, V2, **kw) for w, kw in prepared]
    fit = joint_fit(stats, None, prepared, v1, V2, nonneg=True, output="ccf")
    assert fit.coarse["on_edge"] and fit.fine is None and fit.crlb is fit.coarse
    assert np.isnan(fit.crlb["sigma_v1"]) and fit.crlb["peak_refined"][0] == 50.0
    assert fit.at_peak[0]["b1"].shape == (1,)


def test_joint_fit_needs_one_prepared_window_per_statistics():
    prepared, stats = _prepared()
    with pytest.raises(ValueError, match="one prepared window per"):
        joint_fit(stats, None, prepared[:1], V1, V2, nonneg=True, output="ccf")


def test_a_fit_is_frozen():
    prepared, stats = _prepared()
    fit = joint_fit(stats, None, prepared, V1, V2, nonneg=True, output="ccf")
    with pytest.raises(dataclasses.FrozenInstanceError):
        fit.best = {}


# =============================================================================
# window_noise
# =============================================================================
def test_window_noise_of_one_window_is_the_root_reduced_chi2():
    sigma, fallback = window_noise([90.0], [100.0])
    assert sigma[0] == np.sqrt(90.0 / 96.0) and not fallback[0]


def test_window_noise_charges_the_four_dof_in_proportion_to_nu():
    chi2, nu = np.array([500.0, 2400.0]), np.array([1000.0, 3000.0])
    sigma, fallback = window_noise(chi2, nu)
    np.testing.assert_array_equal(sigma, np.sqrt(chi2 / (nu * (1.0 - 4.0 / 4000.0))))
    assert not fallback.any()


@pytest.mark.parametrize("chi2, nu", [([500.0, 20.0], [1000.0, 30.0]),     # too few dof
                                      ([500.0, 0.0], [1000.0, 1000.0]),     # no residual
                                      ([500.0, np.nan], [1000.0, 1000.0])])
def test_window_noise_falls_back_to_the_common_estimate_without_warning(chi2, nu):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sigma, fallback = window_noise(chi2, nu)
    dof = np.asarray(nu) * (1.0 - 4.0 / np.sum(nu))
    assert fallback.tolist() == [False, True]
    assert sigma[1] == sigma[0] == np.sqrt(500.0 / dof[0])


def test_window_noise_pools_every_window_if_none_measures_its_own():
    chi2, nu = np.array([25.0, 12.0]), np.array([30.0, 20.0])
    sigma, fallback = window_noise(chi2, nu)
    dof = nu * (1.0 - 4.0 / 50.0)
    assert fallback.all()
    np.testing.assert_array_equal(sigma, np.sqrt(chi2.sum() / dof.sum()))
    sigma, fallback = window_noise(chi2, nu, min_dof=10)
    assert not fallback.any()


@pytest.mark.parametrize("chi2", [[0.0, 0.0], [np.nan, np.nan], [-1.0, 0.0]])
def test_window_noise_refuses_when_no_noise_can_be_estimated(chi2):
    with pytest.raises(ValueError, match="cannot estimate the noise"):
        window_noise(chi2, [1000.0, 1000.0])


def test_window_noise_needs_one_chi2_per_nu():
    with pytest.raises(ValueError, match="one chi2 per window"):
        window_noise([1.0, 2.0], [100.0])
