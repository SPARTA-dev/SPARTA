"""The per-window atoms of :mod:`todcor.estimator`.

``prepare_window`` (observed pixels, templates extended past them),
``refine_grid``
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
    C_KMS, TodcorFit, TodcorWindow, chi2_from_stats, combine_stats, composite_flux,
    crlb_from_map, joint_fit, log_grid_step, log_lambda_grid, prepare_window,
    profile_from_stats, profile_per_window, refine_grid, resample, window_noise,
    window_stats,
)
from todcor import estimator
from unicor.ccf import extend_log_grid

V_TRUE, A_TRUE = (60.0, -95.0), (0.7, 0.4)
V1 = np.arange(30.0, 91.0, 2.0)
V2 = np.arange(-125.0, -64.0, 2.0)
V_MAX = 130.0          # the grids' largest |v| plus the fine map's reach (km/s)


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
        win = prepare_window(grid, flux, w1, f1, w2, f2, v_max=V_MAX)
        kw = dict(sigma=s)
        prepared.append((win, kw))
        stats.append(window_stats(win.data_g, win.g1, win.g2, win.dv, V1, V2, **kw))
    return prepared, stats


def _extended_grid(wave, e):
    """The window's grid continued by ``e`` pixels on the left and ``e`` (odd
    window) or ``e + 1`` (even window) on the right: odd total length."""
    ext = extend_log_grid(wave, e + 1)
    return ext[1:] if wave.size % 2 == 0 else ext[1:-1]


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
def test_prepare_window_keeps_the_observed_pixels_and_extends_the_templates():
    # Templates covering the window and +-V_MAX: every observed pixel is kept,
    # untapered (only its median removed), and each template is resampled onto
    # the window's grid continued by `extension` pixels on the left and as
    # many (odd window) or one more (even window) on the right.
    [(grid, flux), _], ((w1, f1), (w2, f2)) = _observed()
    win = prepare_window(grid, flux, w1, f1, w2, f2, v_max=V_MAX)
    assert isinstance(win, TodcorWindow)
    assert np.array_equal(win.wave, grid)
    assert win.dv == log_grid_step(grid)
    assert np.array_equal(win.data_g, flux - np.median(flux))
    # The extension holds the largest shift, plus the ramp.
    assert win.extension * win.dv > C_KMS * np.log1p(V_MAX / C_KMS) + 20 * win.dv
    assert grid.size % 2 == 0
    ext = _extended_grid(grid, win.extension)
    for g, (w, f) in ((win.g1, (w1, f1)), (win.g2, (w2, f2))):
        assert g.size == grid.size + 2 * win.extension + 1
        inner = slice(20, -20)                               # inside the ramps
        offset = resample(w, f, ext)[inner] - g[inner]
        np.testing.assert_allclose(offset, offset[0], rtol=0, atol=1e-12)


@pytest.mark.parametrize("n_cut", [0, 1])
def test_the_extended_templates_have_odd_length(n_cut):
    # Odd length (prepare_window's stated design): the Fourier shift then has
    # no Nyquist mode, whatever the parity of the window.
    [(grid, flux), _], ((w1, f1), (w2, f2)) = _observed()
    n_pix = grid.size - n_cut
    win = prepare_window(grid[:n_pix], flux[:n_pix], w1, f1, w2, f2, v_max=V_MAX)
    assert win.g1.size % 2 == 1 and win.g2.size % 2 == 1, (n_pix, win.g1.size)
    # `extension` is the left one, the offset the fit reads the window from.
    assert (win.g1.size - n_pix) // 2 == win.extension


def test_the_extended_templates_ramp_to_zero():
    # The outermost `ramp` pixels fall smoothly (raised cosine) to zero,
    # keeping the shift's wrap-around junction away from every pixel the fit
    # reads. (Sloped templates, so the ends are not trivially at the median.)
    [(grid, flux), _], ((w1, f1), (w2, f2)) = _observed()
    f1, f2 = f1 * (1.0 + 1e-3 * (w1 - 4200.0)), f2 * (1.0 - 2e-3 * (w2 - 4200.0))
    for n_pix in (grid.size, grid.size - 1):
        for ramp in (2, 20, 35):
            win = prepare_window(grid[:n_pix], flux[:n_pix], w1, f1, w2, f2, v_max=V_MAX,
                                 ramp=ramp)
            ext = _extended_grid(win.wave, win.extension)
            for g, (w, f) in ((win.g1, (w1, f1)), (win.g2, (w2, f2))):
                assert g[0] == 0.0 and g[-1] == 0.0
                plain = resample(w, f, ext)
                plain = plain - (plain[ramp:-ramp] - g[ramp:-ramp])[0]
                edge = 0.5 * (1.0 - np.cos(np.pi * np.arange(ramp) / ramp))     # 0 -> 1
                np.testing.assert_allclose(g[:ramp], plain[:ramp] * edge, rtol=0, atol=1e-12)
                np.testing.assert_allclose(g[::-1][:ramp], plain[::-1][:ramp] * edge,
                                           rtol=0, atol=1e-12)


def test_window_stats_refuses_a_shift_beyond_the_extension():
    # A trial velocity past v_max would read the ramp or the wrapped far end
    # of the extended templates; it is refused, not silently evaluated.
    [(grid, flux), _], ((w1, f1), (w2, f2)) = _observed()
    win = prepare_window(grid, flux, w1, f1, w2, f2, v_max=V_MAX)
    reach = win.extension * win.dv
    ok = np.array([-0.9 * reach, 0.9 * reach])
    estimator.window_stats(win.data_g, win.g1, win.g2, win.dv, ok, ok)
    with pytest.raises(ValueError, match="v2_grid.*beyond"):
        estimator.window_stats(win.data_g, win.g1, win.g2, win.dv, ok, ok * 1.2)


def test_prepare_window_drops_observed_pixels_the_templates_do_not_cover():
    # Template 1 ends at 4200 A and template 2 starts at 4100 A: only the
    # observed pixels whose templates reach +-V_MAX beyond them are kept --
    # never filled with a continuum -- and they are a contiguous run.
    [(grid, flux), _], ((w1, f1), (w2, f2)) = _observed()
    red, blue = w1 < 4200.0, w2 > 4100.0
    win = prepare_window(grid, flux, w1[red], f1[red], w2[blue], f2[blue], v_max=V_MAX)
    doppler = 1.0 + V_MAX / C_KMS
    first = np.flatnonzero(grid == win.wave[0])[0]
    assert np.array_equal(win.wave, grid[first:first + win.wave.size])
    assert np.array_equal(win.data_g, flux[first:first + win.wave.size]
                          - np.median(flux[first:first + win.wave.size]))
    assert win.wave[0] / doppler >= w2[blue][0] and win.wave[-1] * doppler <= w1[red][-1]
    # ...and no more than a couple of pixels beyond what +-V_MAX needs.
    assert grid[first - 3] / doppler < w2[blue][0]
    assert grid[first + win.wave.size + 2] * doppler > w1[red][-1]
    for g in (win.g1, win.g2):
        assert g.size == win.wave.size + 2 * win.extension + (win.wave.size % 2 == 0)
    # Templates that reach no observed pixel across +-V_MAX are refused.
    narrow = (w1 > 4150.0) & (w1 < 4152.0)
    with pytest.raises(ValueError, match="does not cover any pixel"):
        prepare_window(grid, flux, w1[narrow], f1[narrow], w2, f2, v_max=V_MAX)


@pytest.mark.parametrize("name", ["spectrum", "template1", "template2"])
def test_prepare_window_refuses_non_finite_flux(name):
    [(grid, flux), _], ((w1, f1), (w2, f2)) = _observed()
    arrays = {"spectrum": flux.copy(), "template1": f1.copy(), "template2": f2.copy()}
    arrays[name][len(arrays[name]) // 3] = np.nan
    with pytest.raises(ValueError, match=f"^{name} flux contains NaN/inf; mask or interpolate"):
        prepare_window(grid, arrays["spectrum"], w1, arrays["template1"], w2,
                       arrays["template2"], v_max=V_MAX)


def test_prepare_window_needs_one_log_lambda_grid():
    [(grid, flux), _], ((w1, f1), (w2, f2)) = _observed()
    with pytest.raises(ValueError, match="uniform"):
        prepare_window(np.linspace(grid[0], grid[-1], grid.size), flux, w1, f1, w2, f2,
                       v_max=V_MAX)
    with pytest.raises(ValueError, match="one flux per wavelength"):
        prepare_window(grid, flux[:-1], w1, f1, w2, f2, v_max=V_MAX)


@pytest.mark.parametrize("kw, match", [(dict(v_max=0.0), "v_max"), (dict(v_max=-5.0), "v_max"),
                                       (dict(v_max=np.nan), "v_max"),
                                       (dict(v_max=V_MAX, ramp=1), "ramp")])
def test_prepare_window_needs_a_positive_v_max_and_a_ramp(kw, match):
    [(grid, flux), _], ((w1, f1), (w2, f2)) = _observed()
    with pytest.raises(ValueError, match=match):
        prepare_window(grid, flux, w1, f1, w2, f2, **kw)
    with pytest.raises(TypeError):
        prepare_window(grid, flux, w1, f1, w2, f2)          # v_max has no default


def test_a_prepared_window_is_frozen():
    [(grid, flux), _], ((w1, f1), (w2, f2)) = _observed()
    win = prepare_window(grid, flux, w1, f1, w2, f2, v_max=V_MAX)
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
    _assert_same_dict(fit.coarse, crlb_from_map(profile["logL_profile"], V1, V2, half=4))

    i, j = np.unravel_index(np.argmax(profile["logL_profile"]), profile["logL_profile"].shape)
    f1, f2 = refine_grid(V1, V2, i, j)
    fine = [window_stats(w.data_g, w.g1, w.g2, w.dv, f1, f2, **kw) for w, kw in prepared]
    fine = profile_from_stats(combine_stats(fine, weights), f1, f2, nonneg=True, output=output)
    _assert_same_dict(fit.fine, fine)
    crlb = crlb_from_map(fine["logL_profile"], f1, f2, half=4)
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


def test_window_noise_per_order_charges_each_window_its_own_amplitudes():
    # Per-order amplitudes: the two shared velocities are charged in
    # proportion to nu_i, and each window pays its own two amplitudes.
    chi2, nu = np.array([500.0, 2400.0]), np.array([1000.0, 3000.0])
    sigma, fallback = window_noise(chi2, nu, amplitudes="per_order")
    dof = nu * (1.0 - 2.0 / 4000.0) - 2.0
    np.testing.assert_array_equal(sigma, np.sqrt(chi2 / dof))
    assert not fallback.any()
    # The dof sum to sum(nu) - 2 - 2 n_windows, the per_order chi2_red's.
    assert dof.sum() == pytest.approx(nu.sum() - 2.0 - 2.0 * nu.size, rel=1e-12)
    with pytest.raises(ValueError, match="amplitudes"):
        window_noise(chi2, nu, amplitudes="per_window")


# =============================================================================
# Per-order amplitudes (amplitudes='per_order')
# =============================================================================
_C = C_KMS
_WINDOWS = ((5000.0, 5030.0), (5060.0, 5090.0))
_V_PO = (23.37, -31.62)
_G1_PO, _G2_PO = np.arange(17.0, 30.0), np.arange(-38.0, -25.0)


def _forest(seed, sigma_kms, lam_lo=4990.0, lam_hi=5100.0, lines_per_A=1.0):
    """Rest-frame Gaussian-line forest (in velocity) as a function of lambda."""
    rng = np.random.default_rng(seed)
    n = int(lines_per_A * (lam_hi - lam_lo))
    centres = rng.uniform(0.0, _C * np.log(lam_hi / lam_lo), n)
    depths = rng.uniform(0.05, 0.6, n)

    def flux(wave):
        u = _C * np.log(np.asarray(wave, float) / lam_lo)
        return np.prod(1.0 - depths * np.exp(-0.5 * ((u[:, None] - centres) / sigma_kms) ** 2),
                       axis=1)
    return flux


_F1, _F2 = _forest(5, 4.0), _forest(6, 5.0)


def _per_order_case(amps, sigma, n_real, seed):
    """Two windows of an SB2 at _V_PO with amplitudes ``amps[i]`` in window i,
    white noise ``sigma`` on their 1 km/s pixels; yields each realisation's
    (prepared, stats) at the true sigma."""
    t_wave, _ = log_lambda_grid(4995.0, 5095.0, 0.1)
    t1, t2 = _F1(t_wave), _F2(t_wave)
    waves = [log_lambda_grid(lo, hi, 1.0)[0] for lo, hi in _WINDOWS]
    clean = [1.0 + a1 * (_F1(w / (1.0 + _V_PO[0] / _C)) - 1.0)
             + a2 * (_F2(w / (1.0 + _V_PO[1] / _C)) - 1.0) for w, (a1, a2) in zip(waves, amps)]
    rng = np.random.default_rng(seed)
    for _ in range(n_real):
        prepared, stats = [], []
        for w, f in zip(waves, clean):
            win = prepare_window(w, f + rng.normal(0.0, sigma, w.size), t_wave, t1, t_wave, t2,
                                 v_max=40.0)
            kw = dict(sigma=sigma)
            prepared.append((win, kw))
            stats.append(window_stats(win.data_g, win.g1, win.g2, win.dv, _G1_PO, _G2_PO, **kw))
        yield prepared, stats


def _fit_both(prepared, stats):
    return {mode: joint_fit(stats, None, prepared, _G1_PO, _G2_PO, nonneg=True, output="ccf",
                            amplitudes=mode)
            for mode in ("per_order", "shared")}


def test_per_order_amplitudes_recover_each_windows_light_ratio():
    # The light ratio a2/a1 is 0.25 in one window and 0.6 in the other. Each
    # window's own amplitudes are recovered (Monte Carlo: unbiased within 3
    # sigma/sqrt(N), every realisation within 4 sigma) with chi2_red ~ 1; one
    # shared pair cannot fit both, and its chi2_red says so.
    amps = np.array([[0.7, 0.175], [0.5, 0.3]])
    a, chi2_red = [], {"per_order": [], "shared": []}
    for prepared, stats in _per_order_case(amps, 0.01, n_real=20, seed=1):
        fits = _fit_both(prepared, stats)
        po = fits["per_order"]
        assert po.best["a1"].shape == po.best["a2"].shape == (2,)   # one per window
        a.append(np.column_stack([po.best["a1"], po.best["a2"]]))
        for mode, fit in fits.items():
            chi2_red[mode].append(fit.best["chi2_red"])
    a = np.array(a)                                          # (N, window, component)
    mean, std = a.mean(axis=0), a.std(axis=0, ddof=1)
    assert np.all(np.abs(mean - amps) < 3.0 * std / np.sqrt(len(a))), (mean, std)
    assert np.all(np.abs(a - amps) < 4.0 * std)
    assert np.mean(chi2_red["per_order"]) == pytest.approx(1.0, abs=0.03)
    assert np.min(chi2_red["shared"]) > 2.0


def test_per_order_and_shared_agree_on_a_constant_light_ratio():
    # The same amplitudes in both windows: the per-order fit spends four free
    # parameters for nothing, and its velocities match the shared fit's to a
    # small fraction of the error.
    amps = np.array([[0.6, 0.3], [0.6, 0.3]])
    for prepared, stats in _per_order_case(amps, 0.01, n_real=4, seed=2):
        fits = _fit_both(prepared, stats)
        po, sh = fits["per_order"].crlb, fits["shared"].crlb
        err = np.array([sh["sigma_v1"], sh["sigma_v2"]])
        assert np.all(np.abs(np.subtract(po["peak_refined"], sh["peak_refined"])) < 0.15 * err)
        np.testing.assert_allclose([po["sigma_v1"], po["sigma_v2"]], err, rtol=0.05)
        assert fits["per_order"].best["chi2_red"] == pytest.approx(
            fits["shared"].best["chi2_red"], abs=0.01)


def test_joint_fit_per_order_is_profile_per_window_and_its_own_chi2_shares():
    # The per-order joint fit is profile_per_window on the coarse, fine and
    # peak statistics; each window's chi2 share is at its own amplitudes.
    prepared, stats = next(_per_order_case([[0.7, 0.175], [0.5, 0.3]], 0.01, 1, seed=3))
    weights = np.array([1.5, 0.5])
    fit = joint_fit(stats, weights, prepared, _G1_PO, _G2_PO, nonneg=True, output="loglike",
                    amplitudes="per_order")
    _assert_same_dict(fit.profile, profile_per_window(stats, weights, _G1_PO, _G2_PO,
                                                      nonneg=True, output="loglike"))
    p1, p2 = (np.array([v]) for v in fit.crlb["peak_refined"])
    best = profile_per_window(fit.at_peak, weights, p1, p2, nonneg=True)["best"]
    _assert_same_dict(fit.best, best)
    shares = [chi2_from_stats(st, a1, a2).item()
              for st, a1, a2 in zip(fit.at_peak, best["a1"], best["a2"])]
    assert np.array_equal(fit.window_chi2, shares)
    assert np.sum(fit.window_chi2 * weights) == pytest.approx(best["chi2"], rel=1e-9)
    nu = sum(st["nu"] for st in fit.at_peak)
    assert best["chi2_red"] == pytest.approx(best["chi2"] / (nu - 2.0 - 2.0 * 2), rel=1e-12)


def test_joint_fit_refuses_an_unknown_amplitudes_mode():
    prepared, stats = _prepared()
    with pytest.raises(ValueError, match="amplitudes"):
        joint_fit(stats, None, prepared, V1, V2, nonneg=True, output="ccf",
                  amplitudes="per_window")


# =============================================================================
# A component with no lines in a window
# =============================================================================
def _flat_secondary_window(seed, sigma=0.01, v1=23.37, a1=0.7):
    """One 1 km/s window of a single-lined star, prepared against the real
    primary template and a flat secondary template -- constant up to
    round-off (1e-15), as a featureless model spectrum is."""
    t_wave, _ = log_lambda_grid(4995.0, 5095.0, 0.1)
    wave, _ = log_lambda_grid(*_WINDOWS[0], 1.0)
    flux = 1.0 + a1 * (_F1(wave / (1.0 + v1 / _C)) - 1.0)
    flux = flux + np.random.default_rng(seed).normal(0.0, sigma, wave.size)
    flat = 0.97 * (1.0 + 1e-15 * np.random.default_rng(seed + 1).normal(size=t_wave.size))
    return prepare_window(wave, flux, t_wave, _F1(t_wave), t_wave, flat, v_max=40.0)


def test_a_flat_secondary_template_gets_no_amplitude_and_no_velocity():
    # Template 2 has no lines (flat to round-off): prepare_window zeroes it,
    # so its amplitude is exactly 0 on every node (not b2 / S22 of round-off,
    # ~1e13), and v2 is unconstrained (non-finite error), while v1 and a1
    # stay right.
    win = _flat_secondary_window(seed=7)
    assert np.all(win.g2 == 0.0)
    st = window_stats(win.data_g, win.g1, win.g2, win.dv, _G1_PO, _G2_PO, sigma=0.01)
    assert np.all(st["S22"] == 0.0) and np.all(st["S11"] > 0.0)
    for nonneg in (False, True):
        res = profile_from_stats(st, _G1_PO, _G2_PO, nonneg=nonneg)
        assert np.all(res["a2"] == 0.0)
        assert res["best"]["a1"] == pytest.approx(0.7, abs=0.02)
        assert abs(res["best"]["v1"] - 23.37) <= 0.5               # nearest 1 km/s node
        cr = crlb_from_map(res["logL_profile"], _G1_PO, _G2_PO)
        assert not np.isfinite(cr["sigma_v2"])


def test_a_flat_secondary_window_pooled_per_order_with_a_normal_one():
    # Per-order amplitudes: the window whose secondary template is flat gets
    # a2 = 0 on every node; the normal SB2 window carries both components,
    # and the pooled velocities and its amplitudes are right.
    flat = _flat_secondary_window(seed=8)
    prepared, _ = next(_per_order_case([[0.7, 0.175], [0.5, 0.3]], 0.01, 1, seed=9))
    win_sb2 = prepared[1][0]                     # window 1: a2/a1 = 0.6
    stats = [window_stats(w.data_g, w.g1, w.g2, w.dv, _G1_PO, _G2_PO, sigma=0.01)
             for w in (flat, win_sb2)]
    res = profile_per_window(stats, None, _G1_PO, _G2_PO, nonneg=True)
    assert np.all(res["a2"][0] == 0.0)
    best = res["best"]
    assert best["a1"] == pytest.approx([0.7, 0.5], abs=0.02)
    assert best["a2"][1] == pytest.approx(0.3, abs=0.02)
    assert abs(best["v1"] - 23.37) <= 0.5 and abs(best["v2"] + 31.62) <= 0.5
    cr = crlb_from_map(res["logL_profile"], _G1_PO, _G2_PO)
    assert np.isfinite(cr["sigma_v1"]) and np.isfinite(cr["sigma_v2"])
    assert np.all(np.abs(np.subtract(cr["peak_refined"], _V_PO))
                  < 3.0 * np.array([cr["sigma_v1"], cr["sigma_v2"]]))


def _po_window(scale1=1.0, scale2=1.0, v_max=40.0):
    """One noiseless 1 km/s window of the SB2 at _V_PO, templates scaled."""
    t_wave, _ = log_lambda_grid(4995.0, 5095.0, 0.1)
    wave, _ = log_lambda_grid(*_WINDOWS[0], 1.0)
    flux = (1.0 + 0.7 * (_F1(wave / (1.0 + _V_PO[0] / _C)) - 1.0)
            + 0.3 * (_F2(wave / (1.0 + _V_PO[1] / _C)) - 1.0))
    return prepare_window(wave, flux, t_wave, scale1 * _F1(t_wave),
                          t_wave, scale2 * _F2(t_wave), v_max=v_max)


@pytest.mark.parametrize("scale1, scale2", [(1.0, 1e-7), (1.0, 1e7), (1e-9, 1e5)])
def test_the_fit_does_not_depend_on_the_templates_scale(scale1, scale2):
    # The amplitudes absorb a template's scale: the velocities and R must not
    # move, and no guard may mistake a small-valued template for a line-free one.
    ref, scaled = _po_window(), _po_window(scale1, scale2)
    fits = []
    for win in (ref, scaled):
        st = window_stats(win.data_g, win.g1, win.g2, win.dv, _G1_PO, _G2_PO,
                          reach=win.reach)
        fits.append(profile_from_stats(st, _G1_PO, _G2_PO)["best"])
    assert fits[1]["v1"] == fits[0]["v1"] and fits[1]["v2"] == fits[0]["v2"]
    assert fits[1]["R"] == pytest.approx(fits[0]["R"], rel=1e-9)
    assert fits[1]["a1"] * scale1 == pytest.approx(fits[0]["a1"], rel=1e-6)
    assert fits[1]["a2"] * scale2 == pytest.approx(fits[0]["a2"], rel=1e-6)


def test_the_templates_cover_the_most_negative_cz_shift():
    # Under 'cz' the shift at -v_max, c ln(1 - v/c), is larger than at +v_max
    # by ~v^2/2c; the reach must cover it (at 500 km/s and a fine grid the
    # difference is a few pixels).
    t_wave, _ = log_lambda_grid(4950.0, 5150.0, 0.01)
    wave, _ = log_lambda_grid(5040.0, 5060.0, 0.1)
    win = prepare_window(wave, np.ones_like(wave), t_wave, _F1(t_wave), t_wave, _F2(t_wave),
                         v_max=500.0)
    most = abs(estimator.shift_kms(-500.0, "cz")) / win.dv
    assert win.reach >= most and win.extension == win.reach + 20


def test_window_stats_refuses_a_shift_into_the_ramp():
    # Shifts beyond the reach (into the ramp, still inside the extension)
    # are refused when the window's reach is passed.
    win = _po_window(v_max=10.0)
    inside = np.array([-0.95, 0.95]) * win.reach * win.dv
    ramp = np.array([(win.reach + 5) * win.dv])
    window_stats(win.data_g, win.g1, win.g2, win.dv, inside, inside, reach=win.reach)
    with pytest.raises(ValueError, match="v1_grid.*reach"):
        window_stats(win.data_g, win.g1, win.g2, win.dv, ramp, inside, reach=win.reach)


def test_profile_per_window_needs_one_weight_per_window():
    win = _po_window()
    st = window_stats(win.data_g, win.g1, win.g2, win.dv, _G1_PO, _G2_PO)
    with pytest.raises(ValueError, match="one factor per window"):
        profile_per_window([st, st, st], [1.0], _G1_PO, _G2_PO)


@pytest.mark.parametrize("half", [0, -1, 2.5])
def test_crlb_from_map_needs_a_positive_integer_half_width(half):
    v = np.arange(-5.0, 6.0)
    logL = -np.add.outer(v ** 2, v ** 2)
    with pytest.raises(ValueError, match="half"):
        crlb_from_map(logL, v, v, half=half)


def test_per_order_without_nonneg_recovers_each_windows_amplitudes():
    # amplitudes='per_order' with unconstrained amplitudes, noiseless.
    t_wave, _ = log_lambda_grid(4995.0, 5095.0, 0.1)
    stats = []
    for (lo, hi), (a1, a2) in zip(_WINDOWS, ((0.75, 0.25), (0.6, 0.4))):
        wave, _ = log_lambda_grid(lo, hi, 1.0)
        flux = (1.0 + a1 * (_F1(wave / (1.0 + _V_PO[0] / _C)) - 1.0)
                + a2 * (_F2(wave / (1.0 + _V_PO[1] / _C)) - 1.0))
        win = prepare_window(wave, flux, t_wave, _F1(t_wave), t_wave, _F2(t_wave), v_max=40.0)
        stats.append(window_stats(win.data_g, win.g1, win.g2, win.dv,
                                  np.array([_V_PO[0]]), np.array([_V_PO[1]])))
    best = profile_per_window(stats, None, np.array([_V_PO[0]]), np.array([_V_PO[1]]),
                              nonneg=False)["best"]
    np.testing.assert_allclose(best["a1"], [0.75, 0.6], atol=2e-3)
    np.testing.assert_allclose(best["a2"], [0.25, 0.4], atol=2e-3)
