"""Tests for the array-level UNICOR engine (:mod:`unicor.ccf`).

Synthetic single-line spectra on a log-lambda grid: the CCF peak lands on the
injected shift, identical orders combine to the single-order CCF, and the
sub-pixel parabola is exact on a parabola. A noise-free line forest checks that
the CCF has no pull toward v = 0. No external data.
"""
import numpy as np
import pytest
from scipy.signal import correlate

import unicor

DV = 0.5  # km/s


def _line_spectrum(v_kms, wave):
    """A Gaussian absorption line at 5050 A, Doppler-shifted by ``v_kms``."""
    rest = 5050.0 * (1.0 + v_kms / unicor.C_KMS)
    return 1.0 - 0.5 * np.exp(-0.5 * ((wave - rest) / 0.15) ** 2)


@pytest.fixture
def grid():
    wave, _ = unicor.log_lambda_grid(5000.0, 5100.0, DV)
    return wave


def _ccf(v_template, signal, wave, vmax, dv=DV):
    """CCF of ``signal`` against the line template, built on the extended grid."""
    ext = unicor.extend_log_grid(wave, int(round(vmax / dv)))
    return unicor.ccf_vs_velocity(_line_spectrum(v_template, ext), signal, dv)


@pytest.mark.parametrize("v_inj", [-12.3, 0.0, 7.7])
def test_ccf_peak_recovers_injected_shift(grid, v_inj):
    v, c = _ccf(0.0, _line_spectrum(v_inj, grid), grid, vmax=40)
    rv, peak = unicor.subpixel_peak(v, c)
    assert rv == pytest.approx(v_inj, abs=0.05)
    assert peak == pytest.approx(1.0, abs=0.01)


def test_ccf_lags_are_symmetric_and_normalized(grid):
    v, c = _ccf(0.0, _line_spectrum(0.0, grid), grid, vmax=10)
    np.testing.assert_allclose(v, -v[::-1])
    assert v[1] - v[0] == pytest.approx(DV)
    assert v[-1] == pytest.approx(10.0)
    assert c.max() == pytest.approx(1.0, abs=1e-12) and v[np.argmax(c)] == 0.0


def test_ccf_is_the_pearson_correlation_at_every_lag():
    # Each lag k correlates all N observed pixels with the N-pixel template
    # segment shifted by k, normalised by that segment's own mean and norm.
    rng = np.random.default_rng(3)
    n, k = 50, 7
    template = rng.normal(size=n + 2 * k) + np.linspace(0, 3, n + 2 * k)
    signal = rng.normal(size=n)
    v, c = unicor.ccf_vs_velocity(template, signal, DV)
    np.testing.assert_allclose(v, np.arange(-k, k + 1) * DV)
    # A signal red-shifted by +lag pixels matches the segment starting at k - lag.
    expected = [np.corrcoef(template[k - lag: k - lag + n], signal)[0, 1]
                for lag in range(-k, k + 1)]
    np.testing.assert_allclose(c, expected, atol=1e-12)


def test_extend_log_grid_continues_the_grid_on_both_sides(grid):
    ext = unicor.extend_log_grid(grid, 40)
    assert ext.size == grid.size + 80
    np.testing.assert_allclose(ext[40:-40], grid, rtol=1e-13)
    assert unicor.dsp.log_grid_step(ext) == pytest.approx(DV, rel=1e-9)
    with pytest.raises(ValueError, match="uniform"):
        unicor.extend_log_grid(np.delete(grid, 100), 40)
    with pytest.raises(ValueError, match="n_lags"):
        unicor.extend_log_grid(grid, 0)


def test_combining_identical_orders_returns_the_single_order_ccf(grid):
    v, c = _ccf(0.0, _line_spectrum(3.0, grid), grid, vmax=20)
    np.testing.assert_allclose(unicor.combine_ccfs(np.vstack([c, c, c])),
                               np.abs(c), atol=1e-12)


def test_subpixel_peak_is_exact_on_a_parabola():
    v = np.arange(-10, 11) * 1.0
    ccf = 0.9 - 0.01 * (v - 0.37) ** 2
    rv, peak = unicor.subpixel_peak(v, ccf)
    assert rv == pytest.approx(0.37, abs=1e-10)
    assert peak == pytest.approx(0.9, abs=1e-12)
    assert unicor.ccf_at(v, ccf, 0.37) == pytest.approx(0.9, abs=1e-12)


@pytest.mark.parametrize("npts", [2, 4])
def test_subpixel_peak_rejects_bad_window(npts):
    with pytest.raises(ValueError):
        unicor.subpixel_peak(np.arange(9.0), np.ones(9), npts=npts)


@pytest.mark.parametrize("n_template", [10, 12, 13])
def test_ccf_rejects_a_template_not_extended_by_whole_lags_on_each_side(n_template):
    # Shorter, equal-length (the pre-extension contract) or odd excess: refused.
    signal = np.random.default_rng(0).normal(size=12)
    template = np.random.default_rng(1).normal(size=n_template)
    with pytest.raises(ValueError, match="2K"):
        unicor.ccf_vs_velocity(template, signal, DV)


@pytest.mark.parametrize("argmax", [0, 1, 19, 20])
def test_subpixel_peak_refuses_a_peak_at_the_scan_edge(argmax):
    # Near the ends the 5-point window used to wrap to the far end of the array
    # and return a plausible-looking velocity of the wrong sign.
    v = np.arange(21) * 1.0
    ccf = np.exp(-0.5 * ((v - argmax) / 2.0) ** 2)
    with pytest.raises(ValueError, match="edge"):
        unicor.subpixel_peak(v, ccf)


def test_subpixel_peak_refuses_a_non_concave_ccf():
    # A spike flanked by a W: the best 5-point parabola opens upwards.
    v = np.arange(-5, 6) * 1.0
    ccf = np.array([0.0, 0.0, 0.0, 0.99, 0.5, 1.0, 0.5, 0.99, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="concave"):
        unicor.subpixel_peak(v, ccf)


def test_ccf_at_refuses_velocities_outside_the_scan():
    v = np.arange(-5, 6) * 1.0
    with pytest.raises(ValueError, match="outside"):
        unicor.ccf_at(v, np.ones_like(v), 7.0)


def test_ccf_refuses_a_flat_or_non_finite_input(grid):
    flux = _line_spectrum(0.0, grid)
    ext = _line_spectrum(0.0, unicor.extend_log_grid(grid, 20))
    with pytest.raises(ValueError, match="constant"):
        unicor.ccf_vs_velocity(np.ones_like(ext), flux, DV)
    with pytest.raises(ValueError, match="constant"):
        unicor.ccf_vs_velocity(ext, np.ones_like(flux), DV)
    bad = flux.copy()
    bad[10] = np.nan
    with pytest.raises(ValueError, match="finite"):
        unicor.ccf_vs_velocity(ext, bad, DV)
    bad = ext.copy()
    bad[-1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        unicor.ccf_vs_velocity(bad, flux, DV)


def test_ccf_refuses_a_lag_whose_template_segment_is_constant():
    # Lines only near one end: at the far lags the N-pixel template segment is
    # pure continuum, where the correlation is undefined (0/0).
    n, k = 100, 30
    template = np.ones(n + 2 * k)
    template[5:15] = 0.5
    signal = np.ones(n)
    signal[0:10] = 0.5
    with pytest.raises(ValueError, match="constant"):
        unicor.ccf_vs_velocity(template, signal, DV)


def test_log_grid_step_verifies_the_grid(grid):
    assert unicor.dsp.log_grid_step(grid) == pytest.approx(DV, rel=1e-9)
    with pytest.raises(ValueError, match="uniform"):
        unicor.dsp.log_grid_step(np.delete(grid, 100))   # a dropped pixel
    with pytest.raises(ValueError, match="increasing"):
        unicor.dsp.log_grid_step(grid[::-1])


def test_resample_refuses_a_decreasing_source_axis(grid):
    flux = _line_spectrum(0.0, grid)
    with pytest.raises(ValueError, match="increasing"):
        unicor.resample(grid[::-1], flux[::-1], grid)


def _three_order_ccfs(grid):
    shifts = (1.0, 3.0, 6.0)
    ccfs = [_ccf(0.0, _line_spectrum(s, grid), grid, vmax=20)[1] for s in shifts]
    return np.vstack(ccfs)


def test_equal_weight_combination_is_the_zucker_formula(grid):
    ccfs = _three_order_ccfs(grid)
    expected = np.sqrt(1 - np.prod(1 - ccfs ** 2, axis=0) ** (1 / 3))
    np.testing.assert_allclose(unicor.combine_ccfs(ccfs), expected, atol=1e-12)
    np.testing.assert_allclose(unicor.combine_ccfs(ccfs, n_eff=[700, 700, 700]),
                               expected, atol=1e-12)


def test_weighted_combination_matches_the_summed_log_likelihood(grid):
    # Zucker (2003): log L = -1/2 sum_i N_i log(1 - C_i^2) must equal the
    # single-CCF form -N_tot/2 log(1 - ML^2) of the combined CCF.
    ccfs = _three_order_ccfs(grid)
    n_eff = np.array([100.0, 400.0, 1500.0])
    ml = unicor.combine_ccfs(ccfs, n_eff=n_eff)
    np.testing.assert_allclose(n_eff.sum() * np.log(1 - ml ** 2),
                               (n_eff[:, None] * np.log(1 - ccfs ** 2)).sum(axis=0),
                               rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("bad", [[100.0, 100.0], [100.0, 0.0, 100.0], [100.0, np.nan, 1.0]])
def test_weighted_combination_refuses_bad_bin_counts(grid, bad):
    with pytest.raises(ValueError, match="n_eff"):
        unicor.combine_ccfs(_three_order_ccfs(grid), n_eff=bad)


@pytest.mark.parametrize("vmax", [10.0, 60.0])
def test_rv_uncertainty_matches_the_closed_form_for_a_gaussian_ccf(vmax):
    # C(v) = A exp(-v^2 / 2s^2) has C'' = -A/s^2 at the peak, so Zucker's
    # sigma^2 = -[N C'' C / (1 - C^2)]^-1 = s^2 (1 - A^2) / (N A^2) -- with no
    # dependence on how many velocity lags were scanned.
    A, s, n_eff = 0.8, 4.0, 2500.0
    v = np.arange(-vmax, vmax + 1e-9, 0.25)
    ccf = A * np.exp(-0.5 * (v / s) ** 2)
    expected = s * np.sqrt(1 - A ** 2) / (A * np.sqrt(n_eff))
    assert unicor.rv_uncertainty(v, ccf, 0.0, n_eff) == pytest.approx(expected, rel=1e-3)


def test_rv_uncertainty_refuses_a_non_positive_bin_count():
    v = np.arange(-10, 10.25, 0.25)
    with pytest.raises(ValueError, match="n_eff"):
        unicor.rv_uncertainty(v, np.exp(-0.5 * v ** 2), 0.0, 0)


# --- No pull toward v = 0 ----------------------------------------------------
# Correlating against a template on the SAME span as the observation (scipy
# 'full' mode, one global norm) leaves only N - |k| overlapping pixels at lag k,
# so E[C(k)] is tapered by ~(1 - |k|/N) and the peak is pulled toward zero --
# by tens to hundreds of m/s per 60 A order for rotating stars, always toward
# zero. With the template on a grid extended by K pixels on each side, every
# lag correlates all N observed pixels and the pull is gone.

FOREST_DV = 0.75        # km/s
FOREST_VMAX = 100.0     # km/s
FOREST_V0 = (-50.0, -20.0, -5.0, 5.0, 20.0, 50.0)   # km/s, rounded to whole lags


def _line_forest(lam_lo, lam_hi, vsini, seed, n_lines=200, resolution=100_000,
                 fine_dv=0.05, margin=10.0, sigma_intrinsic=1.5):
    """Rest-frame normalised spectrum of ~``n_lines`` Gaussian lines per order.

    Lines (depths U(0.05, 0.6), intrinsic sigma 1.5 km/s) are combined
    multiplicatively on a fine log-lambda grid (step ``fine_dv`` km/s) covering
    ``[lam_lo - margin, lam_hi + margin]`` Angstrom, then broadened by the
    instrument (FWHM c/R) and by rotation (``vsini`` km/s).
    """
    wave, u = unicor.log_lambda_grid(lam_lo - margin, lam_hi + margin, fine_dv)
    rng = np.random.default_rng(seed)
    n_total = int(round(n_lines * (wave[-1] - wave[0]) / (lam_hi - lam_lo)))
    centres = rng.uniform(u[0], u[-1], n_total)
    depths = rng.uniform(0.05, 0.6, n_total)
    half = int(8 * sigma_intrinsic / fine_dv)
    flux = np.ones_like(wave)
    for uc, depth in zip(centres, depths):
        i = int(uc / fine_dv)
        sl = slice(max(0, i - half), i + half + 2)
        flux[sl] *= 1 - depth * np.exp(-0.5 * ((u[sl] - uc) / sigma_intrinsic) ** 2)
    flux = unicor.broaden_gaussian(flux, fine_dv, unicor.C_KMS / resolution)
    if vsini:
        flux = unicor.broaden_rotational(flux, fine_dv, vsini)
    return wave, flux


def _forest_sweep(ccf_fn, vsini, seed):
    """Recovered-minus-injected lag velocity (m/s) for each FOREST_V0.

    The shift is a whole number ``k0`` of grid steps (observed flux at lambda
    is the template at lambda * exp(-k0 dln)), so the expected peak is exactly
    ``k0 * dv`` and the sub-pixel parabola has no pixel-phase error; how a lag
    maps to a Doppler velocity is a separate convention, not tested here.
    """
    wave, _ = unicor.log_lambda_grid(5000.0, 5060.0, FOREST_DV)
    dln = np.log(wave[-1] / wave[0]) / (wave.size - 1)
    t_wave, t_flux = _line_forest(5000.0, 5060.0, vsini, seed)
    errors = []
    for v0 in FOREST_V0:
        k0 = round(np.log1p(v0 / unicor.C_KMS) / dln)
        observed = unicor.resample(t_wave, t_flux, wave * np.exp(-k0 * dln), kind='cubic')
        v, c = ccf_fn(t_wave, t_flux, wave, observed)
        rv, _ = unicor.subpixel_peak(v, c)
        errors.append((rv - k0 * FOREST_DV) * 1e3)
    return np.array(errors)


def _extended_template_ccf(t_wave, t_flux, wave, observed):
    k = int(round(FOREST_VMAX / FOREST_DV))
    template = unicor.resample(t_wave, t_flux, unicor.extend_log_grid(wave, k))
    return unicor.ccf_vs_velocity(template, observed, FOREST_DV)


def _same_span_ccf(t_wave, t_flux, wave, observed):
    """The superseded estimator, kept as the control that shows the pull."""
    t = unicor.resample(t_wave, t_flux, wave)
    t = t - t.mean()
    s = observed - observed.mean()
    k = int(round(FOREST_VMAX / FOREST_DV))
    mid = t.size - 1
    full = correlate(s, t, mode='full', method='fft')[mid - k: mid + k + 1]
    return np.arange(-k, k + 1) * FOREST_DV, full / np.sqrt((t ** 2).sum() * (s ** 2).sum())


@pytest.mark.parametrize("vsini", [0.0, 25.0])
@pytest.mark.parametrize("seed", [0, 1])
def test_forest_rv_has_no_pull_toward_zero(vsini, seed):
    errors = _forest_sweep(_extended_template_ccf, vsini, seed)
    # What remains is the 5-point parabola reading a CCF whose wings are not
    # exactly symmetric about an integer-lag peak (edge content differs by j
    # pixels at lag +-j): <~1.2 m/s, either sign. The same-span CCF erred by
    # up to 8 m/s (sharp lines) and 370 m/s (vsini 25 km/s) on these forests.
    np.testing.assert_array_less(np.abs(errors), 1.5)
    # Sign test: the error projected on sign(v0) (negative = pulled toward 0).
    pull = np.mean(np.sign(FOREST_V0) * errors)
    assert abs(pull) < 0.5


def test_same_span_control_does_show_the_pull_toward_zero():
    # The sweep is sensitive: the superseded same-span CCF, on the same
    # vsini 25 km/s forests, is pulled toward zero by tens of m/s.
    pulls = [np.mean(np.sign(FOREST_V0) * _forest_sweep(_same_span_ccf, 25.0, seed))
             for seed in (0, 1)]
    assert max(pulls) < -30.0
