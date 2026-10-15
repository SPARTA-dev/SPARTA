"""Tests for the array-level UNICOR engine (:mod:`unicor.ccf`).

Synthetic single-line spectra on a log-lambda grid: the CCF peak lands on the
injected shift, identical orders combine to the single-order CCF, and the
spline peak is exact on a parabola and does not depend on the sub-lag phase. A
noise-free line forest checks that
the CCF has no pull toward v = 0, and noise-free injections under both velocity
conventions check the lag -> velocity mapping. No external data.
"""
import inspect

import numpy as np
import pytest
from scipy.signal import correlate

import unicor

DV = 0.5  # km/s
C = 299792.458  # km/s
CONVENTIONS = ('cz', 'relativistic')


def _factor(v_kms, convention):
    """lambda_obs / lambda_rest, written out from the two definitions."""
    b = np.asarray(v_kms, float) / C
    return 1.0 + b if convention == 'cz' else np.sqrt((1.0 + b) / (1.0 - b))


def _line_spectrum(v_kms, wave, convention='cz'):
    """A Gaussian absorption line at 5050 A, its centre Doppler-shifted by ``v_kms``."""
    centre = 5050.0 * _factor(v_kms, convention)
    return 1.0 - 0.5 * np.exp(-0.5 * ((wave - centre) / 0.15) ** 2)


def _log_wave(n, dln=DV / C, lam0=5000.0):
    """``n`` nodes of a log-lambda grid with ln-step ``dln`` (Angstrom)."""
    return lam0 * np.exp(dln * np.arange(n))


@pytest.fixture
def grid():
    wave, _ = unicor.log_lambda_grid(5000.0, 5100.0, DV)
    return wave


def _ccf(v_template, signal, wave, vmax, dv=DV, **kwargs):
    """CCF of ``signal`` against the line template, built on the extended grid."""
    ext = unicor.extend_log_grid(wave, int(round(vmax / dv)))
    return unicor.ccf_vs_velocity(_line_spectrum(v_template, ext), signal, wave, **kwargs)


@pytest.mark.parametrize("v_inj", [-12.3, 0.0, 7.7])
def test_ccf_peak_recovers_injected_shift(grid, v_inj):
    v, c = _ccf(0.0, _line_spectrum(v_inj, grid), grid, vmax=40)
    rv, peak = unicor.subpixel_peak(v, c)
    assert rv == pytest.approx(v_inj, abs=0.05)
    assert peak == pytest.approx(1.0, abs=0.01)


def test_ccf_lags_are_cz_velocities_and_normalized(grid):
    # Lag k multiplies lambda by exp(k dln); under the default (cz) convention
    # that is v = c (exp(k dln) - 1): 0 at the centre, a hair above k * dv
    # (by (k dv)^2 / 2c) on either side, so the axis is not antisymmetric.
    v, c = _ccf(0.0, _line_spectrum(0.0, grid), grid, vmax=10)
    k = np.arange(-20, 21)
    np.testing.assert_allclose(v, C * np.expm1(k * DV / C), rtol=1e-12, atol=1e-15)
    assert v[20] == 0.0
    assert v[-1] - 10.0 == pytest.approx(10.0 ** 2 / (2 * C), rel=1e-3)
    assert c.max() == pytest.approx(1.0, abs=1e-12) and v[np.argmax(c)] == 0.0


def test_ccf_is_the_pearson_correlation_at_every_lag():
    # Each lag k correlates all N observed pixels with the N-pixel template
    # segment shifted by k, normalised by that segment's own mean and norm.
    rng = np.random.default_rng(3)
    n, k = 50, 7
    template = rng.normal(size=n + 2 * k) + np.linspace(0, 3, n + 2 * k)
    signal = rng.normal(size=n)
    v, c = unicor.ccf_vs_velocity(template, signal, _log_wave(n))
    np.testing.assert_allclose(v, C * np.expm1(np.arange(-k, k + 1) * DV / C))
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


@pytest.mark.parametrize("v_peak", [38.02741, -149.98765, 0.01234])
def test_subpixel_peak_is_exact_far_from_zero_velocity(v_peak):
    # A parabola fitted in absolute velocity (x ~ 38 km/s, steps of 0.05 km/s)
    # is ill-conditioned: it lost ~0.1-0.25 m/s to round-off on real CCFs.
    # Fitted about the central node it is exact to round-off.
    v = np.round(v_peak / 0.05) * 0.05 + np.arange(-10, 11) * 0.05
    ccf = 0.9 - 3.0 * (v - v_peak) ** 2
    rv, peak = unicor.subpixel_peak(v, ccf)
    assert abs(rv - v_peak) * 1e3 < 1e-6          # m/s
    assert peak == pytest.approx(0.9, abs=1e-12)


@pytest.mark.parametrize("n_template", [10, 12, 13])
def test_ccf_rejects_a_template_not_extended_by_whole_lags_on_each_side(n_template):
    # Shorter, equal-length (the pre-extension contract) or odd excess: refused.
    signal = np.random.default_rng(0).normal(size=12)
    template = np.random.default_rng(1).normal(size=n_template)
    with pytest.raises(ValueError, match="2K"):
        unicor.ccf_vs_velocity(template, signal, _log_wave(12))


@pytest.mark.parametrize("argmax", [0, 1, 19, 20])
def test_subpixel_peak_refuses_a_peak_at_the_scan_edge(argmax):
    # Near the ends the 5-point window used to wrap to the far end of the array
    # and return a plausible-looking velocity of the wrong sign.
    v = np.arange(21) * 1.0
    ccf = np.exp(-0.5 * ((v - argmax) / 2.0) ** 2)
    with pytest.raises(ValueError, match="edge"):
        unicor.subpixel_peak(v, ccf)


def test_a_one_lag_spike_is_found_but_flagged_as_undersampled():
    # A spike flanked by a W: the old 5-point parabola opened upwards and gave
    # up. The spline puts the peak on the spike, and warns that a peak one lag
    # wide cannot be located between lags.
    v = np.arange(-5, 6) * 1.0
    ccf = np.array([0.0, 0.0, 0.0, 0.99, 0.5, 1.0, 0.5, 0.99, 0.0, 0.0, 0.0])
    with pytest.warns(UserWarning, match="undersampled"):
        rv, _ = unicor.subpixel_peak(v, ccf)
    assert rv == pytest.approx(0.0, abs=1e-12)


def test_ccf_at_refuses_velocities_outside_the_scan():
    v = np.arange(-5, 6) * 1.0
    with pytest.raises(ValueError, match="outside"):
        unicor.ccf_at(v, np.ones_like(v), 7.0)


def test_ccf_refuses_a_flat_or_non_finite_input(grid):
    flux = _line_spectrum(0.0, grid)
    ext = _line_spectrum(0.0, unicor.extend_log_grid(grid, 20))
    with pytest.raises(ValueError, match="constant"):
        unicor.ccf_vs_velocity(np.ones_like(ext), flux, grid)
    with pytest.raises(ValueError, match="constant"):
        unicor.ccf_vs_velocity(ext, np.ones_like(flux), grid)
    bad = flux.copy()
    bad[10] = np.nan
    with pytest.raises(ValueError, match="finite"):
        unicor.ccf_vs_velocity(ext, bad, grid)
    bad = ext.copy()
    bad[-1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        unicor.ccf_vs_velocity(bad, flux, grid)


def test_ccf_refuses_a_lag_whose_template_segment_is_constant():
    # Lines only near one end: at the far lags the N-pixel template segment is
    # pure continuum, where the correlation is undefined (0/0).
    n, k = 100, 30
    template = np.ones(n + 2 * k)
    template[5:15] = 0.5
    signal = np.ones(n)
    signal[0:10] = 0.5
    with pytest.raises(ValueError, match="constant"):
        unicor.ccf_vs_velocity(template, signal, _log_wave(n))


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
    log_l = (n_eff[:, None] * np.log(1 - ccfs ** 2)).sum(axis=0)
    # Compared as ML, not as log(1 - ML^2): near the peak 1 - ML^2 ~ 3e-8, and
    # recovering it from ML would cancel away ~eps/3e-8 of its precision.
    np.testing.assert_allclose(ml, np.sqrt(-np.expm1(log_l / n_eff.sum())), rtol=1e-12)


@pytest.mark.parametrize("bad", [[100.0, 100.0], [100.0, 0.0, 100.0], [100.0, np.nan, 1.0]])
def test_weighted_combination_refuses_bad_bin_counts(grid, bad):
    with pytest.raises(ValueError, match="n_eff"):
        unicor.combine_ccfs(_three_order_ccfs(grid), n_eff=bad)


# --- The log-likelihood mode (output='loglike') -------------------------------
def test_loglike_mode_is_the_zucker_log_likelihood_of_the_ccf(grid):
    rng = np.random.default_rng(3)
    signal = _line_spectrum(4.2, grid) + 0.02 * rng.standard_normal(grid.size)
    v, c = _ccf(0.0, signal, grid, vmax=20)
    v_l, lnl = _ccf(0.0, signal, grid, vmax=20, output='loglike', n_eff=812.0)
    np.testing.assert_array_equal(v_l, v)
    np.testing.assert_allclose(lnl, -0.5 * 812.0 * np.log(1 - c ** 2), rtol=1e-12)
    np.testing.assert_allclose(unicor.loglike_from_ccf(c, 812.0), lnl, rtol=0, atol=0)
    assert np.argmax(lnl) == np.argmax(c ** 2)
    assert np.all(lnl >= 0)


def test_combined_loglike_is_the_sum_over_orders_and_the_ml_ccf_form(grid):
    # Zucker (2003): sum_i -(N_i/2) ln(1 - C_i^2) = -(N_tot/2) ln(1 - ML^2).
    ccfs = _three_order_ccfs(grid)
    n_eff = np.array([100.0, 400.0, 1500.0])
    total = unicor.combine_ccfs(ccfs, n_eff=n_eff, output='loglike')
    per_order = unicor.loglike_from_ccf(ccfs, n_eff)
    np.testing.assert_allclose(total, per_order.sum(axis=0), rtol=1e-14)
    ml = unicor.combine_ccfs(ccfs, n_eff=n_eff)
    np.testing.assert_allclose(total, unicor.loglike_from_ccf(ml, n_eff.sum()),
                               rtol=1e-9, atol=1e-9)
    assert np.argmax(total) == np.argmax(ml)


def test_a_perfect_match_has_unbounded_log_likelihood():
    assert unicor.loglike_from_ccf(np.array([1.0, 0.0, -1.0]), 10.0).tolist() == \
        [np.inf, 0.0, np.inf]


@pytest.mark.parametrize("bad", ['CCF', 'likelihood', None])
def test_an_unknown_output_mode_is_refused(grid, bad):
    with pytest.raises(ValueError, match="output"):
        _ccf(0.0, _line_spectrum(1.0, grid), grid, vmax=10, output=bad)
    with pytest.raises(ValueError, match="output"):
        unicor.combine_ccfs(_three_order_ccfs(grid), n_eff=[1.0, 1.0, 1.0], output=bad)


def test_loglike_needs_the_number_of_independent_bins(grid):
    with pytest.raises(ValueError, match="n_eff"):
        _ccf(0.0, _line_spectrum(1.0, grid), grid, vmax=10, output='loglike')
    with pytest.raises(ValueError, match="n_eff"):
        unicor.combine_ccfs(_three_order_ccfs(grid), output='loglike')
    with pytest.raises(ValueError, match="n_eff"):
        unicor.loglike_from_ccf(_three_order_ccfs(grid), [1.0, 2.0])
    with pytest.raises(ValueError, match="n_eff"):
        unicor.loglike_from_ccf(np.zeros(5), -1.0)


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
    """Recovered-minus-injected velocity (m/s) for each FOREST_V0.

    The shift is a whole number ``k0`` of grid steps (observed flux at lambda
    is the template at lambda * exp(-k0 dln)), so the expected peak is exactly
    the cz velocity of that lag, ``c (exp(k0 dln) - 1)``, with no sub-lag
    interpolation involved.
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
        errors.append((rv - C * np.expm1(k0 * dln)) * 1e3)
    return np.array(errors)


def _extended_template_ccf(t_wave, t_flux, wave, observed):
    k = int(round(FOREST_VMAX / FOREST_DV))
    template = unicor.resample(t_wave, t_flux, unicor.extend_log_grid(wave, k))
    return unicor.ccf_vs_velocity(template, observed, wave)


def _same_span_ccf(t_wave, t_flux, wave, observed):
    """The superseded estimator, kept as the control that shows the pull."""
    t = unicor.resample(t_wave, t_flux, wave)
    t = t - t.mean()
    s = observed - observed.mean()
    k = int(round(FOREST_VMAX / FOREST_DV))
    mid = t.size - 1
    full = correlate(s, t, mode='full', method='fft')[mid - k: mid + k + 1]
    dln = np.log(wave[-1] / wave[0]) / (wave.size - 1)
    return (C * np.expm1(np.arange(-k, k + 1) * dln),
            full / np.sqrt((t ** 2).sum() * (s ** 2).sum()))


@pytest.mark.parametrize("vsini", [0.0, 25.0])
@pytest.mark.parametrize("seed", [0, 1])
def test_forest_rv_has_no_pull_toward_zero(vsini, seed):
    errors = _forest_sweep(_extended_template_ccf, vsini, seed)
    # <= 0.005 m/s with the spline peak (the 5-point parabola left up to
    # 1.2 m/s here, reading wings that are not symmetric about an integer-lag
    # peak). The same-span CCF erred by up to 8 m/s (sharp lines) and 370 m/s
    # (vsini 25 km/s) on these forests.
    np.testing.assert_array_less(np.abs(errors), 0.05)
    # Sign test: the error projected on sign(v0) (negative = pulled toward 0).
    pull = np.mean(np.sign(FOREST_V0) * errors)
    assert abs(pull) < 0.5


def test_same_span_control_does_show_the_pull_toward_zero():
    # The sweep is sensitive: the superseded same-span CCF, on the same
    # vsini 25 km/s forests, is pulled toward zero by tens of m/s.
    pulls = [np.mean(np.sign(FOREST_V0) * _forest_sweep(_same_span_ccf, 25.0, seed))
             for seed in (0, 1)]
    assert max(pulls) < -30.0


# --- Velocity conventions ----------------------------------------------------
# On a grid uniform in ln(lambda) with step dln, a shift of k pixels multiplies
# lambda by exp(k dln). What velocity that is depends on the convention:
#   cz (IAU 2000 radial-velocity measure; IRAF fxcor):
#       lambda_obs / lambda_rest = 1 + v/c         ->  v = c (exp(k dln) - 1)
#   relativistic Doppler:
#       lambda_obs / lambda_rest = sqrt((1+b)/(1-b)) -> v = c tanh(k dln)
# Reading the lag as k * dv instead is off from cz by -v^2/2c (-37.5 m/s at
# 150 km/s). The expectations below are written from these definitions.

INJECT_V0 = (-150.0, -100.0, -50.0, -10.0, 10.0, 50.0, 100.0, 150.0)   # km/s


def _velocity_of_lag(k, dln, convention):
    """The Doppler velocity (km/s) whose factor is exp(k dln), by definition."""
    x = np.asarray(k, float) * dln
    return C * np.expm1(x) if convention == 'cz' else C * np.tanh(x)


def _snap_to_lag(v0, dln, convention):
    """The velocity nearest ``v0`` (km/s) that shifts by a whole number of steps.

    At a whole-lag shift the CCF is symmetric about its peak, so no sub-lag
    interpolation is involved and any residual is the mapping's.
    """
    return _velocity_of_lag(round(np.log(_factor(v0, convention)) / dln), dln, convention)


def _interior_forest(seed=4, n_lines=120, sigma_kms=3.0, lam_lo=5000.0, lam_hi=5060.0):
    """Rest-frame forest of Gaussian lines (in velocity), as a function of lambda.

    Line centres keep 400 km/s clear of the order edges, so a shift of up to
    150 km/s leaves continuum around both edges and the CCF is symmetric
    about a whole-lag peak.
    """
    rng = np.random.default_rng(seed)
    span = C * np.log(lam_hi / lam_lo)
    centres = rng.uniform(400.0, span - 400.0, n_lines)
    depths = rng.uniform(0.05, 0.6, n_lines)

    def flux(wave):
        u = C * np.log(np.asarray(wave, float) / lam_lo)
        return np.prod(1.0 - depths * np.exp(
            -0.5 * ((u[:, None] - centres) / sigma_kms) ** 2), axis=1)
    return flux


def test_doppler_factor_and_shift_follow_the_definitions():
    v = np.array([-150.0, -10.0, 0.0, 0.3, 50.0, 2.9e5])
    for conv in CONVENTIONS:
        np.testing.assert_allclose(unicor.doppler_factor(v, conv), _factor(v, conv),
                                   rtol=1e-14)
        np.testing.assert_allclose(unicor.shift_kms(v, conv), C * np.log(_factor(v, conv)),
                                   rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(unicor.doppler_shift_wave([5000.0, 6000.0], 42.0, conv),
                                   np.array([5000.0, 6000.0]) * _factor(42.0, conv),
                                   rtol=1e-14)
    # The relativistic shift is the rapidity: c artanh(v/c).
    np.testing.assert_allclose(unicor.shift_kms(v, 'relativistic'), C * np.arctanh(v / C),
                               rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("conv", CONVENTIONS)
def test_velocity_from_log_shift_inverts_the_shift(conv):
    v = np.array([-150.0, -1e-6, 0.0, 1e-9, 10.0, 150.0, 1e5])
    np.testing.assert_allclose(unicor.velocity_from_log_shift(unicor.shift_kms(v, conv) / C,
                                                              conv), v, rtol=1e-12, atol=1e-18)
    dln = np.array([-5e-4, -1e-6, 0.0, 1e-6, 5e-4])
    np.testing.assert_allclose(unicor.velocity_from_log_shift(dln, conv),
                               _velocity_of_lag(1, dln, conv), rtol=1e-12)
    # A tiny shift keeps full relative precision (exp(x) - 1 would lose ~4 digits).
    assert unicor.velocity_from_log_shift(1e-12, conv) == pytest.approx(C * 1e-12, rel=1e-11)


def test_velocities_at_or_beyond_the_speed_of_light_are_refused():
    for conv, v in (('cz', -C), ('cz', -2 * C), ('relativistic', C), ('relativistic', -1.5 * C)):
        with pytest.raises(ValueError, match="speed of light"):
            unicor.doppler_factor(v, conv)
        with pytest.raises(ValueError, match="speed of light"):
            unicor.shift_kms(np.array([0.0, v]), conv)
        assert np.isnan(unicor.doppler_factor(np.nan, conv))   # NaN in, NaN out


@pytest.mark.parametrize("bad", ['classical', 'CZ', 'optical', None])
def test_an_unknown_convention_is_refused(grid, bad):
    calls = (lambda: unicor.doppler_factor(1.0, bad),
             lambda: unicor.velocity_from_log_shift(1e-5, bad),
             lambda: unicor.shift_kms(1.0, bad),
             lambda: unicor.doppler_shift_wave(grid, 1.0, bad),
             lambda: _ccf(0.0, _line_spectrum(0.0, grid), grid, vmax=10,
                          velocity_convention=bad))
    for call in calls:
        with pytest.raises(ValueError, match="convention"):
            call()


def test_the_default_convention_is_cz(grid):
    for fn in (unicor.doppler_factor, unicor.velocity_from_log_shift, unicor.shift_kms,
               unicor.doppler_shift_wave):
        assert inspect.signature(fn).parameters['convention'].default == 'cz'
    param = inspect.signature(unicor.ccf_vs_velocity).parameters['velocity_convention']
    assert param.default == 'cz'
    signal = _line_spectrum(30.0, grid)
    v_default, c_default = _ccf(0.0, signal, grid, vmax=40)
    v_cz, c_cz = _ccf(0.0, signal, grid, vmax=40, velocity_convention='cz')
    np.testing.assert_array_equal(v_default, v_cz)
    np.testing.assert_array_equal(c_default, c_cz)


@pytest.mark.parametrize("conv", CONVENTIONS)
@pytest.mark.parametrize("dln", [DV / C, np.log1p(DV / C)], ids=["dv/c", "ln(1+dv/c)"])
def test_the_velocity_axis_follows_the_actual_grid_step(conv, dln):
    # log_lambda_grid steps ln(lambda) by dv/c; Spectrum.interpolate
    # by ln(1 + dv/c). The axis comes from the grid itself either way.
    wave = _log_wave(2000, dln, lam0=5040.0)
    k = 300
    ext = unicor.extend_log_grid(wave, k)
    v, _ = unicor.ccf_vs_velocity(_line_spectrum(0.0, ext), _line_spectrum(0.0, wave), wave,
                                  velocity_convention=conv)
    np.testing.assert_allclose(v, _velocity_of_lag(np.arange(-k, k + 1), dln, conv),
                               rtol=1e-11, atol=1e-15)


def test_ccf_refuses_a_wave_grid_that_does_not_match_the_signal(grid):
    flux = _line_spectrum(0.0, grid)
    ext = _line_spectrum(0.0, unicor.extend_log_grid(grid, 20))
    with pytest.raises(ValueError, match="wave"):
        unicor.ccf_vs_velocity(ext, flux, grid[:-1])
    with pytest.raises(ValueError, match="wave"):
        unicor.ccf_vs_velocity(ext, flux, DV)      # the former dv argument
    with pytest.raises(ValueError, match="uniform"):
        unicor.ccf_vs_velocity(ext, flux, np.linspace(grid[0], grid[-1], grid.size))


def _inject_and_recover(flux_fn, v0, conv, dv=DV, vmax=200.0, lam_lo=5000.0, lam_hi=5060.0):
    """Recovered minus injected velocity (m/s) for a noise-free injection."""
    dln = dv / C
    wave = _log_wave(int(np.log(lam_hi / lam_lo) / dln) + 1, dln, lam_lo)
    v_inj = _snap_to_lag(v0, dln, conv)
    observed = flux_fn(wave / _factor(v_inj, conv))
    template = flux_fn(unicor.extend_log_grid(wave, int(round(vmax / dv))))
    v, c = unicor.ccf_vs_velocity(template, observed, wave, velocity_convention=conv)
    assert c.max() == pytest.approx(1.0, abs=1e-9)     # a whole-lag shift
    rv, _ = unicor.subpixel_peak(v, c)
    return (rv - v_inj) * 1e3


@pytest.mark.parametrize("conv", CONVENTIONS)
@pytest.mark.parametrize("v0", INJECT_V0)
def test_single_line_injection_is_recovered_to_the_cm_per_s(conv, v0):
    single = lambda w: 1.0 - 0.5 * np.exp(-0.5 * ((w - 5030.0) / 0.1) ** 2)
    assert abs(_inject_and_recover(single, v0, conv)) < 0.05


@pytest.mark.parametrize("conv", CONVENTIONS)
@pytest.mark.parametrize("v0", INJECT_V0)
def test_line_forest_injection_is_recovered_to_the_cm_per_s(conv, v0):
    assert abs(_inject_and_recover(_interior_forest(), v0, conv)) < 0.05


@pytest.mark.parametrize("v0", [-150.0, -50.0, 50.0, 150.0])
def test_cz_and_relativistic_readings_differ_by_v2_over_2c(v0):
    # The same CCF read under both conventions: c (expm1(x) - tanh(x)) is
    # c x^2 / 2 + O(x^3), i.e. ~v^2/2c -- +37.5 m/s at |v| = 150 km/s, the
    # same sign for red and blue shifts.
    wave = _log_wave(int(np.log(5060.0 / 5000.0) / (DV / C)) + 1)
    forest = _interior_forest()
    observed = forest(wave / _factor(v0, 'cz'))
    template = forest(unicor.extend_log_grid(wave, 400))
    rv = {conv: unicor.subpixel_peak(*unicor.ccf_vs_velocity(
        template, observed, wave, velocity_convention=conv))[0] for conv in CONVENTIONS}
    assert rv['cz'] - rv['relativistic'] == pytest.approx(v0 ** 2 / (2 * C), rel=3e-3)


# --- Per-order atoms: one order in, its CCF and RV out -------------------------
# ccf_order is the loop body of SPARTA's CCF1d.cross_correlate; lag_axis,
# effective_bins and measure_rv the steps around it. Each is pinned here
# against the lower-level functions it composes.

def _template_arrays():
    """A line template (5050 A) on a fine log grid much wider than ``grid``."""
    t_wave, _ = unicor.log_lambda_grid(4950.0, 5200.0, 0.1)
    return t_wave, _line_spectrum(0.0, t_wave)


def test_template_coverage_of_a_wide_template_is_the_whole_order(grid):
    k = 40
    ext = unicor.extend_log_grid(grid, k)
    assert unicor.template_coverage(ext, _template_arrays()[0], k) == (0, grid.size)


def test_template_coverage_on_the_observed_span_drops_the_scan_at_each_end(grid):
    # Pixel i needs the template at nodes i .. i + 2K of the extended grid, so
    # a template on the observed span alone covers pixels K .. N - K - 1.
    k = 40
    ext = unicor.extend_log_grid(grid, k)
    assert unicor.template_coverage(ext, grid, k) == (k, grid.size - k)


def test_template_coverage_refuses_a_template_that_covers_no_pixel(grid):
    k = 40
    ext = unicor.extend_log_grid(grid, k)
    with pytest.raises(ValueError, match="does not cover any pixel"):
        unicor.template_coverage(ext, np.linspace(6000.0, 6100.0, 50), k)
    # Narrower than the scan: each pixel misses the template at some lag.
    with pytest.raises(ValueError, match="does not cover any pixel"):
        unicor.template_coverage(ext, grid[100:150], k)


@pytest.mark.parametrize("conv", CONVENTIONS)
def test_lag_axis_is_the_shared_velocity_of_each_lag(grid, conv):
    k = 30
    waves = [grid, grid[50:400]]
    dln, v = unicor.lag_axis(waves, DV, k, velocity_convention=conv)
    # The orders' steps agree to round-off; the axis uses their median.
    assert dln == np.median([unicor.log_grid_dln(w) for w in waves])
    assert dln == pytest.approx(DV / C, rel=1e-12)
    np.testing.assert_array_equal(
        v, unicor.velocity_from_log_shift(np.arange(-k, k + 1) * dln, conv))
    assert v[k] == 0.0 and v.size == 2 * k + 1


def test_lag_axis_refuses_a_grid_whose_step_is_not_the_declared_one(grid):
    with pytest.raises(ValueError, match="order 1: the wavelength grid step"):
        unicor.lag_axis([grid, unicor.log_lambda_grid(5000.0, 5100.0, 0.6)[0]], DV, 10)


def test_lag_axis_refuses_orders_whose_ln_steps_differ():
    # Within the rtol=1e-3 step check, but 100 lags apart the two axes would be
    # misregistered by 0.04 lag.
    waves = [unicor.log_lambda_grid(5000.0, 5050.0, step)[0] for step in (0.5, 0.5002)]
    with pytest.raises(ValueError, match="velocity axis"):
        unicor.lag_axis(waves, 0.5, 100)
    unicor.lag_axis(waves, 0.5, 1)


def test_lag_axis_refuses_bad_grids_conventions_and_lag_counts(grid):
    with pytest.raises(ValueError, match="uniform"):
        unicor.lag_axis([np.delete(grid, 100)], DV, 10)
    with pytest.raises(ValueError, match="convention"):
        unicor.lag_axis([grid], DV, 10, velocity_convention='classical')
    with pytest.raises(ValueError, match="n_lags"):
        unicor.lag_axis([grid], DV, 0)


def test_effective_bins_count_native_pixels():
    assert unicor.effective_bins(1000, 0.25, 1.0) == 250.0      # 4x oversampled
    assert unicor.effective_bins(1000, 1.0, 0.25) == 1000.0     # coarser than native
    assert unicor.effective_bins(1000, 0.5, 0.5) == 1000.0
    np.testing.assert_array_equal(
        unicor.effective_bins(np.array([1000, 600]), 0.5, np.array([1.0, 0.25])),
        [500.0, 600.0])


@pytest.mark.parametrize("dv, native", [(0.0, 1.0), (0.5, 0.0), (0.5, -1.0),
                                        (0.5, np.nan), (np.inf, 1.0)])
def test_effective_bins_refuse_non_positive_steps(dv, native):
    with pytest.raises(ValueError, match="positive"):
        unicor.effective_bins(1000, dv, native)


@pytest.mark.parametrize("conv", CONVENTIONS)
@pytest.mark.parametrize("v_inj", [-12.3, 0.0, 7.7])
def test_ccf_order_recovers_the_injected_rv(grid, conv, v_inj):
    t_wave, t_flux = _template_arrays()
    order = unicor.ccf_order(grid, _line_spectrum(v_inj, grid, conv), t_wave, t_flux, 80,
                             dv_kms=DV, native_step_kms=DV, velocity_convention=conv,
                             errors=True)
    assert order.rv == pytest.approx(v_inj, abs=0.05)
    assert order.peak == pytest.approx(1.0, abs=0.01)
    assert 0.0 < order.erv < 0.05
    assert (order.lo, order.hi, order.n_eff) == (0, grid.size, float(grid.size))


def test_ccf_order_is_the_composition_of_the_lower_level_functions(grid):
    # A template starting inside the order: only part of it is correlated.
    t_wave, t_flux = _template_arrays()
    cut = np.searchsorted(t_wave, 5010.0)
    t_wave, t_flux = t_wave[cut:], t_flux[cut:]
    k, native = 60, 2.0
    flux = _line_spectrum(4.2, grid) + np.random.default_rng(9).normal(0, 0.01, grid.size)
    order = unicor.ccf_order(grid, flux, t_wave, t_flux, k, dv_kms=DV,
                             native_step_kms=native, errors=True)
    ext = unicor.extend_log_grid(grid, k)
    lo, hi = unicor.template_coverage(ext, t_wave, k)
    assert lo > 0 and (order.lo, order.hi) == (lo, hi)
    v, c = unicor.ccf_vs_velocity(unicor.resample(t_wave, t_flux, ext)[lo:hi + 2 * k],
                                  flux[lo:hi], grid[lo:hi])
    n_eff = unicor.effective_bins(hi - lo, DV, native)
    rv, peak = unicor.subpixel_peak(v, c)
    assert np.array_equal(order.velocity, v) and np.array_equal(order.ccf, c)
    assert np.array_equal(order.loglike, unicor.loglike_from_ccf(c, n_eff))
    assert (order.rv, order.peak, order.n_eff) == (rv, peak, n_eff)
    assert order.erv == unicor.rv_uncertainty(v, c, rv, n_eff)


def test_ccf_order_without_errors_leaves_the_uncertainty_nan(grid):
    t_wave, t_flux = _template_arrays()
    order = unicor.ccf_order(grid, _line_spectrum(3.0, grid), t_wave, t_flux, 40,
                             dv_kms=DV, native_step_kms=DV)
    assert np.isnan(order.erv) and np.isfinite(order.rv)


def test_ccf_order_beyond_the_scan_is_nan_but_keeps_its_ccf(grid):
    # The peak lies outside +-20 km/s: no RV, but the CCF itself is valid.
    t_wave, t_flux = _template_arrays()
    order = unicor.ccf_order(grid, _line_spectrum(-35.0, grid), t_wave, t_flux, 40,
                             dv_kms=DV, native_step_kms=DV, errors=True)
    assert np.isnan(order.rv) and np.isnan(order.erv) and np.isnan(order.peak)
    assert np.all(np.isfinite(order.ccf)) and order.ccf.size == 81


def test_ccf_order_refuses_an_order_the_template_does_not_cover(grid):
    t_wave = np.linspace(6000.0, 6100.0, 500)
    with pytest.raises(ValueError, match="does not cover any pixel"):
        unicor.ccf_order(grid, _line_spectrum(0.0, grid), t_wave, np.ones(500), 40,
                         dv_kms=DV, native_step_kms=DV)


def test_ccf_order_result_is_frozen(grid):
    t_wave, t_flux = _template_arrays()
    order = unicor.ccf_order(grid, _line_spectrum(0.0, grid), t_wave, t_flux, 20,
                             dv_kms=DV, native_step_kms=DV)
    with pytest.raises(AttributeError):
        order.rv = 1.0


def test_measure_rv_is_the_subpixel_peak_and_its_ml_uncertainty(grid):
    v, c = _ccf(0.0, _line_spectrum(5.5, grid), grid, vmax=30)
    m = unicor.measure_rv(v, c, 812.0)
    rv, peak = unicor.subpixel_peak(v, c)
    assert (m.rv, m.peak) == (rv, peak)
    assert m.erv == unicor.rv_uncertainty(v, c, rv, 812.0)


def test_measure_rv_refuses_a_peak_it_cannot_locate(grid):
    v, c = _ccf(0.0, _line_spectrum(-35.0, grid), grid, vmax=20)
    with pytest.raises(ValueError, match="edge"):
        unicor.measure_rv(v, c, 812.0)
    with pytest.raises(ValueError, match="NaN"):
        unicor.measure_rv(v, np.full_like(c, np.nan), 812.0)


# --- the continuous CCF peak: a local quintic spline of the lag samples ------
# The CCF of adequately sampled spectra is band-limited, so its values at the
# lags determine it between them; the RV is the maximum of that interpolant,
# not of a parabola. The reference is the continuous CCF itself: the Pearson
# correlation of the observed pixels with the exact template at any shift.

_FOREST_RNG = np.random.default_rng(1)
_CENTRES = _FOREST_RNG.uniform(5005.0, 5055.0, 120)
_DEPTHS = _FOREST_RNG.uniform(0.1, 0.7, 120)


def _forest(wave, resolution=100_000):
    """120 Gaussian lines at the instrumental width c/R (FWHM), no rotation."""
    sigma = (C / resolution) / 2.3548 / C * _CENTRES                  # Angstrom
    return 1 - np.sum(_DEPTHS * np.exp(-0.5 * ((wave[:, None] - _CENTRES) / sigma) ** 2),
                      axis=1)


def _forest_ccf(v_true, dv, half_width=20.0, resolution=100_000):
    wave, _ = unicor.log_lambda_grid(5010.0, 5050.0, dv)
    k = int(round(half_width / dv))
    observed = _forest(wave / unicor.doppler_factor(v_true), resolution)
    v, c = unicor.ccf_vs_velocity(_forest(unicor.extend_log_grid(wave, k), resolution),
                                  observed, wave)
    return wave, observed, v, c


def _continuous_ccf(wave, observed, v, resolution=100_000):
    """The reference: the exact template shifted by any v, correlated over the same pixels."""
    return np.corrcoef(observed, _forest(wave / unicor.doppler_factor(v), resolution))[0, 1]


@pytest.mark.parametrize("dv, tolerance_ms", [(0.75, 0.5), (0.5, 0.15)])
def test_the_peak_does_not_depend_on_where_it_falls_between_lags(dv, tolerance_ms):
    # The 5-point parabola misread this forest by up to 22 m/s at dv = 0.75 km/s
    # and 7 m/s at 0.5, depending on the true shift's phase between two lags.
    errors = []
    for phase in np.linspace(0.0, 1.0, 21):
        v_true = 3.0 + phase * dv
        _, _, v, c = _forest_ccf(v_true, dv)
        errors.append(unicor.subpixel_peak(v, c)[0] - v_true)
    assert np.max(np.abs(errors)) * 1e3 < tolerance_ms


def test_the_rv_does_not_depend_on_the_ccf_step():
    rvs = [unicor.subpixel_peak(*_forest_ccf(3.37, dv)[2:])[0] for dv in (0.75, 0.5, 0.25)]
    assert np.ptp(rvs) * 1e3 < 0.5                                           # m/s


def test_the_continuous_ccf_reference_reproduces_the_lag_samples():
    # A consistency check of the reference used below: at whole lags it is the CCF.
    wave, observed, v, c = _forest_ccf(3.37, 0.5)
    for i in (0, len(v) // 3, len(v) // 2, len(v) - 1):
        assert _continuous_ccf(wave, observed, v[i]) == pytest.approx(c[i], abs=1e-10)


def test_combined_orders_peak_where_the_summed_loglike_of_the_continuous_ccfs_does():
    # Zucker's combination is monotone in the summed log-likelihood, so the
    # combined CCF's spline peak must sit at the maximum of the sum of the
    # orders' continuous log-likelihoods.
    from scipy import optimize
    dv, v_true = 0.5, 3.37
    orders, n_eff = [], np.array([800.0, 1300.0])
    for lo, hi, res in ((5010.0, 5030.0, 100_000), (5030.0, 5050.0, 60_000)):
        wave, _ = unicor.log_lambda_grid(lo, hi, dv)
        observed = _forest(wave / unicor.doppler_factor(v_true), res)
        v, c = unicor.ccf_vs_velocity(_forest(unicor.extend_log_grid(wave, 40), res),
                                      observed, wave)
        orders.append((wave, observed, res, c))
    combined = unicor.combine_ccfs(np.vstack([o[3] for o in orders]), n_eff)
    rv = unicor.subpixel_peak(v, combined)[0]

    def summed_loglike(x):
        return sum(-0.5 * n * np.log(1 - _continuous_ccf(w, obs, x, res) ** 2)
                   for (w, obs, res, _), n in zip(orders, n_eff))
    best = optimize.minimize_scalar(lambda x: -summed_loglike(x), bounds=(rv - dv, rv + dv),
                                    method="bounded", options=dict(xatol=1e-9)).x
    assert abs(rv - best) * 1e3 < 0.3                                        # m/s


def test_the_error_comes_from_the_curvature_at_the_refined_peak():
    # On a Gaussian CCF sampled off its centre, the closed-form Zucker error
    # holds at the spline peak, wherever the peak falls between lags.
    A, s, n_eff = 0.8, 4.0, 2500.0
    v = np.arange(-20.0, 20.01, 0.75)
    ccf = A * np.exp(-0.5 * ((v - 0.31) / s) ** 2)
    rv, _ = unicor.subpixel_peak(v, ccf)
    expected = s * np.sqrt(1 - A ** 2) / (A * np.sqrt(n_eff))
    assert rv == pytest.approx(0.31, abs=1e-5)
    assert unicor.rv_uncertainty(v, ccf, rv, n_eff) == pytest.approx(expected, rel=2e-3)


def test_an_undersampled_ccf_is_flagged():
    # Fewer than ~5 lags across the CCF's half-maximum width: no interpolant
    # can recover the peak to a few m/s, so the caller is told.
    v = np.arange(-20.0, 20.01, 1.0)
    narrow = np.exp(-0.5 * ((v - 0.3) / 1.0) ** 2)            # FWHM 2.4 lags
    with pytest.warns(UserWarning, match="undersampled"):
        unicor.subpixel_peak(v, narrow)
    wide = np.exp(-0.5 * ((v - 0.3) / 4.0) ** 2)              # FWHM 9.4 lags
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        unicor.subpixel_peak(v, wide)


def test_the_error_at_a_velocity_away_from_the_peak_is_not_extrapolated():
    # CCF1d.extract_rv(vel=...) asks for the error at an imposed velocity: the
    # curvature there must come from a spline around that velocity, not from
    # the peak's spline extrapolated 7 lags away (that gave 0.48 for 0.34 km/s).
    A, s, n_eff, v0 = 0.8, 4.0, 2500.0, 0.3
    v = np.arange(-30.0, 30.01, 0.5)
    ccf = A * np.exp(-0.5 * ((v - v0) / s) ** 2)
    at = 4.0
    c = A * np.exp(-0.5 * ((at - v0) / s) ** 2)
    c2 = c * (((at - v0) / s ** 2) ** 2 - 1 / s ** 2)
    expected = np.sqrt(-1.0 / (n_eff * c2 * c / (1 - c ** 2)))
    assert unicor.rv_uncertainty(v, ccf, at, n_eff) == pytest.approx(expected, rel=1e-3)


def test_a_scan_too_short_for_the_peak_spline_is_refused_clearly():
    v = np.arange(-2.0, 2.01, 1.0)                     # 5 lags: n_lags = 2
    with pytest.raises(ValueError, match="at least 6"):
        unicor.subpixel_peak(v, np.exp(-0.5 * (v / 2.0) ** 2))


def test_the_undersampling_warning_does_not_depend_on_the_phase():
    # The width comes from the spline's curvature, so a peak near the limit
    # warns (or not) at every sub-lag phase alike.
    for sigma_lags, should_warn in ((1.5, True), (3.0, False)):
        for phase in np.linspace(0.0, 1.0, 11):
            v = np.arange(-20.0, 20.01, 1.0)
            ccf = 0.9 * np.exp(-0.5 * ((v - phase) / sigma_lags) ** 2)
            import warnings
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                unicor.subpixel_peak(v, ccf)
            assert any("undersampled" in str(w.message) for w in caught) == should_warn


def test_a_noise_fraction_scales_the_independent_bins_and_the_error():
    # Preprocessing that filters away part of the noise (fraction g kept)
    # counts g of the native bins: the error grows by 1/sqrt(g).
    wave, observed, _, _ = _forest_ccf(3.37, 0.5)
    rng = np.random.default_rng(0)
    noisy = observed + rng.normal(0, 0.01, observed.size)
    ext = unicor.extend_log_grid(wave, 40)
    plain = unicor.ccf_order(wave, noisy, ext, _forest(ext), 40, dv_kms=0.5,
                             native_step_kms=0.5, errors=True)
    kept = unicor.ccf_order(wave, noisy, ext, _forest(ext), 40, dv_kms=0.5,
                            native_step_kms=0.5, errors=True, noise_fraction=0.25)
    assert kept.n_eff == pytest.approx(0.25 * plain.n_eff)
    assert kept.erv == pytest.approx(2.0 * plain.erv, rel=1e-12)
    assert kept.rv == plain.rv
    with pytest.raises(ValueError, match="noise_fraction"):
        unicor.ccf_order(wave, noisy, ext, _forest(ext), 40, dv_kms=0.5,
                         native_step_kms=0.5, noise_fraction=1.5)


def test_parseval_weights_match_real_space():
    rng = np.random.default_rng(0)
    for n in (256, 257):
        x = rng.standard_normal(n)
        x -= x.mean()  # DC removed, as in the fit
        X = np.fft.rfft(x)[1:]
        w = unicor.parseval_weights(n, sigma=1.0)
        assert np.sum(w * np.abs(X) ** 2) == pytest.approx(np.sum(x ** 2), rel=1e-9)
