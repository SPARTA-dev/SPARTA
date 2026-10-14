"""Tests for the array-level UNICOR engine (:mod:`unicor.ccf`).

Synthetic single-line spectra on a log-lambda grid: the CCF peak lands on the
injected shift, identical orders combine to the single-order CCF, and the
sub-pixel parabola is exact on a parabola. No external data.
"""
import numpy as np
import pytest

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


@pytest.mark.parametrize("v_inj", [-12.3, 0.0, 7.7])
def test_ccf_peak_recovers_injected_shift(grid, v_inj):
    v, c = unicor.ccf_vs_velocity(_line_spectrum(0.0, grid),
                                  _line_spectrum(v_inj, grid), DV, vmax=40)
    rv, peak = unicor.subpixel_peak(v, c)
    assert rv == pytest.approx(v_inj, abs=0.05)
    assert peak == pytest.approx(1.0, abs=0.01)


def test_ccf_lags_are_symmetric_and_normalized(grid):
    flux = _line_spectrum(0.0, grid)
    v, c = unicor.ccf_vs_velocity(flux, flux, DV, vmax=10)
    np.testing.assert_allclose(v, -v[::-1])
    assert v[1] - v[0] == pytest.approx(DV)
    assert c.max() == pytest.approx(1.0) and v[np.argmax(c)] == 0.0


def test_combining_identical_orders_returns_the_single_order_ccf(grid):
    v, c = unicor.ccf_vs_velocity(_line_spectrum(0.0, grid),
                                  _line_spectrum(3.0, grid), DV, vmax=20)
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


def test_ccf_rejects_mismatched_grids():
    with pytest.raises(ValueError):
        unicor.ccf_vs_velocity(np.ones(10), np.ones(12), DV, 5)


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
    with pytest.raises(ValueError, match="constant"):
        unicor.ccf_vs_velocity(np.ones_like(flux), flux, DV, 10)
    bad = flux.copy()
    bad[10] = np.nan
    with pytest.raises(ValueError, match="finite"):
        unicor.ccf_vs_velocity(flux, bad, DV, 10)


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
    ccfs = [unicor.ccf_vs_velocity(_line_spectrum(0.0, grid),
                                   _line_spectrum(s, grid), DV, vmax=20)[1]
            for s in shifts]
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
