"""Correctness tests for the reimplemented CCF / RV extraction.

The legacy CrossCorrelateSpec could not recover non-negative radial velocities
(see KNOWN_TRAPS.md): it truncated the template to a shifted sub-grid and used
a constant (not per-lag) normalization, so only RV <~ 0 was recoverable. Its
first replacement correlated against a template on the observed span only, so
fewer pixels overlapped at larger lags and the peak was pulled toward v = 0.
CrossCorrelateSpec now resamples the template onto the observed log-lambda grid
extended by the scan half-width on each side, so every lag is a Pearson
correlation over the same observed pixels. The golden master is physical:
recovered RV must equal the injected RV, for both signs, with no pull to zero.
"""
import numpy as np
import pytest
from astropy import units as u

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
    x = np.arange(n) * dv

    def line(cv):
        return 1.0 - 0.6 * np.exp(-0.5 * ((x - n // 2 * dv - cv) / 3.0) ** 2)

    k = 80   # lags: +-40 km/s; the template's extension is pure continuum
    template = np.concatenate([np.ones(k), line(0.0), np.ones(k)])
    v, ccf = so.ccf_vs_velocity(template, line(inj), dv)
    assert v[np.argmax(ccf)] == pytest.approx(inj, abs=dv)
    assert ccf.max() == pytest.approx(1.0, abs=1e-12)   # every lag: all n pixels


def test_ccf_vs_velocity_rejects_a_template_on_the_observed_span_only():
    with pytest.raises(ValueError, match="2K"):
        so.ccf_vs_velocity(np.ones(12), np.ones(12), 0.5)


def test_rv_beyond_the_scan_is_nan_not_a_wrong_sign(template):
    # -70 km/s with a +-50 km/s scan: the legacy parabola window wrapped around
    # the array and reported about +2.8 km/s, silently.
    obs = Spectrum(wv=[WAVE.copy()], sp=[_line(5500.0 * (1 - 70.0 / C))], name="o")
    ccf = CCF1d().CrossCorrelateSpec(spec_in=obs, template_in=template,
                                     dv=0.5, VelBound=50).CombineCCFs()
    assert np.isnan(float(ccf.Corr["RV"][0, 0]))
    assert np.isnan(float(np.asarray(ccf.CorrCombined["RV"])))


def test_unitless_velbound_is_km_per_s_whatever_the_unit_of_dv(template):
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
    # Order 0 lies outside the template's span, so no observed pixel has
    # template coverage at every lag; that order is NaN and order 1 still
    # yields its RV.
    tmpl = Template(template=Spectrum(wv=[WAVE.copy()], sp=[_line(5500.0)], name="t"))
    far = np.linspace(6000.0, 6040.0, 4000)
    obs = Spectrum(wv=[far, WAVE.copy()],
                   sp=[np.ones_like(far), _line(5500.0 * (1 + 4.0 / C))], name="o")
    with pytest.warns(UserWarning, match="order 0: .*cover"):
        ccf = CCF1d().CrossCorrelateSpec(spec_in=obs, template_in=tmpl, dv=0.5, VelBound=50)
    assert np.isnan(ccf.Corr["RV"][0, 0])
    assert float(ccf.Corr["RV"][1, 0]) == pytest.approx(4.0, abs=0.05)


def _log_spectrum(wave, flux, dv):
    """One-order Spectrum declared to be on a log-lambda grid of step dv (km/s)."""
    spec = Spectrum(wv=[wave.copy()], sp=[flux], name="o")
    spec.Info["GridType"], spec.Info["GridDelta"] = "log", dv * u.km / u.s
    return spec


@pytest.mark.parametrize("inj", [-5.0, 5.0])
def test_a_template_on_the_observed_span_trims_the_order_by_the_scan(inj):
    # Pixels within VelBound of either end have no template at every lag, so
    # they are left out of the correlation (and of n_eff) instead of being
    # correlated against a clamped, line-free template.
    dv, vel_bound = 0.5, 50.0
    wave, _ = so.log_lambda_grid(5480.0, 5520.0, dv)
    line = lambda centre: 1.0 - 0.6 * np.exp(-0.5 * ((wave - centre) / 0.12) ** 2)
    tmpl = Template(template=Spectrum(wv=[wave.copy()], sp=[line(5500.0)], name="t"))
    obs = _log_spectrum(wave, line(5500.0 * (1 + inj / C)), dv)
    ccf = CCF1d().CrossCorrelateSpec(spec_in=obs, template_in=tmpl, VelBound=vel_bound)
    # dv_from_wave's native step is c*expm1(dln), a hair above dv: rel 1e-6.
    assert ccf.Corr["n_eff"][0] == pytest.approx(wave.size - 2 * round(vel_bound / dv),
                                                 rel=1e-5)
    assert float(ccf.Corr["RV"][0, 0]) == pytest.approx(inj, abs=0.05)


def test_a_template_covering_part_of_the_order_uses_only_the_covered_pixels():
    dv, vel_bound = 0.5, 30.0
    k = round(vel_bound / dv)
    wave, _ = so.log_lambda_grid(5480.0, 5520.0, dv)
    t_wave, _ = so.log_lambda_grid(5470.0, 5500.0, 0.25)
    tmpl = Template(template=Spectrum(wv=[t_wave], sp=[_forest(t_wave, 0.0)], name="t"))
    obs = _log_spectrum(wave, _forest(wave, 3.0), dv)
    ccf = CCF1d().CrossCorrelateSpec(spec_in=obs, template_in=tmpl, VelBound=vel_bound)
    dln = np.log(wave[-1] / wave[0]) / (wave.size - 1)
    covered = ((wave * np.exp(-k * dln) >= t_wave[0]) &
               (wave * np.exp(k * dln) <= t_wave[-1]))
    assert 0 < covered.sum() < wave.size / 2
    assert ccf.Corr["n_eff"][0] == pytest.approx(covered.sum(), abs=1)
    assert float(ccf.Corr["RV"][0, 0]) == pytest.approx(3.0, abs=0.05)


def _rotating_forest(vsini, seed, lam_lo=5000.0, lam_hi=5060.0, fine_dv=0.05):
    """~200 Gaussian lines per 60 A at R = 100 000, rotationally broadened.

    Rest frame, normalised, on a fine log-lambda grid (step ``fine_dv`` km/s)
    with a 10 A margin on each side of ``[lam_lo, lam_hi]`` (Angstrom).
    """
    wave, x = so.log_lambda_grid(lam_lo - 10.0, lam_hi + 10.0, fine_dv)
    rng = np.random.default_rng(seed)
    n_lines = int(round(200 * (wave[-1] - wave[0]) / (lam_hi - lam_lo)))
    centres, depths = rng.uniform(x[0], x[-1], n_lines), rng.uniform(0.05, 0.6, n_lines)
    half = int(12 / fine_dv)
    flux = np.ones_like(wave)
    for xc, depth in zip(centres, depths):
        sl = slice(max(0, int(xc / fine_dv) - half), int(xc / fine_dv) + half + 2)
        flux[sl] *= 1 - depth * np.exp(-0.5 * ((x[sl] - xc) / 1.5) ** 2)
    flux = so.broaden_gaussian(flux, fine_dv, C / 100_000)
    return wave, so.broaden_rotational(flux, fine_dv, vsini)


def test_rotating_star_rv_has_no_pull_toward_zero():
    # vsini 25 km/s, one 60 A order: the same-span CCF this replaced was
    # pulled toward v = 0 by 40-370 m/s on this forest. The shift is a whole
    # number k0 of grid steps, so the expected peak is exactly k0 * dv (the
    # parabola has no pixel-phase error there; the lag -> Doppler velocity
    # mapping is a separate convention).
    dv = 0.75
    wave, _ = so.log_lambda_grid(5000.0, 5060.0, dv)
    dln = np.log(wave[-1] / wave[0]) / (wave.size - 1)
    t_wave, t_flux = _rotating_forest(25.0, seed=1)
    tmpl = Template(template=Spectrum(wv=[t_wave], sp=[t_flux], name="t"))
    v0s = np.array([-50.0, -20.0, -5.0, 5.0, 20.0, 50.0])
    errors = []
    for v0 in v0s:
        k0 = round(np.log1p(v0 / C) / dln)
        flux = so.resample(t_wave, t_flux, wave * np.exp(-k0 * dln), kind="cubic")
        ccf = CCF1d().CrossCorrelateSpec(spec_in=_log_spectrum(wave, flux, dv),
                                         template_in=tmpl, VelBound=100)
        errors.append((float(ccf.Corr["RV"][0, 0]) - k0 * dv) * 1e3)   # m/s
    errors = np.array(errors)
    np.testing.assert_array_less(np.abs(errors), 1.5)
    assert abs(np.mean(np.sign(v0s) * errors)) < 0.5


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


def test_effective_bins_follow_an_intermediate_coarser_resampling(forest_template):
    # A 3x coarser linear resampling before the CCF leaves ~1/3 of the native
    # pixels independent; n_eff used to count the original ones (2166 vs ~727).
    obs = Spectrum(wv=[MC_WAVE.copy()], sp=[_forest(MC_WAVE, 0.0)], name="o")
    obs.InterpolateSpectrum(delta=3.0)
    ccf = CCF1d().CrossCorrelateSpec(spec_in=obs, template_in=forest_template,
                                     dv=0.25, VelBound=20)
    assert ccf.Corr["n_eff"][0] == pytest.approx(MC_WAVE.size / 3.0, rel=0.02)


def test_ccf1d_refuses_native_steps_that_do_not_match_the_orders(forest_template):
    obs = Spectrum(wv=[MC_WAVE.copy()], sp=[_forest(MC_WAVE, 0.0)], name="o")
    obs.Info["NativeStep"] = [1.0, 1.0]
    with pytest.raises(ValueError, match="NativeStep"):
        CCF1d().CrossCorrelateSpec(spec_in=obs, template_in=forest_template, VelBound=20)
