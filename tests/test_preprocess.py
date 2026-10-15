"""The per-order preprocessing atoms (:mod:`sparta.preprocess`).

Three layers:

* each atom on plain arrays: units (Angstrom, km/s), edge cases, errors;
* the ``Spectrum`` methods are loops over the atoms -- one call per order,
  spied through the names ``sparta.spectrum`` (and ``sparta.observations``,
  ``sparta.template``) import;
* the atoms, chained by hand, rebuild the object's result exactly
  (``np.array_equal``, not a tolerance).

Self-contained: no external data.
"""
import copy

import numpy as np
import pytest
from astropy import constants as consts, units as u
from scipy import signal

from unicor import dsp
from sparta import observations as observations_module
from sparta import preprocess
from sparta import spectrum as spectrum_module
from sparta import template as template_module
from sparta.instruments import Simulation
from sparta.observations import Observations
from sparta.preprocess import (
    bandpass, coarsest_native_step, remove_cosmics, resample_flux, resample_to_linear_grid,
    resample_to_log_grid, trim_margins,
)
from sparta.spectrum import Spectrum
from sparta.template import Template

C_KMS = dsp.C_KMS


def _lines(wave, v=0.0):
    """Three Gaussian absorption lines (Angstrom), Doppler-shifted by v (km/s)."""
    rest = np.asarray(wave, float) / (1.0 + v / C_KMS)
    flux = np.ones_like(rest)
    for centre, depth in ((5003.0, 0.5), (5010.0, 0.3), (5016.5, 0.6)):
        flux *= 1.0 - depth * np.exp(-0.5 * ((rest - centre) / 0.2) ** 2)
    return flux


def _raw_order(seed=0, n=2000, lo=5000.0, hi=5020.0, pad=(20, 25)):
    """A blazed, noisy order with noisy near-zero margins and two cosmics."""
    rng = np.random.default_rng(seed)
    wave = np.linspace(lo, hi, n)
    flux = 500.0 * _lines(wave, 12.0) + rng.normal(0.0, 3.0, n)
    flux[:pad[0]] = rng.normal(0.0, 3.0, pad[0])
    flux[n - pad[1]:] = rng.normal(0.0, 3.0, pad[1])
    flux[[700, 1300]] += 400.0
    return wave, flux


def _raw_spectrum(n_ord=3):
    wv, sp = [], []
    for k in range(n_ord):
        w, s = _raw_order(seed=k, lo=5000.0 + 30.0 * k, hi=5020.0 + 30.0 * k)
        wv.append(w)
        sp.append(s)
    sp[0][[50, 51]] = np.nan
    return Spectrum(wv=wv, sp=sp, name="atoms")


# =============================================================================
# resample_flux: moved, still importable from sparta.spectrum
# =============================================================================
def test_resample_flux_lives_in_preprocess_and_resolves_from_spectrum():
    assert spectrum_module.resample_flux is preprocess.resample_flux


def test_resample_flux_reproduces_a_cubic_and_is_nan_outside_the_span():
    wave = np.linspace(5000.0, 5010.0, 101)
    flux = 1.0 + 1e-3 * (wave - 5005.0) ** 3
    new = np.array([5000.5, 5004.25, 5009.9, 5011.0])
    out = resample_flux(wave, flux, new)
    np.testing.assert_allclose(out[:3], 1.0 + 1e-3 * (new[:3] - 5005.0) ** 3, rtol=1e-9)
    assert np.isnan(out[3])


# =============================================================================
# trim_margins
# =============================================================================
def test_trim_margins_counts_pixels_from_each_end():
    wave = np.arange(100.0) + 5000.0
    flux = np.ones(100)
    w, s = trim_margins(wave, flux, n_left=3, n_right=5)
    np.testing.assert_array_equal(w, wave[3:95])
    np.testing.assert_array_equal(s, flux[3:95])


def test_trim_margins_drops_nans_first_or_keeps_them():
    wave = np.arange(10.0) + 5000.0
    flux = np.ones(10)
    flux[[0, 4]] = np.nan
    w, s = trim_margins(wave, flux, n_left=1)
    np.testing.assert_array_equal(w, wave[[2, 3, 5, 6, 7, 8, 9]])
    assert np.all(np.isfinite(s))
    w, s = trim_margins(wave, flux, remove_nans=False)
    np.testing.assert_array_equal(w, wave)
    assert np.isnan(s[4])


def test_trim_margins_removes_noisy_padding_and_keeps_the_last_exposed_pixel():
    wave, flux = _raw_order(pad=(20, 25))
    w, s = trim_margins(wave, flux, clean_margins=True)
    # The running median catches the padding to within half its window.
    assert 12 <= np.flatnonzero(wave == w[0])[0] <= 28
    assert 12 <= wave.size - 1 - np.flatnonzero(wave == w[-1])[0] <= 33
    assert np.median(s[:20]) > 100.0 and np.median(s[-20:]) > 100.0
    w2, _ = trim_margins(wave, flux, n_left=4, n_right=6, clean_margins=True)
    np.testing.assert_array_equal(w2, w[4:-6])


def test_trim_margins_refuses_an_order_without_flux_above_the_margin_level():
    wave = np.arange(50.0) + 5000.0
    with pytest.raises(ValueError, match="margin level"):
        trim_margins(wave, np.zeros(50), clean_margins=True)


# =============================================================================
# resample_to_linear_grid / resample_to_log_grid
# =============================================================================
def test_resample_to_linear_grid_steps_in_units_of_the_median_step():
    wave = np.linspace(5000.0, 5020.0, 2001)                # 0.01 A
    flux = 2.0 + 0.1 * (wave - 5010.0)
    w, s = resample_to_linear_grid(wave, flux, 0.5)
    np.testing.assert_allclose(np.diff(w), 0.005, rtol=1e-9)
    assert w[0] == pytest.approx(5000.005) and w[-1] < 5020.0 - 0.005 + 1e-9
    np.testing.assert_allclose(s, 2.0 + 0.1 * (w - 5010.0), rtol=1e-12)


def test_resample_to_log_grid_is_velocity_even_with_astropys_c():
    wave = np.linspace(5000.0, 5020.0, 2001)
    w, s = resample_to_log_grid(wave, _lines(wave), 0.5)
    step = np.log(1 + 0.5 / consts.c.to('km/s').value)
    np.testing.assert_allclose(np.diff(np.log(w)), step, rtol=1e-6)
    assert w[0] == pytest.approx(5000.01) and w[-1] <= 5020.0 - 0.01
    assert np.all(np.isfinite(s))
    assert s.min() == pytest.approx(0.4, abs=0.01)           # deepest line kept


def test_resample_to_log_grid_is_the_spectrum_formula_bit_for_bit():
    wave = np.linspace(5000.0, 5020.0, 2001)
    med = np.median(np.diff(wave))
    beta = ((2.5 * u.km / u.s / consts.c.to('km/s')).decompose()).value
    expected = np.exp(np.arange(np.log(wave.min() + med), np.log(wave.max() - med),
                                np.log(1 + beta)))
    w, _ = resample_to_log_grid(wave, _lines(wave), 2.5)
    np.testing.assert_array_equal(w, expected)


@pytest.mark.parametrize("atom, step", [(resample_to_linear_grid, 0.5),
                                        (resample_to_log_grid, 1.0)])
def test_resampling_refuses_an_order_of_fewer_than_two_pixels(atom, step):
    with pytest.raises(ValueError, match="2 pixels"):
        atom(np.array([5000.0]), np.array([1.0]), step)


# =============================================================================
# coarsest_native_step
# =============================================================================
def test_coarsest_native_step_keeps_the_coarser_sampling():
    grid = dsp.log_lambda_grid(5000.0, 5010.0, 2.0)[0]          # 2 km/s
    assert coarsest_native_step(3.0, grid) == 3.0
    # unicor's grid steps ln(lambda) by dv/c, so c d(lambda)/lambda = c expm1(dv/c).
    assert coarsest_native_step(0.5, grid) == pytest.approx(2.0, rel=1e-5)
    assert coarsest_native_step(0.5, grid) == dsp.dv_from_wave(grid)


# =============================================================================
# remove_cosmics
# =============================================================================
@pytest.mark.filterwarnings("ignore:Input data contains invalid values")
def test_remove_cosmics_interpolates_upward_outliers_only():
    rng = np.random.default_rng(1)
    wave = np.linspace(5000.0, 5010.0, 500)
    flux = 1.0 + rng.normal(0.0, 0.01, 500)
    flux[[100, 300]] = [3.0, -2.0]
    flux[200] = np.nan
    before = flux.copy()
    out = remove_cosmics(wave, flux, sigma_upper=5)
    np.testing.assert_array_equal(flux[np.isfinite(flux)], before[np.isfinite(before)])
    assert out[100] == pytest.approx(np.interp(wave[100], wave[[99, 101]], flux[[99, 101]]))
    assert out[300] == -2.0 and np.isnan(out[200])
    keep = np.ones(500, bool)
    keep[[100, 200]] = False
    np.testing.assert_array_equal(out[keep], flux[keep])


# =============================================================================
# bandpass and the taper
# =============================================================================
def test_bandpass_standardizes_and_removes_the_slow_trend():
    wave = np.linspace(5000.0, 5020.0, 4000)
    flux = 500.0 * (1.0 + 0.3 * (wave - 5010.0) / 10.0) * _lines(wave)
    out = bandpass(wave, flux, lowcut=3, highcut=0.15)
    assert out.mean() == pytest.approx(0.0, abs=1e-12)
    assert out.std() == pytest.approx(1.0, rel=1e-12)
    # The continuum slope is gone (the input correlates with it at 0.83);
    # the lines are what is left.
    assert abs(np.corrcoef(out, wave - 5010.0)[0, 1]) < 0.05
    lines_only = bandpass(wave, 500.0 * _lines(wave), lowcut=3, highcut=0.15)
    assert np.corrcoef(out, lines_only)[0, 1] > 0.95


def test_the_taper_is_unicors_tukey_window_bit_for_bit():
    flux = np.random.default_rng(2).normal(size=1001)
    for alpha in (0.0, 0.1, 0.3, 1.0):
        window = dsp.tukey_window(flux.size, alpha)
        np.testing.assert_array_equal(window, signal.windows.tukey(flux.size, alpha))
        np.testing.assert_array_equal(flux * window, np.multiply(flux, window))


# =============================================================================
# The Spectrum methods are loops over the atoms
# =============================================================================
def _spy(monkeypatch, module, name):
    calls = []
    real = getattr(module, name)

    def wrapper(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapper)
    return calls


@pytest.mark.parametrize("method, kwargs, atoms", [
    ("trim", dict(n_trim=10, clean_margins=True), ["trim_margins"]),
    ("interpolate", {}, ["resample_to_linear_grid", "coarsest_native_step"]),
    ("interpolate", dict(delta=1.0, method="log"),
     ["resample_to_log_grid", "coarsest_native_step"]),
    ("remove_cosmics", {}, ["remove_cosmics"]),
    ("filter_spectrum", dict(lowcut=3, highcut=0.15, order=1), ["bandpass"]),
    ("preprocess", {}, ["trim_margins", "resample_to_linear_grid", "coarsest_native_step",
                        "remove_cosmics", "bandpass"]),
])
def test_spectrum_methods_call_each_atom_once_per_order(monkeypatch, method, kwargs, atoms):
    spec = _raw_spectrum()
    if method in ("interpolate", "remove_cosmics", "filter_spectrum"):
        spec.trim(clean_margins=True)
    if method in ("remove_cosmics", "filter_spectrum"):
        spec.interpolate()
    calls = {name: _spy(monkeypatch, spectrum_module, name) for name in atoms}
    getattr(spec, method)(**kwargs)
    assert {name: len(c) for name, c in calls.items()} == {name: 3 for name in atoms}


@pytest.mark.parametrize("method, kwargs", [("apply_cosine_bell", dict(alpha=0.3)),
                                            ("preprocess", {})])
def test_the_taper_is_one_tukey_window_per_order(monkeypatch, method, kwargs):
    spec = _raw_spectrum()
    calls = _spy(monkeypatch, spectrum_module.dsp, "tukey_window")
    getattr(spec, method)(**kwargs)
    assert [c[0] for c in calls] == [len(w) for w in spec.wv]


def test_ingest_and_integrate_spec_use_coarsest_native_step(monkeypatch):
    # An archive product seeds the native (detector) step on ingestion.
    sim = Simulation(like="HARPS", components=[dict(teff=5800, logg=3.5, feh=0.5)],
                     times=np.array([0.0]), rv_kms=np.array([0.0]), snr=None,
                     windows_A=[(5000.0, 5010.0), (5500.0, 5510.0)], as_product=True)
    calls = _spy(monkeypatch, observations_module, "coarsest_native_step")
    obs = Observations(survey="SIMULATED", simulation=sim)
    assert len(calls) == 2
    native = sim.product.instrument.native_step_kms
    assert obs.spec_list[0].native_steps() == [
        max(native, dsp.dv_from_wave(w)) for w in obs.spec_list[0].wv]

    calls = _spy(monkeypatch, template_module, "coarsest_native_step")
    wave = np.linspace(5000.0, 5010.0, 1001)
    tmpl = Template(Spectrum(wv=[wave], sp=[_lines(wave)]).interpolate(delta=0.5))
    tmpl.integrate_spec(3)
    assert len(calls) == 1


# =============================================================================
# The atoms rebuild the object's result exactly
# =============================================================================
def test_atoms_rebuild_the_full_preprocess_chain_exactly():
    spec = _raw_spectrum()
    obj = copy.deepcopy(spec).preprocess()

    for I, (w, s) in enumerate(zip(spec.wv, spec.sp)):
        w, s = trim_margins(w, s, n_left=10, n_right=10, clean_margins=True)
        native = dsp.dv_from_wave(w)
        w, s = resample_to_linear_grid(w, s, 0.5)
        native = coarsest_native_step(native, w)
        s = remove_cosmics(w, s, sigma_upper=3)
        s = bandpass(w, s, lowcut=3, highcut=0.15, order=1)
        s = s * dsp.tukey_window(len(s), 0.3)
        assert np.array_equal(w, obj.wv[I])
        assert np.array_equal(s, obj.sp[I])
        assert native == obj.native_steps()[I]


@pytest.mark.parametrize("delta", [0.1, 2.5 * u.km / u.s, 800.0 * u.m / u.s])
def test_atoms_rebuild_the_log_resampling_exactly(delta):
    spec = _raw_spectrum().trim(clean_margins=True)
    obj = copy.deepcopy(spec).interpolate(delta=delta, method="log")
    dv_kms = u.Quantity(delta, u.km / u.s).to_value(u.km / u.s)
    for I, (w, s) in enumerate(zip(spec.wv, spec.sp)):
        w_new, s_new = resample_to_log_grid(w, s, dv_kms)
        assert np.array_equal(w_new, obj.wv[I])
        assert np.array_equal(s_new, obj.sp[I])
        assert coarsest_native_step(dsp.dv_from_wave(w), w_new) == obj.native_steps()[I]


def test_bandpass_ends_with_standardize():
    rng = np.random.default_rng(3)
    wave = np.linspace(5000.0, 5100.0, 4000)
    flux = 1.0 + 0.01 * rng.standard_normal(wave.size) - 0.3 * np.exp(-0.5 * ((wave - 5050) / 0.1) ** 2)
    out = bandpass(wave, flux, lowcut=3, highcut=0.15)
    np.testing.assert_allclose(out, dsp.standardize(out), rtol=0, atol=1e-12)
