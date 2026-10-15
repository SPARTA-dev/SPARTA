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
    bandpass, coarsest_native_step, filtered_noise_fraction, remove_cosmics, resample_flux,
    resample_to_linear_grid, resample_to_log_grid, trim_margins,
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
    out = bandpass(wave, flux, continuum_scale_kms=400.0, noise_scale_kms=4.0)
    assert out.mean() == pytest.approx(0.0, abs=1e-12)
    assert out.std() == pytest.approx(1.0, rel=1e-12)
    # The continuum slope is gone (the input correlates with it at 0.83);
    # the lines are what is left.
    assert abs(np.corrcoef(out, wave - 5010.0)[0, 1]) < 0.05
    lines_only = bandpass(wave, 500.0 * _lines(wave), continuum_scale_kms=400.0,
                          noise_scale_kms=4.0)
    assert np.corrcoef(out, lines_only)[0, 1] > 0.95


def test_the_filter_edges_match_the_same_spectrum_filtered_inside_a_longer_one():
    # An order filtered on its own must agree, right up to its ends, with the
    # same stretch filtered as part of a longer spectrum, where those pixels
    # are far from any edge. With 30 padding pixels the filter started up on
    # badly guessed edge values and spoiled ~300 km/s at each end (the taper
    # used to hide it); Gustafsson's initial conditions remove that.
    rng = np.random.default_rng(5)
    centres, depths = rng.uniform(4800.0, 5300.0, 600), rng.uniform(0.1, 0.6, 600)

    def spectrum(wave):
        lines = np.sum(depths * np.exp(-0.5 * ((wave[:, None] - centres) / 0.05) ** 2), axis=1)
        return (1.0 + 0.3 * (wave - 5050.0) / 250.0) * (1.0 - lines)

    step = 0.01
    long_wave = np.arange(4800.0, 5300.0, step)
    order = (long_wave >= 5000.0) & (long_wave < 5100.0)
    kw = dict(continuum_scale_kms=2000.0, noise_scale_kms=4.0)
    alone = bandpass(long_wave[order], spectrum(long_wave[order]), **kw)
    inside = bandpass(long_wave, spectrum(long_wave), **kw)[order]
    edge = int(round(300.0 / (C_KMS * step / 5050.0)))                # 300 km/s of pixels
    for part in (slice(0, edge), slice(-edge, None)):
        assert np.corrcoef(alone[part], inside[part])[0, 1] > 0.999


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
    ("filter_spectrum", dict(continuum_scale_kms=2000.0, noise_scale_kms=4.0, order=1),
     ["bandpass"]),
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


def test_preprocess_does_not_fade_the_ends():
    # No taper: a Tukey window forces the first and last pixel of every order
    # to exactly 0; preprocess leaves them as the filter made them.
    spec = _raw_spectrum().preprocess()
    for s in spec.sp:
        assert abs(s[0]) > 1e-3 and abs(s[-1]) > 1e-3
    assert not hasattr(spec, "apply_cosine_bell")


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
        s = bandpass(w, s, continuum_scale_kms=2000.0, noise_scale_kms=4.0, order=1)
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
    out = bandpass(wave, flux, continuum_scale_kms=2000.0, noise_scale_kms=4.0)
    np.testing.assert_allclose(out, dsp.standardize(out), rtol=0, atol=1e-12)


# --- the band-pass cuts are velocity scales (km/s): the same physical filter
# whatever the pixel size or the wavelength

def _amplitudes(wave, flux, periods_kms):
    """Amplitudes of the sinusoids of the given velocity periods in the
    interior of the order (one joint least-squares fit, so they do not leak
    into each other)."""
    lam = np.median(wave)
    v = C_KMS * (wave - lam) / lam
    cols = [np.ones_like(v)]
    for p in periods_kms:
        cols += [np.sin(2 * np.pi * v / p), np.cos(2 * np.pi * v / p)]
    inner = slice(wave.size // 5, -wave.size // 5)
    coef = np.linalg.lstsq(np.column_stack(cols)[inner], flux[inner], rcond=None)[0]
    return [np.hypot(coef[1 + 2 * k], coef[2 + 2 * k]) for k in range(len(periods_kms))]


@pytest.mark.parametrize("lam0, step", [(5000.0, 0.005), (5000.0, 0.02), (6500.0, 0.01)])
def test_bandpass_cuts_are_the_same_velocity_scale_on_any_grid(lam0, step):
    # One spectrum with a sinusoid in the pass band (60 km/s) and one at the
    # noise scale (4 km/s), equal in amplitude: their output ratio is the
    # filter's response at the cut, the same on every grid (standardization
    # scales both alike). In pixel units it changed with the pixel size.
    wave = np.arange(lam0 - 50.0, lam0 + 50.0, step)
    v = C_KMS * (wave - lam0) / lam0                            # km/s
    flux = 1.0 + 0.01 * (np.sin(2 * np.pi * v / 60.0) + np.sin(2 * np.pi * v / 4.0))
    out = bandpass(wave, flux, continuum_scale_kms=2000.0, noise_scale_kms=4.0)
    passband, at_cut = _amplitudes(wave, out, [60.0, 4.0])
    # filtfilt squares the first-order response: 1/2 at the cut, ~1 in band.
    assert at_cut / passband == pytest.approx(0.5, abs=0.01)


def test_the_same_spectrum_on_two_pixel_scales_is_filtered_alike():
    # The trap the pixel units set: a template on finer pixels than the data
    # was filtered differently. In velocity units both come out the same.
    def spectrum(step):
        # Narrow lines (FWHM ~6 km/s), well inside the compared window, so
        # the noise cut matters; the old pixel-unit filter gave 0.965 here.
        wave = np.arange(5000.0, 5100.0, step)
        lines = sum(np.exp(-0.5 * ((wave - c) / 0.04) ** 2) for c in (5030.0, 5050.0, 5070.0))
        return wave, 1.0 + 0.2 * (wave - 5050) / 50 - 0.5 * lines
    fine_w, fine_f = spectrum(0.005)
    coarse_w, coarse_f = spectrum(0.02)
    fine = bandpass(fine_w, fine_f, continuum_scale_kms=2000.0, noise_scale_kms=4.0)
    coarse = bandpass(coarse_w, coarse_f, continuum_scale_kms=2000.0, noise_scale_kms=4.0)
    inner = (coarse_w > 5010) & (coarse_w < 5090)
    on_coarse = np.interp(coarse_w, fine_w, fine)
    assert np.corrcoef(on_coarse[inner], coarse[inner])[0, 1] > 0.999


@pytest.mark.parametrize("continuum, noise, match", [
    (2000.0, 0.01, "noise_scale_kms"),         # finer than two grid steps
    (3.0, 4.0, "continuum_scale_kms"),         # not broader than the noise scale
    (2000.0, 0.0, "positive"), (-5.0, 4.0, "positive"), (np.nan, 4.0, "positive"),
    (2000.0 * u.m / u.s, 4.0, "noise_scale_kms"),   # 2 km/s: not broader than the noise scale
])
def test_bandpass_refuses_cuts_it_cannot_apply(continuum, noise, match):
    wave = np.arange(5000.0, 5100.0, 0.01)
    with pytest.raises(ValueError, match=match):
        bandpass(wave, 1.0 + 0.0 * wave, continuum_scale_kms=continuum, noise_scale_kms=noise)


def test_bandpass_takes_velocity_quantities():
    wave = np.arange(5000.0, 5100.0, 0.01)
    flux = 1.0 - 0.5 * _lines(wave)
    plain = bandpass(wave, flux, continuum_scale_kms=2000.0, noise_scale_kms=4.0)
    quantity = bandpass(wave, flux, continuum_scale_kms=2000.0 * u.km / u.s,
                        noise_scale_kms=4000.0 * u.m / u.s)
    np.testing.assert_allclose(quantity, plain, rtol=0, atol=1e-12)


def test_bandpass_refuses_an_uneven_grid():
    # The Nyquist frequency comes from the first step: a ln-lambda or raw grid
    # would be filtered at the wrong scales.
    wave = dsp.log_lambda_grid(5000.0, 5200.0, 0.5)[0]
    with pytest.raises(ValueError, match="evenly spaced"):
        bandpass(wave, 1.0 + 0.0 * wave, continuum_scale_kms=2000.0, noise_scale_kms=4.0)


# --- the noise the band-pass keeps: the error bars of filtered spectra -------

@pytest.mark.parametrize("noise_scale_kms", [4.0, 10.0])
def test_filtered_noise_fraction_is_the_variance_the_filter_leaves_of_white_noise(noise_scale_kms):
    # On a grid at the native sampling, white noise filtered exactly as
    # bandpass does keeps this fraction of its variance.
    from scipy import signal
    from sparta.preprocess import _bandpass_coefficients
    wave = 5000.0 * np.exp(np.arange(200_000) * 0.6 / C_KMS)            # 0.6 km/s pixels
    wave = np.linspace(wave[0], wave[-1], wave.size)                    # evenly spaced, as bandpass needs
    b, a = _bandpass_coefficients(wave, 2000.0, noise_scale_kms)
    x = np.random.default_rng(0).standard_normal(wave.size)
    kept = np.var(signal.filtfilt(b, a, x, method="gust")) / np.var(x)
    fraction = filtered_noise_fraction(wave, 2000.0, noise_scale_kms, native_step_kms=0.6)
    assert fraction == pytest.approx(kept, rel=0.03)
    assert 0.0 < fraction < 1.0


def test_filtered_noise_fraction_counts_only_the_native_band():
    # The same physical filter keeps the same share of the native noise on a
    # grid at the native sampling and on one 4x finer: the finer grid's extra
    # frequencies hold no native noise.
    native = 0.6                                                        # km/s
    step = native / C_KMS * 5050.0                                      # Angstrom
    coarse = np.arange(5000.0, 5100.0, step)
    fine = np.arange(5000.0, 5100.0, step / 4)
    on_native = filtered_noise_fraction(coarse, 2000.0, 4.0, native)
    on_fine = filtered_noise_fraction(fine, 2000.0, 4.0, native)
    # A digital filter's response is warped near its own Nyquist frequency,
    # differently on the two grids: 3% here, 1.5% in an error bar.
    assert on_fine == pytest.approx(on_native, rel=0.05)


def test_filter_spectrum_records_the_noise_fraction_of_each_order():
    spec = _raw_spectrum().trim(clean_margins=True).interpolate()
    natives = spec.native_steps()
    spec.filter_spectrum(continuum_scale_kms=2000.0, noise_scale_kms=4.0, order=1)
    expected = [filtered_noise_fraction(w, 2000.0, 4.0, n) for w, n in zip(spec.wv, natives)]
    assert spec.Info["NoiseFraction"] == pytest.approx(expected, rel=1e-12)
