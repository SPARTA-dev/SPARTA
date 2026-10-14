"""CCF2d as a composition of the TODCOR per-window atoms.

Three layers, as in ``tests/test_preprocess.py``:

* the noise model of the resampling, now public in :mod:`sparta.preprocess`
  (``resampling_noise_psd``, ``native_phases``);
* ``CCF2d.cross_correlate`` calls each atom the expected number of times --
  spied through the names ``sparta.ccf2d`` imports: ``prepare_window`` once
  per window, ``joint_fit`` twice under sigma=None (once with an explicit
  sigma), ``window_noise`` once under sigma=None (never otherwise);
* the atoms, chained by hand, rebuild ``Corr2d``, ``stats``, ``windows`` and
  ``crlb`` exactly (``np.array_equal``, not a tolerance), on the case of the
  byte-identity baseline (``tests/_atoms_baseline.py``), for sigma=None and an
  explicit sigma, in both output modes.

Velocities in km/s, wavelengths in Angstrom. Self-contained: no external data.
"""
import pathlib
import sys

import numpy as np
import pytest
from astropy import units as u

from todcor import joint_fit, prepare_window, window_noise, window_stats
from unicor import dsp
from sparta import ccf2d as ccf2d_module
from sparta import preprocess
from sparta.ccf2d import CCF2d
from sparta.preprocess import native_phases, resample_to_log_grid, resampling_noise_psd
from sparta.spectrum import Spectrum
from sparta.template import Template

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _atoms_baseline import _sb2_case  # noqa: E402

DV, VEL_BOUND, DVEL, ALPHA = 1.0, 100, 2.0, 0.1       # km/s, km/s, km/s, -
_MAP_KEYS = ("v1_grid", "v2_grid", "R", "logL_profile", "logL_marginal", "a1", "a2", "chi2",
             "map")


# =============================================================================
# The noise model of the resampling
# =============================================================================
def test_the_noise_model_is_public_in_preprocess():
    assert {"resampling_noise_psd", "native_phases"} <= set(preprocess.__all__)
    assert not hasattr(ccf2d_module, "_impulse_responses")


def test_native_phases_are_the_sub_pixel_offsets_inside_the_grid():
    wave, _ = dsp.log_lambda_grid(5000.0, 5010.0, 0.5)
    step = dsp.log_grid_step(wave)
    native = wave * np.exp(0.25 * step / dsp.C_KMS)        # a quarter pixel redward
    phases = native_phases(native, wave, step)
    assert phases.size == wave.size - 1                     # the last one is outside
    np.testing.assert_allclose(phases, 0.25, atol=1e-6)
    assert np.all((phases >= 0.0) & (phases < 1.0))


def test_uniform_phases_are_the_default():
    phases = (np.arange(32) + 0.5) / 32
    assert np.array_equal(resampling_noise_psd(4000, 1.0, 0.1, phases),
                          resampling_noise_psd(4000, 1.0, 0.1))


def test_resampling_noise_psd_is_the_splines_variance_factor():
    # Ten analysis pixels per native one: the spline keeps ~0.87 of the
    # native variance. Nodes on the native pixels pass it unchanged.
    fine = resampling_noise_psd(4000, 1.0, 0.1)
    assert fine.shape == (2000,) and 0.8 < fine.mean() < 0.95
    on_nodes = resampling_noise_psd(4000, 1.0, 1.0, phases=np.zeros(3000))
    assert on_nodes.mean() == pytest.approx(1.0, abs=0.03)


def test_the_impulse_responses_are_cached_on_the_two_steps():
    preprocess._impulse_responses.cache_clear()
    for _ in range(3):
        resampling_noise_psd(3000, 1.0, 0.1)
        resampling_noise_psd(3000, 1.0 + 1e-12, 0.1)        # rounded to the same key
    info = preprocess._impulse_responses.cache_info()
    assert (info.misses, info.hits) == (1, 5)


# =============================================================================
# CCF2d calls each atom the expected number of times
# =============================================================================
def _spy(monkeypatch, name):
    calls = []
    real = getattr(ccf2d_module, name)

    def wrapper(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(ccf2d_module, name, wrapper)
    return calls


@pytest.mark.parametrize("sigma, n_fits, n_noise", [(None, 2, 1), (0.005, 1, 0)])
def test_cross_correlate_calls_each_atom(monkeypatch, sigma, n_fits, n_noise):
    spec, t1, t2 = _sb2_case()
    names = ("prepare_window", "joint_fit", "window_noise", "resampling_noise_psd",
             "native_phases")
    calls = {name: _spy(monkeypatch, name) for name in names}
    CCF2d().cross_correlate(spec, t1, t2, dv=DV, vel_bound=VEL_BOUND, dvel=DVEL, sigma=sigma)
    assert {k: len(v) for k, v in calls.items()} == {
        "prepare_window": 2, "joint_fit": n_fits, "window_noise": n_noise,
        "resampling_noise_psd": 2, "native_phases": 2}
    if sigma is None:
        # Pass 2 is weighted by the pass-1 estimates and shares its fine cache.
        (_, w1), (_, w2) = [c[0][:2] for c in calls["joint_fit"]]
        assert w1 is None and w2.shape == (2,)
        assert calls["joint_fit"][0][1]["fine_cache"] is calls["joint_fit"][1][1]["fine_cache"]


def test_a_log_grid_spectrum_skips_the_resampling_noise_model(monkeypatch):
    spec, t1, t2 = _sb2_case()
    spec.interpolate(delta=DV, method="log")
    log_only = Spectrum(wv=spec.wv, sp=spec.sp)
    log_only.Info.update(GridType="log", GridDelta=DV * u.km / u.s)
    calls = {name: _spy(monkeypatch, name)
             for name in ("prepare_window", "resampling_noise_psd", "native_phases")}
    CCF2d().cross_correlate(log_only, t1, t2, vel_bound=VEL_BOUND, dvel=DVEL, sigma=0.005,
                            orders=[1])
    assert {k: len(v) for k, v in calls.items()} == {
        "prepare_window": 1, "resampling_noise_psd": 0, "native_phases": 0}


def test_each_order_gets_its_own_template_order(monkeypatch):
    spec, t1, t2 = _sb2_case()
    spec.interpolate(delta=DV, method="log")
    tw, tf = t1.model.wv[0], t1.model.sp[0]
    cut = [(tw > w[0] - 5.0) & (tw < w[-1] + 5.0) for w in spec.wv]
    multi = Template(Spectrum(wv=[tw[c] for c in cut], sp=[tf[c] for c in cut]))
    calls = _spy(monkeypatch, "prepare_window")
    CCF2d().cross_correlate(spec, multi, t2, vel_bound=VEL_BOUND, dvel=DVEL, sigma=0.005)
    for I, (args, _) in enumerate(calls):
        assert np.array_equal(args[2], multi.model.wv[I])     # template1, order I
        assert np.array_equal(args[4], t2.model.wv[0])        # single-order template2


def test_the_window_error_names_the_order():
    spec, t1, t2 = _sb2_case()
    spec.interpolate(delta=DV, method="log")               # so CCF2d does not resample
    spec.sp[1][200] = np.nan
    with pytest.raises(ValueError, match=r"^order 1: spectrum flux contains NaN/inf; mask"):
        CCF2d().cross_correlate(spec, t1, t2, vel_bound=VEL_BOUND, dvel=DVEL)


def test_the_noise_fallback_warning_points_at_the_caller():
    spec, t1, t2 = _sb2_case()
    ccf = CCF2d()
    ccf.min_noise_dof = 1e9
    with pytest.warns(UserWarning, match="common noise estimate") as record:
        ccf.cross_correlate(spec, t1, t2, dv=DV, vel_bound=VEL_BOUND, dvel=DVEL)
    assert [w["sigma_fallback"] for w in ccf.windows] == [True, True]
    assert {r.filename for r in record} == {__file__}


# =============================================================================
# The atoms rebuild CCF2d exactly
# =============================================================================
def _rebuild(spec, t1, t2, sigma, output):
    """CCF2d.cross_correlate(spec, t1, t2, dv=DV, vel_bound=VEL_BOUND, dvel=DVEL,
    sigma=sigma, alpha=ALPHA, output=output), from the atoms."""
    native = spec.native_steps()
    v = np.arange(-VEL_BOUND, VEL_BOUND + 0.5 * DVEL, DVEL)
    sigmas = np.broadcast_to(np.atleast_1d(np.asarray(1.0 if sigma is None else sigma, float)),
                             (len(spec.wv),))
    prepared, stats = [], []
    for I, (w, s) in enumerate(zip(spec.wv, spec.sp)):
        w_log, s_log = resample_to_log_grid(w, s, DV)
        win = prepare_window(w_log, s_log, t1.model.wv[0], t1.model.sp[0],
                             t2.model.wv[0], t2.model.sp[0], continuum="median", alpha=ALPHA)
        psd = resampling_noise_psd(len(win.wave), native[I], win.dv,
                                   native_phases(w, win.wave, win.dv))
        kw = dict(sigma=sigmas[I], zeta_max=None, oversampling=max(1.0, native[I] / win.dv),
                  noise_psd=win.taper_power * psd, velocity_convention="cz")
        stats.append(window_stats(win.data_g, win.g1, win.g2, win.dv, v, v, **kw))
        prepared.append((win, kw))

    cache = {}
    fit = joint_fit(stats, None, prepared, v, v, nonneg=True, output=output, fine_cache=cache)
    noise = sigma
    if sigma is None:
        noise, _ = window_noise(fit.window_chi2, [st["nu"] for st in fit.at_peak])
        fit = joint_fit(stats, 1.0 / noise ** 2, prepared, v, v, nonneg=True, output=output,
                        fine_cache=cache)
    return fit, prepared, stats, noise


def _same(a, b):
    assert np.array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("output", ["ccf", "loglike"])
@pytest.mark.parametrize("sigma", [None, 0.005])
def test_the_atoms_rebuild_ccf2d_exactly(sigma, output):
    spec, t1, t2 = _sb2_case()
    ccf = CCF2d().cross_correlate(spec, t1, t2, dv=DV, vel_bound=VEL_BOUND, dvel=DVEL,
                                  sigma=sigma, alpha=ALPHA, output=output)
    fit, prepared, stats, noise = _rebuild(spec, t1, t2, sigma, output)
    res = ccf.Corr2d

    for maps, atom in ((res, fit.profile), (res["fine"], fit.fine)):
        for key in _MAP_KEYS:
            _same(maps[key], atom[key])
        assert maps["best"] == atom["best"] and maps["output"] == atom["output"] == output
    assert res["units"] == u.km / u.s

    crlb = fit.crlb
    assert (res["RV1"], res["RV2"]) == crlb["peak_refined"]
    assert (res["eRV1"], res["eRV2"], res["rho12"]) == (
        crlb["sigma_v1"], crlb["sigma_v2"], crlb["rho"])
    _same(res["cov"], crlb["cov"])
    assert res["peak"] == fit.coarse["peak"]
    for key in ("a1", "a2", "R", "chi2", "chi2_red"):
        assert res[f"{key}_best" if key != "chi2_red" else key] == fit.best[key]
    _same(res["sigma"], noise)
    assert res["sigma_from_fit"] is (sigma is None)
    assert ccf.crlb.keys() == crlb.keys()
    for key in crlb:
        _same(ccf.crlb[key], crlb[key])

    assert ccf.stats.keys() == fit.pooled.keys()
    for key in fit.pooled:
        _same(ccf.stats[key], fit.pooled[key])

    assert ccf.n_ord == len(prepared)
    for I, (obj, (win, kw), st) in enumerate(zip(ccf.windows, prepared, stats)):
        assert obj["order"] == I and obj["dv"] == win.dv
        for key in ("S11", "S22", "SD", "nu"):
            assert obj[key] == st[key]
        assert obj["oversampling"] == kw["oversampling"]
        assert obj["n_eff"] == len(win.wave) / kw["oversampling"]
        assert obj["sigma"] == (noise[I] if sigma is None else sigma)
        for key, arr in (("wv", win.wave), ("data_g", win.data_g), ("g1", win.g1),
                         ("g2", win.g2)):
            _same(obj[key], arr)
