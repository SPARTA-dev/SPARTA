"""CCF1d as a composition of the per-order UNICOR atoms (:mod:`unicor.ccf`).

Two layers (the atoms' own unit tests live in packages/unicor/tests):

* ``CCF1d`` calls the atoms -- ``lag_axis`` once, ``ccf_order`` once per
  order, ``measure_rv`` once per combined RV -- spied through the names
  ``sparta.ccf1d`` imports;
* the atoms, chained by hand on plain arrays, rebuild every ``Corr`` and
  ``CorrCombined`` entry exactly (``np.array_equal``, not a tolerance), in
  both output modes, including the NaN row of an order the template does not
  cover and the NaN RV of a peak beyond the scan.

Velocities in km/s, wavelengths in Angstrom. Self-contained: no external data.
"""
import pathlib
import sys
import warnings

import numpy as np
import pytest
from astropy import units as u

from sparta import ccf1d as ccf1d_module
from sparta.ccf1d import CCF1d
from sparta.preprocess import resample_to_log_grid
from sparta.spectrum import Spectrum
from sparta.template import Template
from unicor import (ccf_order, combine_ccfs, effective_bins, lag_axis, log_lambda_grid,
                    measure_rv)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _atoms_baseline import (  # noqa: E402
    _normalized_spectrum, _raw_spectrum, _sb1_template_arrays,
)

C_KMS = 299792.458


def _line(wave, v=0.0, centre=5500.0):
    """One Gaussian absorption line at ``centre`` (A), Doppler-shifted by v (km/s)."""
    return 1.0 - 0.6 * np.exp(-0.5 * ((wave - centre * (1 + v / C_KMS)) / 0.12) ** 2)


def _with_failures():
    """Three orders: uncovered by the template, a normal one, one beyond the scan."""
    wave = np.linspace(5480.0, 5520.0, 4000)
    far = np.linspace(6000.0, 6040.0, 4000)
    spec = Spectrum(wv=[far, wave.copy(), wave.copy()],
                    sp=[np.ones_like(far), _line(wave, 4.0), _line(wave, -70.0)], name="o")
    return spec, Template.from_arrays(wave, _line(wave))


# =============================================================================
# The class calls each atom where its loop body used to be
# =============================================================================
def _spy(monkeypatch, name):
    calls = []
    real = getattr(ccf1d_module, name)

    def wrapper(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(ccf1d_module, name, wrapper)
    return calls


def test_cross_correlate_calls_ccf_order_once_per_order(monkeypatch):
    spec, tmpl = _with_failures()
    calls = {name: _spy(monkeypatch, name) for name in ("lag_axis", "ccf_order", "measure_rv")}
    with pytest.warns(UserWarning, match="order 0: .*cover"):
        ccf = CCF1d().cross_correlate(spec, tmpl, dv=0.5, vel_bound=50)
    assert {n: len(c) for n, c in calls.items()} == {"lag_axis": 1, "ccf_order": 3,
                                                     "measure_rv": 0}
    natives = spec.native_steps()
    for I, (args, kwargs) in enumerate(calls["ccf_order"]):
        assert kwargs["dv_kms"] == 0.5 and kwargs["native_step_kms"] == natives[I]
        assert args[-1] == 100                                     # n_lags = 50 / 0.5
    assert np.isnan(ccf.Corr["RV"][0, 0]) and np.isnan(ccf.Corr["RV"][2, 0])


def test_combine_and_extract_rv_call_measure_rv_once(monkeypatch):
    obs = Spectrum(wv=[np.linspace(5480.0, 5520.0, 4000)] * 2,
                   sp=[_line(np.linspace(5480.0, 5520.0, 4000), 4.0)] * 2, name="o")
    tmpl = Template.from_arrays(obs.wv[0], _line(obs.wv[0]))
    ccf = CCF1d().cross_correlate(obs, tmpl, dv=0.5, vel_bound=50)
    calls = _spy(monkeypatch, "measure_rv")
    ccf.combine()
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0][1], ccf.CorrCombined["corr"])
    assert calls[0][0][2] == ccf.CorrCombined["n_eff"]
    ccf.extract_rv(ccf.CorrCombined["vel"], ccf.CorrCombined["corr"])
    assert len(calls) == 2
    ccf.extract_rv(ccf.CorrCombined["vel"], ccf.CorrCombined["corr"], vel=4.0)
    assert len(calls) == 2                       # an imposed velocity is not measured


# =============================================================================
# The atoms rebuild Corr and CorrCombined exactly
# =============================================================================
def _rebuild(spec, template, dv=None, vel_bound=150, errors_per_order=False,
             velocity_convention="cz"):
    '''CCF1d.cross_correlate(...).combine() from the atoms, on plain arrays.'''
    native = np.asarray(spec.native_steps(), float)
    # A band-passed spectrum carries the share of native noise it kept.
    noise = np.asarray(spec.Info.get("NoiseFraction", [1.0] * len(spec.wv)), float)
    if spec.Info.get("GridType") == "log":
        waves, fluxes = spec.wv, spec.sp
        dv_kms = u.Quantity(spec.Info["GridDelta"]).to_value(u.km / u.s)
    else:
        dv_kms = 0.1 if dv is None else float(dv)
        waves, fluxes = zip(*(resample_to_log_grid(w, s, dv_kms)
                              for w, s in zip(spec.wv, spec.sp)))
    n_lags = int(round(np.max(np.abs(vel_bound)) / dv_kms))
    _, v = lag_axis(waves, dv_kms, n_lags, velocity_convention)

    n_ord = len(waves)
    corr = np.full((n_ord, v.size), np.nan)
    loglike = np.full((n_ord, v.size), np.nan)
    rv, erv, peak = (np.full((n_ord, 1), np.nan) for _ in range(3))
    n_eff = effective_bins(np.array([len(w) for w in waves]), dv_kms, native) * noise
    single = len(template.model.wv) == 1
    for I, (w, s) in enumerate(zip(waves, fluxes)):
        It = 0 if single else I
        try:
            order = ccf_order(w, s, template.model.wv[It], template.model.sp[It], n_lags,
                              dv_kms=dv_kms, native_step_kms=native[I],
                              velocity_convention=velocity_convention,
                              errors=errors_per_order, noise_fraction=noise[I])
        except ValueError:
            continue
        corr[I], loglike[I], n_eff[I] = order.ccf, order.loglike, order.n_eff
        rv[I], erv[I], peak[I] = order.rv, order.erv, order.peak

    combined = combine_ccfs(corr, n_eff=n_eff)
    n_tot = float(np.sum(n_eff))
    try:
        m = measure_rv(v, combined, n_tot)
        scalars = [m.rv, m.erv, m.peak, n_tot]
    except (ValueError, np.linalg.LinAlgError):
        scalars = [np.nan, np.nan, np.nan, n_tot]
    return (dict(vel=v, corr=corr, loglike=loglike, RV=rv, eRV=erv, peakCorr=peak,
                 n_eff=n_eff),
            dict(vel=v, corr=combined, loglike=loglike.sum(axis=0)), scalars)


def _cases():
    wave, flux = _sb1_template_arrays()
    template_pre = Template(Spectrum(wv=[wave], sp=[flux]).preprocess())
    template_raw = Template.from_arrays(wave, flux)
    spec, tmpl = _with_failures()
    return {
        "preprocessed": (_raw_spectrum().preprocess(), template_pre,
                         dict(dv=0.5, vel_bound=80, errors_per_order=True)),
        "normalized": (_normalized_spectrum(), template_raw, dict(vel_bound=60)),
        "log_input_relativistic": (
            _normalized_spectrum().interpolate(delta=0.3, method="log"), template_raw,
            dict(vel_bound=[-40, 70], errors_per_order=True,
                 velocity_convention="relativistic")),
        "nan_orders": (spec, tmpl, dict(dv=0.5, vel_bound=50, errors_per_order=True)),
        "multi_order_template_log_input": _multi_order_template_case(),
    }


def _multi_order_template_case():
    """Two orders on a declared log grid, each with its own template order."""
    centres = (5500.0, 5620.0)
    waves = [log_lambda_grid(c - 20.0, c + 20.0, 0.5)[0] for c in centres]
    spec = Spectrum(wv=waves, sp=[_line(w, 3.0, c) for w, c in zip(waves, centres)],
                    name="o")
    spec.Info["GridType"], spec.Info["GridDelta"] = "log", 0.5 * u.km / u.s
    t_waves = [log_lambda_grid(c - 30.0, c + 30.0, 0.1)[0] for c in centres]
    tmpl = Template(Spectrum(wv=t_waves, sp=[_line(w, 0.0, c) for w, c in zip(t_waves, centres)],
                             name="t"))
    return spec, tmpl, dict(vel_bound=40, errors_per_order=True)


CASES = _cases()


@pytest.mark.parametrize("output", ["ccf", "loglike"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_atoms_rebuild_corr_and_corr_combined_exactly(case, output):
    spec, tmpl, kwargs = CASES[case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        obj = CCF1d().cross_correlate(spec, tmpl, output=output, **kwargs).combine()
    corr, combined, scalars = _rebuild(spec, tmpl, **kwargs)

    assert np.array_equal(obj.Corr["vel"].to_value(u.km / u.s), corr["vel"])
    for key in ("corr", "loglike", "RV", "eRV", "peakCorr", "n_eff"):
        assert np.array_equal(obj.Corr[key], corr[key], equal_nan=True), key
    for key in ("vel", "corr", "loglike"):
        assert np.array_equal(u.Quantity(obj.CorrCombined[key]).value, combined[key],
                              equal_nan=True), key
    got = [obj.CorrCombined[k] for k in ("RV", "eRV", "peakCorr", "n_eff")]
    assert np.array_equal(np.array(got, float), np.array(scalars, float), equal_nan=True)
    assert obj.Corr["output"] == obj.CorrCombined["output"] == output


def test_the_failure_cases_do_fail():
    # The rebuild above covers the NaN-row and NaN-RV policies only if they occur.
    spec, tmpl, kwargs = CASES["nan_orders"]
    with pytest.warns(UserWarning, match="order 0"):
        obj = CCF1d().cross_correlate(spec, tmpl, **kwargs)
    assert np.all(np.isnan(obj.Corr["corr"][0])) and np.all(np.isfinite(obj.Corr["corr"][2]))
    assert np.isnan(obj.Corr["RV"][2, 0]) and np.isfinite(obj.Corr["RV"][1, 0])
