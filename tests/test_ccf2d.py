"""Tests for SPARTA's TODCOR object layer (:class:`sparta.TODCOR.CCF2d`).

``CCF2d`` driven with ``Spectrum``/``Template`` objects on a synthetic SB2: one
spectral order per TODCOR window, pooled into one fit. The array-level estimator
it wraps is tested in ``packages/todcor/tests``.

Self-contained: no external data.
"""

import numpy as np
import pytest
from astropy import units as u

from todcor import composite_flux, log_lambda_grid
from sparta.TODCOR import CCF2d
from sparta.UNICOR.Spectrum import Spectrum
from sparta.UNICOR.Template import Template


def _gauss(wave, center, depth, fwhm):
    """A Gaussian line of fractional ``depth`` (negative = absorption)."""
    sig = fwhm / 2.3548200450309493
    return 1.0 + depth * np.exp(-0.5 * ((wave - center) / sig) ** 2)


def _synthetic_templates():
    """An absorption-line 'primary' and an emission-line 'secondary'."""
    wave = np.linspace(4000.0, 5000.0, 8000)
    t1 = (
        _gauss(wave, 4101.7, -0.5, 1.5)
        * _gauss(wave, 4340.5, -0.45, 1.5)
        * _gauss(wave, 4471.5, -0.25, 1.2)
    )
    t2 = _gauss(wave, 4058.0, +0.6, 2.0) * _gauss(wave, 4685.7, +0.8, 3.0)
    return (wave, t1), (wave, t2)


def _sparta_case(v1_true=60.0, v2_true=-95.0, a1_true=0.7, a2_true=0.4,
                 n_ord=2, dv=1.0):
    """A Spectrum + two Templates carrying a known two-component blend.

    The spectrum is built directly on a log-lambda grid and declares it, so
    CCF2d fits it as-is instead of resampling.
    """
    (w1, f1), (w2, f2) = _synthetic_templates()
    wvs, sps = [], []
    for lo, hi in [(4080.0, 4380.0), (4400.0, 4720.0)][:n_ord]:
        grid, _ = log_lambda_grid(lo, hi, dv)
        wvs.append(grid)
        sps.append(composite_flux(grid, [(w1, f1), (w2, f2)],
                                  [v1_true, v2_true], [a1_true, a2_true]))
    spec = Spectrum(wv=wvs, sp=sps, name='SB2-sim')
    spec.Info['GridType'] = 'log'
    spec.Info['GridDelta'] = dv * u.km / u.s
    return (spec,
            Template(wavelengths=w1, spectrum=f1),
            Template(wavelengths=w2, spectrum=f2))


def test_ccf2d_recovers_injected_velocities():
    v1_true, v2_true, a1_true, a2_true = 60.0, -95.0, 0.7, 0.4
    spec, tmpl1, tmpl2 = _sparta_case(v1_true, v2_true, a1_true, a2_true)

    ccf = CCF2d().CrossCorrelateSpec(spec, tmpl1, tmpl2, VelBound=150, dvel=2.0,
                                     sigma=0.01, alpha=0.1)

    assert ccf.n_ord == 2
    assert abs(ccf.Corr2d['RV1'] - v1_true) < 3.0
    assert abs(ccf.Corr2d['RV2'] - v2_true) < 3.0
    assert ccf.Corr2d['a1_best'] == pytest.approx(a1_true, abs=0.1)
    assert ccf.Corr2d['a2_best'] == pytest.approx(a2_true, abs=0.1)
    # Uncertainties are finite and small on a noiseless, well-resolved peak.
    assert 0 < ccf.Corr2d['eRV1'] < 3.0
    assert 0 < ccf.Corr2d['eRV2'] < 3.0


def test_ccf2d_single_order_template_is_reused_per_order():
    # Both templates are single-order while the spectrum has two: the adapter
    # must resample order 0 of the template onto each observed order.
    spec, tmpl1, tmpl2 = _sparta_case(n_ord=2)
    ccf = CCF2d().CrossCorrelateSpec(spec, tmpl1, tmpl2, VelBound=120, dvel=4.0)
    assert len(ccf.windows) == 2
    assert all(len(w['g1']) == len(w['wv']) for w in ccf.windows)


def test_ccf2d_keep_windows_false_drops_arrays_but_keeps_powers():
    # The time-series driver turns the arrays off; the scalar per-order powers
    # (which say what each order constrains) must survive, and the residual
    # helper must say why it cannot run.
    spec, tmpl1, tmpl2 = _sparta_case()
    ccf = CCF2d().CrossCorrelateSpec(spec, tmpl1, tmpl2, VelBound=120, dvel=4.0,
                                     keep_windows=False)
    assert all('wv' not in w and w['S11'] > 0 for w in ccf.windows)
    assert set(ccf.stats) == {'S11', 'S22', 'SD', 'b1', 'b2', 'A12'}
    with pytest.raises(ValueError, match='keep_windows=True'):
        ccf.BestFitSpectrum(order=0)


def test_ccf2d_best_fit_reproduces_the_data():
    spec, tmpl1, tmpl2 = _sparta_case()
    ccf = CCF2d().CrossCorrelateSpec(spec, tmpl1, tmpl2, VelBound=150, dvel=2.0,
                                     sigma=0.01)
    wv, model, data = ccf.BestFitSpectrum(order=0)
    # Compare in the interior, away from the apodised edges.
    sl = slice(len(wv) // 5, -len(wv) // 5)
    assert np.std(model[sl] - data[sl]) < 0.05


def test_ccf2d_refuses_a_nan_pixel():
    # One NaN used to turn the whole map NaN and report the grid corner.
    spec, tmpl1, tmpl2 = _sparta_case()
    spec.sp[0][100] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        CCF2d().CrossCorrelateSpec(spec, tmpl1, tmpl2, VelBound=150, dvel=2.0)


def test_ccf2d_refuses_a_template_with_no_lines_in_the_fitted_orders():
    # A pure-continuum secondary template gives S22 = 0, and both amplitudes
    # used to collapse silently to zero with the RV at the grid corner. (The
    # emission-line secondary of _sparta_case has no lines in order 0 alone
    # either -- the same guard fires there.)
    spec, tmpl1, _ = _sparta_case(n_ord=1)
    w = np.linspace(4000.0, 5000.0, 8000)
    flat = Template(wavelengths=w, spectrum=np.ones_like(w))
    with pytest.raises(ValueError, match="template2 has no spectral structure"):
        CCF2d().CrossCorrelateSpec(spec, tmpl1, flat, VelBound=150, dvel=2.0)


def test_ccf2d_refuses_a_grid_step_that_contradicts_griddelta():
    spec, tmpl1, tmpl2 = _sparta_case(dv=1.0)
    spec.Info['GridDelta'] = 0.8 * u.km / u.s
    with pytest.raises(ValueError, match="GridDelta"):
        CCF2d().CrossCorrelateSpec(spec, tmpl1, tmpl2, VelBound=150, dvel=2.0)


def test_ccf2d_peak_on_the_grid_edge_warns_and_reports_no_error():
    # Truth v1 = 60 km/s lies beyond a grid ending at 40 km/s.
    spec, tmpl1, tmpl2 = _sparta_case()
    grid = np.arange(-150.0, 41.0, 2.0)
    with pytest.warns(UserWarning, match="edge of the velocity grid"):
        ccf = CCF2d().CrossCorrelateSpec(spec, tmpl1, tmpl2, v1_grid=grid,
                                         v2_grid=np.arange(-150.0, 151.0, 2.0))
    assert ccf.Corr2d['RV1'] == 40.0           # grid-limited, not extrapolated
    assert np.isnan(ccf.Corr2d['eRV1'])


def test_ccf2d_accepts_velocity_grids_with_units():
    spec, tmpl1, tmpl2 = _sparta_case()
    grid_ms = np.arange(-150.0, 151.0, 2.0) * 1e3 * u.m / u.s
    ccf = CCF2d().CrossCorrelateSpec(spec, tmpl1, tmpl2, v1_grid=grid_ms, v2_grid=grid_ms)
    assert ccf.Corr2d['RV1'] == pytest.approx(60.0, abs=1.0)
