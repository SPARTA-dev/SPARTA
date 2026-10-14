"""Tests for SPARTA's TODCOR object layer (:class:`sparta.TODCOR.CCF2d`).

``CCF2d`` driven with ``Spectrum``/``Template`` objects on a synthetic SB2: one
spectral order per TODCOR window, pooled into one fit. The array-level estimator
it wraps is tested in ``packages/todcor/tests``.

Self-contained: no external data.
"""

import inspect

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
    assert set(ccf.stats) == {'S11', 'S22', 'SD', 'b1', 'b2', 'A12', 'nu'}
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
    assert ccf.Corr2d['fine'] is None          # no refinement off a grid edge


def test_ccf2d_accepts_velocity_grids_with_units():
    spec, tmpl1, tmpl2 = _sparta_case()
    grid_ms = np.arange(-150.0, 151.0, 2.0) * 1e3 * u.m / u.s
    ccf = CCF2d().CrossCorrelateSpec(spec, tmpl1, tmpl2, v1_grid=grid_ms, v2_grid=grid_ms)
    assert ccf.Corr2d['RV1'] == pytest.approx(60.0, abs=1.0)


# =============================================================================
# A realistic narrow-line SB2 observed on a ~1 km/s native grid
# =============================================================================
_LO, _HI = 4450.0, 4520.0
_C_KMS = 299792.458


def _line_list(seed, n=25, lo=_LO, hi=_HI):
    rng = np.random.default_rng(seed)
    return rng.uniform(lo + 3.0, hi - 3.0, n), rng.uniform(0.1, 0.6, n)


_LINES1, _LINES2 = _line_list(1), _line_list(2)


def _narrow_flux(wave, lines, v=0.0, fwhm=8.0):
    """Analytic absorption-line spectrum (line FWHM in km/s), shifted by ``v``."""
    centers, depths = lines
    rest = np.asarray(wave, float)[None, :] / (1.0 + v / _C_KMS)
    sig = centers[:, None] * fwhm / 2.3548200450309493 / _C_KMS
    return 1.0 - np.sum(depths[:, None]
                        * np.exp(-0.5 * ((rest - centers[:, None]) / sig) ** 2), axis=0)


def _narrow_case(v1, v2, a1=0.7, a2=0.4, lines=(_LINES1, _LINES2)):
    """Noiseless composite on a 1 km/s native grid (NOT flagged as the analysis
    grid, so CCF2d resamples it to its default 0.1 km/s), plus finely sampled
    rest-frame templates. The composite is analytic: no forward-model
    interpolation error enters the accuracy tests."""
    wave, _ = log_lambda_grid(_LO, _HI, 1.0)
    flux = (1.0 + a1 * (_narrow_flux(wave, lines[0], v1) - 1.0)
            + a2 * (_narrow_flux(wave, lines[1], v2) - 1.0))
    wt, _ = log_lambda_grid(_LO - 5.0, _HI + 5.0, 0.05)
    return (wave, flux,
            Template(wavelengths=wt, spectrum=_narrow_flux(wt, lines[0])),
            Template(wavelengths=wt, spectrum=_narrow_flux(wt, lines[1])))


def _local_grids(v1, v2, half=15):
    return (np.arange(round(v1) - half, round(v1) + half + 1.0),
            np.arange(round(v2) - half, round(v2) + half + 1.0))


def test_ccf2d_crlb_matches_monte_carlo_on_a_resampled_native_grid():
    # Noise lives on the native 1 km/s pixels; CCF2d interpolates onto 0.1 km/s,
    # so ~10 analysis pixels share each noise value. Counting each of them as
    # independent made the CRLB ~2-3x too small; the CRLB must now predict the
    # Monte-Carlo scatter and chi2_red must be ~1 for the true sigma.
    v1, v2, sigma = 37.3, -52.6, 0.01
    wave, flux0, tmpl1, tmpl2 = _narrow_case(v1, v2)
    g1, g2 = _local_grids(v1, v2)
    rng = np.random.default_rng(0)
    rv1, rv2, e1, e2, chi2_red = [], [], [], [], []
    for _ in range(60):
        spec = Spectrum(wv=[wave], sp=[flux0 + rng.normal(0.0, sigma, wave.size)])
        r = CCF2d().CrossCorrelateSpec(spec, tmpl1, tmpl2, v1_grid=g1, v2_grid=g2,
                                       sigma=sigma, keep_windows=False).Corr2d
        rv1.append(r['RV1'])
        rv2.append(r['RV2'])
        e1.append(r['eRV1'])
        e2.append(r['eRV2'])
        chi2_red.append(r['chi2_red'])
    assert 0.7 < np.std(rv1) / np.mean(e1) < 1.4
    assert 0.7 < np.std(rv2) / np.mean(e2) < 1.4
    assert np.mean(chi2_red) == pytest.approx(1.0, abs=0.05)


def test_ccf2d_records_effective_pixel_counts():
    wave, flux, tmpl1, tmpl2 = _narrow_case(37.3, -52.6)
    ccf = CCF2d().CrossCorrelateSpec(Spectrum(wv=[wave], sp=[flux]), tmpl1, tmpl2,
                                     v1_grid=_local_grids(37.3, -52.6)[0],
                                     v2_grid=_local_grids(37.3, -52.6)[1])
    win = ccf.windows[0]
    assert win['oversampling'] == pytest.approx(10.0, rel=1e-3)
    assert win['n_eff'] == pytest.approx(len(win['wv']) / 10.0, rel=1e-3)
    # The noise dof: n_eff times the spline's noise-variance factor (~0.87)
    # times the taper's mean square (~0.94).
    assert 0.75 * win['n_eff'] < win['nu'] < 0.9 * win['n_eff']


def test_ccf2d_sigma_none_estimates_the_noise_from_the_fit():
    # The default sigma=None estimates each order's noise from its residual;
    # with a single order that is sqrt(chi2_red) at unit sigma. The errors
    # must match those computed with the true sigma.
    v1, v2, sigma = 37.3, -52.6, 0.01
    wave, flux0, tmpl1, tmpl2 = _narrow_case(v1, v2)
    g1, g2 = _local_grids(v1, v2)
    rng = np.random.default_rng(1)
    for _ in range(4):
        spec = Spectrum(wv=[wave], sp=[flux0 + rng.normal(0.0, sigma, wave.size)])
        fit = CCF2d().CrossCorrelateSpec(spec, tmpl1, tmpl2, v1_grid=g1, v2_grid=g2)
        est = fit.Corr2d
        ref = CCF2d().CrossCorrelateSpec(spec, tmpl1, tmpl2, v1_grid=g1, v2_grid=g2,
                                         sigma=sigma).Corr2d
        assert est['sigma_from_fit'] and not ref['sigma_from_fit']
        assert est['sigma'] == pytest.approx(sigma, rel=0.1)
        assert fit.windows[0]['sigma'] == est['sigma'][0]
        assert not fit.windows[0]['sigma_fallback']
        assert est['RV1'] == pytest.approx(ref['RV1'], abs=1e-9)
        assert est['eRV1'] / ref['eRV1'] == pytest.approx(1.0, abs=0.15)
        assert est['eRV2'] / ref['eRV2'] == pytest.approx(1.0, abs=0.15)


def test_ccf2d_explicit_sigma_is_taken_at_face_value():
    # An explicit sigma is not rescaled, but chi2_red still reports the misfit.
    v1, v2 = 37.3, -52.6
    wave, flux0, tmpl1, tmpl2 = _narrow_case(v1, v2)
    g1, g2 = _local_grids(v1, v2)
    noisy = flux0 + np.random.default_rng(2).normal(0.0, 0.01, wave.size)
    r1 = CCF2d().CrossCorrelateSpec(Spectrum(wv=[wave], sp=[noisy]), tmpl1, tmpl2,
                                    v1_grid=g1, v2_grid=g2, sigma=0.01).Corr2d
    r2 = CCF2d().CrossCorrelateSpec(Spectrum(wv=[wave], sp=[noisy]), tmpl1, tmpl2,
                                    v1_grid=g1, v2_grid=g2, sigma=0.02).Corr2d
    assert r2['eRV1'] / r1['eRV1'] == pytest.approx(2.0, rel=1e-6)
    assert r2['chi2_red'] == pytest.approx(r1['chi2_red'] / 4.0, rel=1e-9)
    assert r1['chi2_red'] == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize('v1, v2', [(37.3, -52.6), (12.45, -8.8), (37.0, -52.0)])
def test_ccf2d_refined_rvs_unbiased_for_narrow_lines(v1, v2):
    # Noiseless, 8 km/s lines, default 1 km/s map: a quadratic over +/-3 coarse
    # nodes was biased by 0.03-0.1 km/s (several eRV); the fine second pass
    # must land on the truth, with the amplitudes taken at the same point.
    wave, flux, tmpl1, tmpl2 = _narrow_case(v1, v2)
    ccf = CCF2d().CrossCorrelateSpec(Spectrum(wv=[wave], sp=[flux]), tmpl1, tmpl2,
                                     VelBound=150, sigma=0.01)
    r = ccf.Corr2d
    assert abs(r['RV1'] - v1) < 0.01
    assert abs(r['RV2'] - v2) < 0.01
    assert r['a1_best'] == pytest.approx(0.7, abs=2e-3)
    assert r['a2_best'] == pytest.approx(0.4, abs=2e-3)
    # The coarse map is kept for plotting; the fine map sits under its own key.
    assert r['logL_profile'].shape == (301, 301)
    assert r['fine']['logL_profile'].shape == (41, 41)
    assert np.ptp(r['fine']['v1_grid']) == pytest.approx(4.0)
    assert r['peak'] == (round(v1), round(v2))


@pytest.mark.parametrize('v1, v2', [(40.0, -30.0), (25.0, -12.0)])
def test_ccf2d_twin_templates_with_truth_on_a_node(v1, v2):
    # Both components share one template; the truth sits exactly on a coarse
    # node, so the node itself is the right answer and refinement must not
    # move it. (Disjoint grids select one labelling of the twin pair.)
    wave, flux, tmpl1, tmpl2 = _narrow_case(v1, v2, a1=0.6, a2=0.4,
                                            lines=(_LINES1, _LINES1))
    r = CCF2d().CrossCorrelateSpec(Spectrum(wv=[wave], sp=[flux]), tmpl1, tmpl2,
                                   v1_grid=np.arange(0.0, 81.0),
                                   v2_grid=np.arange(-70.0, 11.0), sigma=0.01).Corr2d
    assert abs(r['RV1'] - v1) < 0.01
    assert abs(r['RV2'] - v2) < 0.01


def test_ccf2d_refinement_does_not_need_kept_windows():
    v1, v2 = 37.3, -52.6
    wave, flux, tmpl1, tmpl2 = _narrow_case(v1, v2)
    g1, g2 = _local_grids(v1, v2)
    kept = CCF2d().CrossCorrelateSpec(Spectrum(wv=[wave], sp=[flux]), tmpl1, tmpl2,
                                      v1_grid=g1, v2_grid=g2, sigma=0.01)
    dropped = CCF2d().CrossCorrelateSpec(Spectrum(wv=[wave], sp=[flux]), tmpl1, tmpl2,
                                         v1_grid=g1, v2_grid=g2, sigma=0.01,
                                         keep_windows=False)
    assert 'data_g' not in dropped.windows[0]
    for k in ('RV1', 'RV2', 'eRV1', 'eRV2', 'a1_best', 'chi2_red'):
        assert dropped.Corr2d[k] == kept.Corr2d[k]


@pytest.mark.parametrize('factor', [1.5, 2.0, 3.0])
def test_ccf2d_noise_budget_on_a_grid_coarser_than_native(factor):
    # The analysis grid is 1.5-3x coarser than the 1 km/s native grid. The
    # noise model's impulse probes all sat at one sub-pixel phase whenever
    # 40 * dv_native / dv was an integer: nu ~ 0.001 at 2x (chi2_red nan, and
    # sigma=None refused to run), chi2_red 1.12 / 1.30 at 1.5x / 3x for
    # honest noise.
    v1, v2, sigma = 37.3, -52.6, 0.01
    wave, flux0, tmpl1, tmpl2 = _narrow_case(v1, v2)
    g1, g2 = _local_grids(v1, v2)
    rng = np.random.default_rng(5)
    chi2_red, sig, rv1, rv2, e1, e2 = [], [], [], [], [], []
    for _ in range(30):
        spec = Spectrum(wv=[wave], sp=[flux0 + rng.normal(0.0, sigma, wave.size)])
        kw = dict(dv=factor, v1_grid=g1, v2_grid=g2, keep_windows=False)
        r = CCF2d().CrossCorrelateSpec(spec, tmpl1, tmpl2, sigma=sigma, **kw).Corr2d
        est = CCF2d().CrossCorrelateSpec(spec, tmpl1, tmpl2, **kw).Corr2d
        chi2_red.append(r['chi2_red'])
        sig.append(est['sigma'])
        rv1.append(r['RV1'])
        rv2.append(r['RV2'])
        e1.append(r['eRV1'])
        e2.append(r['eRV2'])
    assert np.mean(chi2_red) == pytest.approx(1.0, abs=0.05)
    assert np.mean(sig) == pytest.approx(sigma, rel=0.03)
    assert 0.7 < np.std(rv1) / np.mean(e1) < 1.4
    assert 0.7 < np.std(rv2) / np.mean(e2) < 1.4


def test_ccf2d_noise_psd_is_cached_per_sampling():
    # The probe depends only on the native and analysis steps; it must not be
    # rebuilt for every order of every epoch.
    from sparta.TODCOR.CCF2d import _impulse_responses
    _impulse_responses.cache_clear()
    wave, flux, tmpl1, tmpl2 = _narrow_case(37.3, -52.6)
    g1, g2 = _local_grids(37.3, -52.6)
    for _ in range(3):
        CCF2d().CrossCorrelateSpec(Spectrum(wv=[wave, wave], sp=[flux, flux]),
                                   tmpl1, tmpl2, v1_grid=g1, v2_grid=g2, sigma=0.01)
    info = _impulse_responses.cache_info()
    assert info.misses == 1 and info.hits == 5


def test_ccf2d_sigma_none_reports_chi2_at_the_estimated_noise():
    # chi2_best / chi2_red describe the fit at the noise behind the errors.
    # Under sigma=None that is the fitted noise, so chi2_red is 1 by
    # construction (it used to be reported at unit noise, i.e. ~sigma^2).
    # chi2_map is on the noise scale of the maps: unit noise under sigma=None.
    v1, v2 = 37.3, -52.6
    wave, flux0, tmpl1, tmpl2 = _narrow_case(v1, v2)
    g1, g2 = _local_grids(v1, v2)
    noisy = flux0 + np.random.default_rng(4).normal(0.0, 0.01, wave.size)
    est = CCF2d().CrossCorrelateSpec(Spectrum(wv=[wave], sp=[noisy]), tmpl1, tmpl2,
                                     v1_grid=g1, v2_grid=g2).Corr2d
    ref = CCF2d().CrossCorrelateSpec(Spectrum(wv=[wave], sp=[noisy]), tmpl1, tmpl2,
                                     v1_grid=g1, v2_grid=g2, sigma=est['sigma']).Corr2d
    assert est['chi2_red'] == pytest.approx(1.0, rel=1e-9)
    assert ref['chi2_red'] == pytest.approx(1.0, rel=1e-6)
    assert est['chi2_best'] == pytest.approx(ref['chi2_best'], rel=1e-6)
    assert est['eRV1'] == pytest.approx(ref['eRV1'], rel=1e-6)
    assert est['eRV2'] == pytest.approx(ref['eRV2'], rel=1e-6)
    # The maps sit on the noise scale of the reported sigma in both modes:
    # their minimum is next to chi2_best.
    for r in (est, ref):
        assert r['fine']['chi2'].min() == pytest.approx(r['chi2_best'], rel=1e-3)


def test_ccf2d_refuses_native_steps_that_do_not_match_the_orders():
    spec, tmpl1, tmpl2 = _sparta_case(n_ord=2)
    spec.Info['NativeStep'] = [1.0]
    with pytest.raises(ValueError, match="NativeStep"):
        CCF2d().CrossCorrelateSpec(spec, tmpl1, tmpl2, VelBound=120, dvel=4.0)


# =============================================================================
# Orders of different S/N under sigma=None
# =============================================================================
_ORDERS = ((4450.0, 4485.0), (4495.0, 4530.0))


def _order_lines(seed):
    """One component's lines in both orders, 3 A clear of every order edge: a
    line shifted across an edge is a misfit that would pose as noise."""
    parts = [_line_list(seed + 10 * k, n=12, lo=lo, hi=hi)
             for k, (lo, hi) in enumerate(_ORDERS)]
    return tuple(np.concatenate(x) for x in zip(*parts))


def _two_order_case(v1, v2, a1=0.7, a2=0.4):
    """Two noiseless orders on 1 km/s native grids, plus rest-frame templates
    covering both."""
    lines = (_order_lines(1), _order_lines(2))
    waves = [log_lambda_grid(lo, hi, 1.0)[0] for lo, hi in _ORDERS]
    fluxes = [1.0 + a1 * (_narrow_flux(w, lines[0], v1) - 1.0)
              + a2 * (_narrow_flux(w, lines[1], v2) - 1.0) for w in waves]
    wt, _ = log_lambda_grid(_ORDERS[0][0] - 5.0, _ORDERS[-1][1] + 5.0, 0.05)
    return (waves, fluxes,
            Template(wavelengths=wt, spectrum=_narrow_flux(wt, lines[0])),
            Template(wavelengths=wt, spectrum=_narrow_flux(wt, lines[1])))


def _per_order_noise_mc(sigmas, n_real, seed):
    """Fit each noisy realisation with sigma=None and with the true per-order
    sigma. Returns ({'none': ..., 'true': ...} arrays of (RV1, RV2, eRV1,
    eRV2) per realisation, and the per-order sigma estimates)."""
    v1, v2 = 37.3, -52.6
    waves, fluxes, tmpl1, tmpl2 = _two_order_case(v1, v2)
    g1, g2 = _local_grids(v1, v2)
    rng = np.random.default_rng(seed)
    out, sig_est = {'none': [], 'true': []}, []
    for _ in range(n_real):
        noisy = [f + rng.normal(0.0, s, f.size) for f, s in zip(fluxes, sigmas)]
        for key, sigma in (('none', None), ('true', sigmas)):
            ccf = CCF2d().CrossCorrelateSpec(Spectrum(wv=list(waves), sp=list(noisy)),
                                             tmpl1, tmpl2, v1_grid=g1, v2_grid=g2,
                                             sigma=sigma, keep_windows=False)
            r = ccf.Corr2d
            out[key].append([r['RV1'], r['RV2'], r['eRV1'], r['eRV2']])
            if sigma is None:
                sig_est.append([w['sigma'] for w in ccf.windows])
    return {k: np.array(v) for k, v in out.items()}, np.array(sig_est)


def test_ccf2d_sigma_none_weights_each_order_by_its_own_noise():
    # Orders with noise 0.003 and 0.03 on their native pixels. One common
    # noise level (the former sigma=None) weighted them equally: honest errors,
    # but an RV scatter ~5x that of the fit with the true per-order sigma.
    sigmas = np.array([0.003, 0.03])
    rv, sig_est = _per_order_noise_mc(sigmas, n_real=60, seed=7)
    # The cleaner order's residual at the unit-noise fit carries that fit's
    # misfit too (it is pulled by the noisier order): its estimate runs ~3
    # per cent high at this 10x contrast, on top of ~2 per cent scatter.
    assert np.mean(sig_est / sigmas, axis=0) == pytest.approx([1.0, 1.0], abs=0.05)
    assert np.all(np.abs(sig_est / sigmas - 1.0) < 0.15)
    for c in (0, 1):
        scatter = np.std(rv['none'][:, c])
        assert scatter / np.std(rv['true'][:, c]) == pytest.approx(1.0, abs=0.15)
        assert 0.8 < scatter / np.mean(rv['none'][:, 2 + c]) < 1.25


def test_ccf2d_sigma_none_with_homogeneous_noise_matches_the_true_sigma():
    # Equal noise in both orders: the per-order estimates differ only by their
    # own statistical scatter (~1.5 per cent), so the fit must match the one at
    # the true sigma to a small fraction of its error.
    sigmas = np.array([0.01, 0.01])
    rv, sig_est = _per_order_noise_mc(sigmas, n_real=6, seed=8)
    assert np.all(np.abs(sig_est / sigmas - 1.0) < 0.1)
    assert np.all(np.abs(rv['none'][:, :2] - rv['true'][:, :2]) < 0.2 * rv['true'][:, 2:])
    assert np.all(np.abs(rv['none'][:, 2:] / rv['true'][:, 2:] - 1.0) < 0.05)


@pytest.mark.parametrize('kind', ['flat', 'short'])
def test_ccf2d_sigma_none_falls_back_to_the_common_noise(kind):
    # A flat order (masked and filled with continuum: zero residual) or one
    # with too few pixels to measure its own noise gets the common estimate,
    # with a warning, instead of an infinite or wildly noisy weight.
    spec, tmpl1, tmpl2 = _sparta_case(n_ord=2)
    rng = np.random.default_rng(9)
    lo, hi = (4750.0, 4950.0) if kind == 'flat' else (4800.0, 4800.4)
    extra, _ = log_lambda_grid(lo, hi, 1.0)
    extra_flux = (np.ones_like(extra) if kind == 'flat'
                  else 1.0 + rng.normal(0.0, 0.01, extra.size))
    noisy = Spectrum(wv=list(spec.wv) + [extra],
                     sp=[f + rng.normal(0.0, 0.01, f.size) for f in spec.sp] + [extra_flux])
    noisy.Info.update(GridType='log', GridDelta=1.0 * u.km / u.s)
    with pytest.warns(UserWarning, match="common noise estimate"):
        ccf = CCF2d().CrossCorrelateSpec(noisy, tmpl1, tmpl2, VelBound=150, dvel=2.0)
    assert [w['sigma_fallback'] for w in ccf.windows] == [False, False, True]
    assert ccf.windows[2]['sigma'] == pytest.approx(0.01, rel=0.1)
    assert ccf.Corr2d['sigma'][2] == ccf.windows[2]['sigma']
    assert abs(ccf.Corr2d['RV1'] - 60.0) < 1.0
    assert abs(ccf.Corr2d['RV2'] + 95.0) < 1.0
    assert np.isfinite(ccf.Corr2d['eRV1']) and np.isfinite(ccf.Corr2d['eRV2'])


def test_ccf2d_explicit_per_order_sigma_is_reported_per_window():
    spec, tmpl1, tmpl2 = _sparta_case(n_ord=2)
    ccf = CCF2d().CrossCorrelateSpec(spec, tmpl1, tmpl2, VelBound=150, dvel=2.0,
                                     sigma=[0.01, 0.02])
    assert [w['sigma'] for w in ccf.windows] == [0.01, 0.02]
    assert not any(w['sigma_fallback'] for w in ccf.windows)



# =============================================================================
# Velocity conventions
# =============================================================================
# A component at v is shifted by c ln(D(v)) along the log-lambda axis, with
# D = 1 + v/c ('cz', the default) or sqrt((1+b)/(1-b)) ('relativistic'), in
# units of the order's ACTUAL ln-step: on an InterpolateSpectrum grid
# (ln-step ln(1 + dv/c), GridDelta = dv) taking the step to be dv scaled every
# velocity by 1 + dv/2c (0.25 m/s at 150 km/s for dv = 1 km/s).
_CONVENTIONS = ('cz', 'relativistic')


def _doppler(v, convention):
    """lambda_obs / lambda_rest, written out from the two definitions."""
    b = v / _C_KMS
    return 1.0 + b if convention == 'cz' else np.sqrt((1.0 + b) / (1.0 - b))


def _velocity_forest(seed, n_lines, sigma_kms, lam_lo=5000.0, lam_hi=5060.0):
    """Rest-frame Gaussian-line forest (in velocity) as a function of lambda,
    its lines 400 km/s clear of the order edges."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(400.0, _C_KMS * np.log(lam_hi / lam_lo) - 400.0, n_lines)
    depths = rng.uniform(0.05, 0.6, n_lines)

    def flux(wave):
        x = _C_KMS * np.log(np.asarray(wave, float) / lam_lo)
        return np.prod(1.0 - depths * np.exp(-0.5 * ((x[:, None] - centres) / sigma_kms) ** 2),
                       axis=1)
    return flux


def _convention_case(v1, v2, convention, a1=0.7, a2=0.4, dv=1.0):
    """Analytic noise-free SB2 on an InterpolateSpectrum-style log grid, with
    templates sampled on that same grid (so their resampling is the identity)."""
    wave = np.exp(np.arange(np.log(5000.0), np.log(5060.0), np.log1p(dv / _C_KMS)))
    t1, t2 = _velocity_forest(21, 60, 4.0), _velocity_forest(22, 40, 6.0)
    flux = (1.0 + a1 * (t1(wave / _doppler(v1, convention)) - 1.0)
            + a2 * (t2(wave / _doppler(v2, convention)) - 1.0))
    spec = Spectrum(wv=[wave], sp=[flux], name='SB2-conv')
    spec.Info['GridType'], spec.Info['GridDelta'] = 'log', dv * u.km / u.s
    return (spec, Template(wavelengths=wave, spectrum=t1(wave)),
            Template(wavelengths=wave, spectrum=t2(wave)))


def _fit_convention_case(v1, v2, fit_convention, inject_convention=None):
    """CCF2d.Corr2d on a 0.01 km/s map around the truth (off its nodes), so
    the fine map's step is 1 m/s and the refined peak is the mapping's."""
    spec, tmpl1, tmpl2 = _convention_case(v1, v2, inject_convention or fit_convention)
    grid = np.arange(-15, 16) * 0.01
    return CCF2d().CrossCorrelateSpec(
        spec, tmpl1, tmpl2, v1_grid=v1 + 0.0037 + grid, v2_grid=v2 - 0.0042 + grid,
        sigma=0.01, alpha=0.0, continuum=1.0,
        velocity_convention=fit_convention).Corr2d


@pytest.mark.parametrize('conv', _CONVENTIONS)
@pytest.mark.parametrize('v1, v2', [(150.0, -150.0), (-150.0, 150.0), (100.0, -100.0),
                                    (-100.0, 100.0), (50.0, -50.0), (-50.0, 50.0),
                                    (10.0, -10.0), (-10.0, 10.0)])
def test_ccf2d_recovers_sb2_injections_in_either_convention(v1, v2, conv):
    r = _fit_convention_case(v1, v2, conv)
    assert r['velocity_convention'] == conv
    assert abs(r['RV1'] - v1) * 1e3 < 0.05      # m/s
    assert abs(r['RV2'] - v2) * 1e3 < 0.05
    assert r['a1_best'] == pytest.approx(0.7, abs=1e-4)
    assert r['a2_best'] == pytest.approx(0.4, abs=1e-4)


def test_ccf2d_cz_and_relativistic_differ_by_v2_over_2c():
    v1, v2 = 150.0, -100.0
    cz = _fit_convention_case(v1, v2, 'cz')
    rel = _fit_convention_case(v1, v2, 'relativistic', inject_convention='cz')
    assert cz['RV1'] - rel['RV1'] == pytest.approx(v1 ** 2 / (2 * _C_KMS), rel=3e-3)
    assert cz['RV2'] - rel['RV2'] == pytest.approx(v2 ** 2 / (2 * _C_KMS), rel=3e-3)


def test_ccf2d_measures_each_orders_actual_step():
    spec, tmpl1, tmpl2 = _convention_case(50.0, -50.0, 'cz')
    ccf = CCF2d().CrossCorrelateSpec(spec, tmpl1, tmpl2, VelBound=80, dvel=4.0, sigma=0.01)
    # np.arange quantises the ln-step to an ulp of ln(lambda): ~1e-10 relative.
    assert ccf.windows[0]['dv'] == pytest.approx(_C_KMS * np.log1p(1.0 / _C_KMS), rel=1e-9)
    assert 1.0 - ccf.windows[0]['dv'] == pytest.approx(1.0 / (2 * _C_KMS), rel=1e-3)


def test_ccf2d_default_convention_is_cz():
    sig = inspect.signature(CCF2d.CrossCorrelateSpec)
    assert sig.parameters['velocity_convention'].default == 'cz'
    spec, tmpl1, tmpl2 = _sparta_case()
    ccf = CCF2d().CrossCorrelateSpec(spec, tmpl1, tmpl2, VelBound=120, dvel=4.0, sigma=0.01)
    assert ccf.Corr2d['velocity_convention'] == 'cz'
    # BestFitSpectrum models the composite in the convention of the fit.
    wv, model, data = ccf.BestFitSpectrum(order=0)
    assert np.all(np.isfinite(model))


def test_ccf2d_refuses_an_unknown_convention():
    spec, tmpl1, tmpl2 = _sparta_case()
    with pytest.raises(ValueError, match='convention'):
        CCF2d().CrossCorrelateSpec(spec, tmpl1, tmpl2, VelBound=120, dvel=4.0,
                                   velocity_convention='classical')
