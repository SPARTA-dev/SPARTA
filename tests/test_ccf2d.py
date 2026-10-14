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


def _line_list(seed, n=25):
    rng = np.random.default_rng(seed)
    return rng.uniform(_LO + 3.0, _HI - 3.0, n), rng.uniform(0.1, 0.6, n)


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
    # The default sigma=None fits a common noise level (sqrt(chi2_red) at unit
    # sigma) and scales the errors with it: they must match the errors
    # computed with the true sigma.
    v1, v2, sigma = 37.3, -52.6, 0.01
    wave, flux0, tmpl1, tmpl2 = _narrow_case(v1, v2)
    g1, g2 = _local_grids(v1, v2)
    rng = np.random.default_rng(1)
    for _ in range(4):
        spec = Spectrum(wv=[wave], sp=[flux0 + rng.normal(0.0, sigma, wave.size)])
        est = CCF2d().CrossCorrelateSpec(spec, tmpl1, tmpl2, v1_grid=g1, v2_grid=g2).Corr2d
        ref = CCF2d().CrossCorrelateSpec(spec, tmpl1, tmpl2, v1_grid=g1, v2_grid=g2,
                                         sigma=sigma).Corr2d
        assert est['sigma_from_fit'] and not ref['sigma_from_fit']
        assert est['sigma'] == pytest.approx(sigma, rel=0.1)
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
    assert est['chi2_map'] == pytest.approx(est['chi2_best'] * est['sigma'] ** 2, rel=1e-9)
    assert ref['chi2_map'] == ref['chi2_best']
    # The maps sit on chi2_map's scale: their minimum is next to it.
    for r in (est, ref):
        assert r['fine']['chi2'].min() == pytest.approx(r['chi2_map'], rel=1e-3)


def test_ccf2d_refuses_native_steps_that_do_not_match_the_orders():
    spec, tmpl1, tmpl2 = _sparta_case(n_ord=2)
    spec.Info['NativeStep'] = [1.0]
    with pytest.raises(ValueError, match="NativeStep"):
        CCF2d().CrossCorrelateSpec(spec, tmpl1, tmpl2, VelBound=120, dvel=4.0)
