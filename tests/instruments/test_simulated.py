"""The simulated instrument (sparta.instruments.simulated): injection-recovery
through the same Observations path as archive files.

Noiseless spectra must give back the injected RVs (UNICOR and TODCOR, both
Doppler conventions, barycentric and topocentric products, native and archive
grids); noisy ones must scatter as the reported uncertainties say. PHOENIX
models come from the bundled cache only (allow_download=False is the default).
"""
import sys
import warnings

import numpy as np
import pytest

from sparta.Observations import Observations
from sparta.TODCOR.CCF2d import CCF2d
from sparta.UNICOR.CCF1d import CCF1d
from sparta.UNICOR.Template import Template
from sparta import instruments
from sparta.instruments import Component, Simulation
from sparta.instruments.simulated import (component_spectra, component_templates, simulate,
                                          upper_envelope)
from unicor import dsp

SUN_LIKE = dict(teff=5800, logg=3.5, feh=0.5, vsini=3.0)       # bundled lte05800-3.50+0.5
K_DWARF = dict(teff=4900, logg=4.5, vsini=2.0)                  # bundled lte04900-4.50-0.0
HARPS_NATIVE = instruments.get("HARPS").instrument.native_step_kms


def _ccf1d_rvs(obs, template, vel_bound, convention="cz"):
    return np.array([CCF1d().CrossCorrelateSpec(sp, template, dv=0.1, VelBound=vel_bound,
                                                velocity_convention=convention)
                     .CombineCCFs().CorrCombined["RV"] for sp in obs.spec_list])


@pytest.mark.parametrize("convention", ["cz", "relativistic"])
def test_noiseless_sb1_recovers_the_injected_rvs(convention):
    rv = np.array([-50.0, -12.345, 0.0, 7.7, 50.0])
    sim = Simulation(like="HARPS", components=[SUN_LIKE], times=np.arange(5.0), rv_kms=rv,
                     velocity_convention=convention)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # a simulator is not an unverified loader
        obs = Observations(survey="SIMULATED", simulation=sim)
    np.testing.assert_array_equal(obs.truth.rv_kms[:, 0], rv)
    assert obs.truth.velocity_convention == convention
    measured = _ccf1d_rvs(obs, component_templates(sim)[0], 60, convention)
    np.testing.assert_allclose(measured, rv, rtol=0, atol=1e-3)        # < 1 m/s


def test_records_follow_the_emulated_product():
    sim = Simulation(like="HARPS", components=[SUN_LIKE], times=[2458000.25, 2458001.5],
                     rv_kms=[1.0, 2.0])
    records, truth = simulate(sim)
    harps = instruments.get("HARPS")
    for k, rec in enumerate(records):
        assert (rec.survey, rec.specsys, rec.wavelength_frame) == ("SIMULATED", "BARYCENT", "air")
        assert rec.extras == {"like": "HARPS", "epoch": k, "as_product": False}
        assert (rec.bcv, rec.native_step_kms, rec.path) == (0.0, None, None)
        (lo, hi), = harps.recommended.window_A,
        assert lo <= rec.wave[0][0] and rec.wave[0][-1] <= hi
        # Native pixels: log-lambda at the detector step.
        assert dsp.log_grid_step(rec.wave[0]) == pytest.approx(HARPS_NATIVE, rel=1e-9)
    np.testing.assert_array_equal(truth.times, [2458000.25, 2458001.5])
    assert truth.snr is None and records[0].snr == np.inf
    obs = Observations(survey="SIMULATED", simulation=sim)
    assert obs.time_list == [0.25, 1.5] and obs.file_list == []
    assert "NativeStep" not in obs.spec_list[0].Info


@pytest.mark.parametrize("convention", ["cz", "relativistic"])
def test_noiseless_sb2_through_ccf2d_recovers_both_components(convention):
    # TODCOR's Fourier model shifts the templates circularly within the window,
    # so lines crossing the window edges bias it on a line-crowded spectrum:
    # ~10 m/s at |v| ~ 45 km/s over 50 A, ~2 m/s over 200 A. Hence the wide
    # window and the stronger taper; continuum=1.0 is exact for these
    # normalised spectra (the median leaves the amplitudes ~0.5% low).
    rv = np.array([[30.0, -45.0], [-42.3, 51.7], [5.0, -8.0]])
    sim = Simulation(like="HARPS", windows_A=[(5000.0, 5200.0)], times=np.arange(3.0), rv_kms=rv,
                     components=[dict(SUN_LIKE, vsini=4.0, amplitude=0.7),
                                 dict(K_DWARF, amplitude=0.3)],
                     velocity_convention=convention)
    obs = Observations(survey="SIMULATED", simulation=sim)
    t1, t2 = component_templates(sim)
    for k, sp in enumerate(obs.spec_list):
        g1, g2 = (np.arange(round(v) - 5.0, round(v) + 6.0) for v in rv[k])
        fit = CCF2d().CrossCorrelateSpec(sp, t1, t2, dv=0.3, v1_grid=g1, v2_grid=g2,
                                         continuum=1.0, alpha=0.3,
                                         velocity_convention=convention).Corr2d
        np.testing.assert_allclose([fit["RV1"], fit["RV2"]], rv[k], rtol=0, atol=5e-3)
        np.testing.assert_allclose([fit["a1_best"], fit["a2_best"]], obs.truth.amplitudes,
                                   rtol=0, atol=2e-3)


def test_topocentric_truth_round_trips_with_bcv():
    # NRES-like: topocentric pixels observed at lambda D(v) / (1 + bcv/c);
    # Observations' BarycentricCorrection must bring them back to lambda D(v).
    rv, bcv = np.array([-50.0, 3.3, 50.0]), np.array([25.1, -17.4, 8.0])
    sim = Simulation(like="NRES", components=[dict(SUN_LIKE, vsini=6.0)], times=[1.0, 2.0, 3.0],
                     rv_kms=rv, bcv_kms=bcv, windows_A=[(5000.0, 5060.0), (5500.0, 5560.0)])
    records, _ = simulate(sim)
    obs = Observations(survey="SIMULATED", simulation=sim)
    assert obs.bcv == list(bcv) and obs.truth.wavelength_frame == obs.wavelength_frame == "vacuum"
    for rec, sp, b in zip(records, obs.spec_list, bcv):
        assert rec.specsys == "TOPOCENT" and len(sp.wv) == 2
        np.testing.assert_allclose(sp.wv[1], rec.wave[1] * (1 + b / dsp.C_KMS), rtol=1e-15)
    measured = _ccf1d_rvs(obs, component_templates(sim)[0], 70)
    np.testing.assert_allclose(measured, rv, rtol=0, atol=1e-3)


def test_as_product_rebins_onto_the_archive_grid_and_seeds_the_native_step():
    # 0.01 A at 6750 A is ~0.44 km/s: finer than the ~0.609 km/s HARPS pixels.
    sim = Simulation(like="HARPS", components=[SUN_LIKE], times=[0.0, 1.0], rv_kms=[-50.0, 50.0],
                     windows_A=[(6700.0, 6800.0)], as_product=True)
    records, _ = simulate(sim)
    for rec in records:
        np.testing.assert_allclose(np.diff(rec.wave[0]), 0.01, atol=1e-9)
        assert rec.native_step_kms == [HARPS_NATIVE]
    obs = Observations(survey="SIMULATED", simulation=sim)
    assert all(sp.Info["NativeStep"] == [HARPS_NATIVE] for sp in obs.spec_list)
    np.testing.assert_allclose(_ccf1d_rvs(obs, component_templates(sim)[0], 60), [-50.0, 50.0],
                               rtol=0, atol=1e-3)


def _unicor_mc(as_product, window, n_epochs=60, seed=1):
    rv = np.random.default_rng(seed).uniform(-20.0, 20.0, n_epochs)
    sim = Simulation(like="HARPS", components=[SUN_LIKE], times=np.arange(float(n_epochs)),
                     rv_kms=rv, windows_A=[window], snr=100.0, noise="constant", seed=seed,
                     as_product=as_product)
    obs = Observations(survey="SIMULATED", simulation=sim)
    template = component_templates(sim)[0]
    fits = [CCF1d().CrossCorrelateSpec(sp, template, dv=0.1, VelBound=30).CombineCCFs().CorrCombined
            for sp in obs.spec_list]
    residual = np.array([f["RV"] for f in fits]) - rv
    return np.std(residual, ddof=1) / np.mean([f["eRV"] for f in fits])


@pytest.mark.parametrize("as_product, window", [(False, (4990.0, 5090.0)),
                                                (True, (6700.0, 6800.0))],
                         ids=["native", "archive-grid"])
def test_unicor_scatter_matches_its_uncertainty(as_product, window):
    # White (constant) noise on the native pixels, the estimator's assumption.
    # Over 7 seeds x 50 epochs: mean std/eRV 1.03 (native) and 1.07 on the
    # HARPS archive grid with the seeded native step -- 1.25 without it, since
    # the 0.44 km/s archive pixels would count as independent. Photon noise
    # (quieter line cores) gives 0.89: eRV ~10% conservative.
    assert 0.8 < _unicor_mc(as_product, window) < 1.25


def test_todcor_scatter_matches_its_uncertainty_with_sigma_none():
    n_epochs, seed = 50, 1
    draw = np.random.default_rng(seed)
    rv = np.column_stack([draw.uniform(-30.0, 30.0, n_epochs), draw.uniform(-30.0, 30.0, n_epochs)])
    sim = Simulation(like="HARPS", windows_A=[(5000.0, 5200.0)], times=np.arange(float(n_epochs)),
                     rv_kms=rv, snr=25.0, noise="constant", seed=seed,
                     components=[dict(SUN_LIKE, vsini=4.0, amplitude=0.7),
                                 dict(K_DWARF, amplitude=0.3)])
    obs = Observations(survey="SIMULATED", simulation=sim)
    t1, t2 = component_templates(sim)
    residual, erv = [], []
    for k, sp in enumerate(obs.spec_list):
        g1, g2 = (np.arange(round(v) - 5.0, round(v) + 6.0) for v in rv[k])
        fit = CCF2d().CrossCorrelateSpec(sp, t1, t2, dv=0.3, v1_grid=g1, v2_grid=g2, continuum=1.0,
                                         alpha=0.3, keep_windows=False).Corr2d
        residual.append([fit["RV1"], fit["RV2"]] - rv[k])
        erv.append([fit["eRV1"], fit["eRV2"]])
    ratio = np.std(residual, axis=0, ddof=1) / np.mean(erv, axis=0)
    # Over 7 seeds: mean 0.94 (primary) and 1.02 (secondary).
    assert np.all((0.8 < ratio) & (ratio < 1.25)), ratio


def test_a_fixed_seed_reproduces_the_noise():
    kw = dict(like="HARPS", components=[SUN_LIKE], times=[0.0, 1.0], rv_kms=[0.0, 0.0], snr=50.0)
    first, _ = simulate(Simulation(seed=7, **kw))
    again, _ = simulate(Simulation(seed=7, **kw))
    other, _ = simulate(Simulation(seed=8, **kw))
    fresh, truth = simulate(Simulation(**kw))
    replay, _ = simulate(Simulation(seed=truth.seed, **kw))
    for a, b, c, d, e in zip(first, again, other, fresh, replay):
        np.testing.assert_array_equal(a.flux[0], b.flux[0])
        np.testing.assert_array_equal(d.flux[0], e.flux[0])
        assert not np.array_equal(a.flux[0], c.flux[0])
    # Each epoch has its own stream: same spectrum, different noise.
    assert not np.array_equal(first[0].flux[0], first[1].flux[0])


@pytest.mark.parametrize("noise", ["photon", "constant"])
def test_noise_level_is_per_native_pixel_at_the_continuum(noise):
    kw = dict(like="HARPS", components=[SUN_LIKE], times=[0.0], rv_kms=[0.0])
    clean, _ = simulate(Simulation(**kw))
    noisy, _ = simulate(Simulation(snr=200.0, noise=noise, seed=3, **kw))
    flux = clean[0].flux[0]
    pull = (noisy[0].flux[0] - flux) * 200.0
    if noise == "photon":
        pull = pull / np.sqrt(flux)
    assert np.std(pull) == pytest.approx(1.0, abs=0.03)


def test_allow_download_false_never_touches_the_network(monkeypatch):
    import astropy.utils.data
    template_module = sys.modules["sparta.UNICOR.Template"]   # the package exports the class

    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(astropy.utils.data, "download_file", no_network)
    monkeypatch.setattr(template_module, "download_file", no_network)
    simulate(Simulation(like="HARPS", components=[SUN_LIKE], times=[0.0], rv_kms=[0.0]))
    with pytest.raises(FileNotFoundError, match="allow_download=False"):
        simulate(Simulation(like="HARPS", components=[dict(teff=6100, logg=4.0)], times=[0.0],
                            rv_kms=[0.0]))


def test_a_template_component_matches_its_phoenix_model():
    sim = Simulation(like="HARPS", components=[SUN_LIKE], times=[0.0], rv_kms=[0.0])
    phoenix = Template(temp=5800, log_g=3.5, metal=0.5, alpha=0.0, min_val=4900.0,
                       max_val=5200.0, air=True, allow_download=False)
    own = Simulation(like="HARPS", components=[Component(template=phoenix, vsini=3.0)],
                     times=[0.0], rv_kms=[0.0])
    (w, f), = component_spectra(sim)[0]
    (w2, f2), = component_spectra(own)[0]
    np.testing.assert_array_equal(w, w2)
    # The envelope bins are anchored at the first pixel, so the two extraction
    # ranges normalise slightly differently (~0.5%).
    np.testing.assert_allclose(f2, f, rtol=0.01)


def test_upper_envelope_recovers_a_sloped_continuum():
    wave = np.linspace(5000.0, 5100.0, 20001)
    continuum = 2.0 + 0.01 * (wave - 5000.0)
    rng = np.random.default_rng(0)
    lines = sum(0.6 * np.exp(-0.5 * ((wave - c) / 0.05) ** 2) for c in rng.uniform(5000, 5100, 150))
    estimate = upper_envelope(wave, continuum * (1.0 - np.clip(lines, 0, 0.95)))
    np.testing.assert_allclose(estimate, continuum, rtol=2e-3)


@pytest.mark.parametrize("kwargs, match", [
    (dict(like="HARPS", bcv_kms=[3.0]), "not topocentric"),
    (dict(like="NRES", windows_A=[(5000.0, 5100.0)], as_product=True), "as_product"),
    (dict(like="TRES"), "resolving power and native step"),
    (dict(like="HARPS", rv_kms=[[0.0, 1.0]]), "rv_kms has shape"),
    (dict(like="HARPS", noise="poisson"), "noise"),
    (dict(like="HARPS", snr=-5.0), "snr"),
    (dict(like="HARPS", velocity_convention="classical"), "convention"),
    (dict(like="NRES"), "recommends no window"),
    (dict(like="HARPS", windows_A=[(5100.0, 5000.0)]), "lo < hi"),
])
def test_bad_simulations_are_refused(kwargs, match):
    base = dict(components=[SUN_LIKE], times=[0.0], rv_kms=[0.0])
    with pytest.raises(ValueError, match=match):
        Simulation(**{**base, **kwargs})


def test_a_component_needs_exactly_one_source():
    with pytest.raises(ValueError, match="either PHOENIX"):
        Component(vsini=2.0)
    with pytest.raises(ValueError, match="either PHOENIX"):
        Component(teff=5800, logg=4.5, template=Template())


def test_observations_route_simulations_only_through_the_simulated_survey(tmp_path):
    sim = Simulation(like="HARPS", components=[SUN_LIKE], times=[0.0], rv_kms=[0.0])
    with pytest.raises(ValueError, match="is simulated"):
        Observations(survey="SIMULATED")
    with pytest.raises(ValueError, match="is simulated"):
        Observations(survey="SIMULATED", simulation=sim, target_visits_lib=str(tmp_path))
    with pytest.raises(ValueError, match="simulated survey"):
        Observations(survey="HARPS", simulation=sim, target_visits_lib=str(tmp_path))
