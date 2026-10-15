"""Golden-master + primitive tests for the USuRPER spectra path.

USuRPER's spectral distance is the zero-lag (unshifted, pointwise) correlation
of resampled + normalized spectra -- no Fourier transform, no velocity scan.
This pins that path across the spectral-ops refactor that removed the Template
coupling. Golden captured 2026-07-22 before the refactor; the refactor
reproduced it bit-for-bit.
"""
import numpy as np
from types import SimpleNamespace
import pytest

import unicor as so
from sparta.timeseries import TimeSeries
from sparta.periodogram import PeriodicityDetector

C = 299792.458
WV = np.linspace(5495.0, 5505.0, 600)
LAM0 = 5500.0
TIMES = [0.0, 1.3, 2.1, 3.7, 5.2, 6.9, 8.4, 10.1]
P, K = 3.0, 30.0   # period (d), RV semi-amplitude (km/s)

GOLDEN_USURPER = np.array([
    -0.169731980725, -0.193478291821, -0.077062704921, -0.264894055084,
    -0.047192473574, 0.73671879451, 0.540573415678, -0.237841435141,
    -0.171423823482, -0.245463503442, -0.288395621389, -0.145541876037,
    0.235208770245, -0.13099122592, -0.232810121313, 0.091913819099,
    -0.308862788786, 0.565507878636, 0.708840592639,
])


def _spectrum(t):
    v = K * np.sin(2 * np.pi * t / P)
    center = LAM0 * (1 + v / C)
    sp = 1.0 - 0.6 * np.exp(-0.5 * ((WV - center) / 0.08) ** 2)
    return SimpleNamespace(wv=[WV.copy()], sp=[sp])


@pytest.fixture(scope="module")
def usurper():
    vals = [_spectrum(t) for t in TIMES]
    ts = TimeSeries(size=len(TIMES), times=list(TIMES), vals=vals)
    det = PeriodicityDetector(time_series=ts, freq_range=(0.05, 1.0),
                              periodogram_grid_resolution=20)
    det.calc_usurper(calc_unbiased_flag=True)
    return det


def test_usurper_matches_golden(usurper):
    power = np.asarray(usurper.results_power["USURPER_unbiased"])
    # atol 1e-7: the distance matrix is now a BLAS Gram product (Stage 4
    # vectorization) whose summation order differs from the original per-pair
    # np.sum at the ~8th significant figure. Physics/peak unchanged.
    np.testing.assert_allclose(power, GOLDEN_USURPER, rtol=1e-6, atol=1e-7)


def test_usurper_recovers_period(usurper):
    freqs = np.asarray(usurper.results_frequency["USURPER_unbiased"])
    power = np.asarray(usurper.results_power["USURPER_unbiased"])
    assert freqs[np.argmax(power)] == pytest.approx(0.3)   # nearest grid to f=1/3
    assert power.max() > 0.5


def test_doppler_shift_wave_classical():
    wave = np.array([5000.0, 5500.0, 6000.0])
    np.testing.assert_allclose(so.doppler_shift_wave(wave, 30.0), wave * (1 + 30.0 / C))


def test_zero_lag_and_normalize():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    n = so.normalize(x)
    assert n.mean() == pytest.approx(0.0)                        # mean-subtracted
    assert so.zero_lag_correlation(n, n) == pytest.approx(1.0)   # unit L2 norm


# --- input validation and partial-periodogram purity (audited bug fixes) -----

# First-call shift/shape partial periodograms, captured 2026-10-14 before the
# in-place wavelength mutation was removed; re-blessed 2026-10-15 for the exact
# shift to rest, lambda / (1 + v/c) instead of lambda * (1 - v/c).
GOLDEN_SHIFT = np.array([-0.238213350007, -0.278300015605, -0.051830271299, -0.271295567571,
    -0.114516819893, 0.692136783685, 0.674247310939, -0.229188954418,
    -0.060332612932, -0.166785783427, -0.249338399728, -0.220732854592,
    0.12682665642, -0.190099710461, -0.267692813893, 0.049847126092,
    -0.350919169489, 0.623686543626, 0.793442334662,
])
GOLDEN_SHAPE = np.array([-0.164579027238, -0.074902551857, 0.034598661318, 0.066760320077,
    -0.052617709663, -0.124510476593, 0.032610698995, -0.226390278085,
    0.136983363077, 0.486622848435, 0.250311762724, 0.015127698725,
    -0.088076602616, -0.110778366716, -0.041389658279, 0.134477335384,
    -0.266113165045, -0.026005682369, -0.040635046976,
])


def _detector(vals, times=TIMES, **ts_kwargs):
    ts = TimeSeries(size=len(times), times=list(times), vals=vals, **ts_kwargs)
    return PeriodicityDetector(time_series=ts, freq_range=(0.05, 1.0),
                               periodogram_grid_resolution=20)


def test_usurper_biased_and_unbiased_together():
    det = _detector([_spectrum(t) for t in TIMES])
    det.calc_usurper(calc_biased_flag=True, calc_unbiased_flag=True)
    assert {"USURPER_unbiased", "USURPER_biased"} <= set(det.results_power)
    np.testing.assert_allclose(det.results_power["USURPER_unbiased"], GOLDEN_USURPER,
                               rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("kind, golden", [("shift", GOLDEN_SHIFT),
                                          ("shape", GOLDEN_SHAPE)])
def test_partial_periodogram_does_not_mutate_input(kind, golden):
    vals = [_spectrum(t) for t in TIMES]
    vr = [K * np.sin(2 * np.pi * t / P) for t in TIMES]
    det = _detector(vals, calculated_vrad_list=vr)

    det.calc_partial_periodogram(partial_type=kind)
    first = np.array(det.results_power["partial_" + kind])
    det.calc_partial_periodogram(partial_type=kind)
    second = np.array(det.results_power["partial_" + kind])

    np.testing.assert_allclose(first, golden, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(first, second)
    for s in vals:
        np.testing.assert_array_equal(s.wv[0], WV)


def test_usurper_rejects_n_le_3():
    det = _detector([_spectrum(t) for t in TIMES[:3]], times=TIMES[:3])
    with pytest.raises(ValueError, match="n <= 3"):
        det.calc_usurper()


def test_usurper_rejects_non_overlapping_spectra():
    vals = [_spectrum(t) for t in TIMES]
    vals[3] = SimpleNamespace(wv=[WV + 100.0], sp=vals[3].sp)
    with pytest.raises(ValueError, match="overlap"):
        _detector(vals).calc_usurper()


def test_usurper_rejects_decreasing_wavelengths():
    vals = [_spectrum(t) for t in TIMES]
    vals[2] = SimpleNamespace(wv=[WV[::-1]], sp=[vals[2].sp[0][::-1]])
    with pytest.raises(ValueError, match="increasing"):
        _detector(vals).calc_usurper()


def test_usurper_rejects_multi_order_spectra():
    vals = [_spectrum(t) for t in TIMES]
    vals[0] = SimpleNamespace(wv=[WV, WV + 20.0], sp=[vals[0].sp[0]] * 2)
    with pytest.raises(ValueError, match="single order"):
        _detector(vals).calc_usurper()


def test_shape_alone_matches_shape_after_shift():
    # Each partial periodogram builds its own distance matrices: the result
    # cannot depend on which one ran before.
    vr = [K * np.sin(2 * np.pi * t / P) for t in TIMES]
    alone = _detector([_spectrum(t) for t in TIMES], calculated_vrad_list=vr)
    alone.calc_partial_periodogram(partial_type="shape")
    after = _detector([_spectrum(t) for t in TIMES], calculated_vrad_list=vr)
    after.calc_partial_periodogram(partial_type="shift")
    after.calc_partial_periodogram(partial_type="shape")
    np.testing.assert_array_equal(alone.results_power["partial_shape"],
                                  after.results_power["partial_shape"])


def test_partial_periodogram_rejects_an_unknown_type():
    det = _detector([_spectrum(t) for t in TIMES],
                    calculated_vrad_list=[0.0] * len(TIMES))
    with pytest.raises(ValueError, match="partial_type"):
        det.calc_partial_periodogram(partial_type="width")


@pytest.mark.parametrize("convention", ["cz", "relativistic"])
def test_partial_periodogram_shifts_each_spectrum_exactly_to_rest(monkeypatch, convention):
    # The shape distance compares the spectra at rest: lambda / doppler_factor(v),
    # the exact inverse of the shift the RV measures -- not lambda * (1 - v/c),
    # which leaves a residual shift of ~v**2/c (33 m/s at 100 km/s).
    import sparta.pdc as pdc
    seen = {}
    spectral_distance_matrix = pdc.spectral_distance_matrix

    def spy(waves, fluxes):
        seen["waves"] = [np.array(w) for w in waves]
        return spectral_distance_matrix(waves, fluxes)

    monkeypatch.setattr(pdc, "spectral_distance_matrix", spy)
    vr = [150.0 * np.sin(2 * np.pi * t / P) for t in TIMES]          # km/s
    det = _detector([_spectrum(t) for t in TIMES], calculated_vrad_list=vr)
    det.calc_partial_periodogram(partial_type="shape", velocity_convention=convention)
    assert len(seen["waves"]) == len(TIMES)
    for w, v in zip(seen["waves"], vr):
        np.testing.assert_array_equal(w, WV / so.doppler_factor(v, convention))


def test_partial_periodogram_rejects_an_unknown_convention():
    vr = [K * np.sin(2 * np.pi * t / P) for t in TIMES]
    det = _detector([_spectrum(t) for t in TIMES], calculated_vrad_list=vr)
    with pytest.raises(ValueError, match="convention"):
        det.calc_partial_periodogram(partial_type="shape", velocity_convention="classical")


def test_spectra_shifted_to_rest_by_their_rvs_have_no_shape_distance(monkeypatch):
    # The physics behind the exact shift: pure Doppler copies of one spectrum,
    # sampled finely enough for a v**2/c residual to show, are identical at
    # rest. The old lambda * (1 - v/c) left a shape distance of ~0.012 here.
    import sparta.pdc as pdc
    k = 150.0                                                    # km/s
    wv = np.linspace(5490.0, 5510.0, 20000)
    vr = [k * np.sin(2 * np.pi * t / P) for t in TIMES]

    def spectrum(v):
        lam = 5500.0 * (1 + v / C)
        return SimpleNamespace(wv=[wv.copy()],
                               sp=[1.0 - 0.6 * np.exp(-0.5 * ((wv - lam) / 0.08) ** 2)])

    seen = {}
    spectral_distance_matrix = pdc.spectral_distance_matrix

    def spy(waves, fluxes):
        seen["d"] = spectral_distance_matrix(waves, fluxes)
        return seen["d"]

    monkeypatch.setattr(pdc, "spectral_distance_matrix", spy)
    det = _detector([spectrum(v) for v in vr], calculated_vrad_list=vr)
    det.calc_partial_periodogram(partial_type="shape")
    assert np.max(np.abs(seen["d"])) < 2e-3


def test_partial_periodogram_names_the_epochs_without_an_rv():
    vr = [K * np.sin(2 * np.pi * t / P) for t in TIMES]
    vr[2] = np.nan
    det = _detector([_spectrum(t) for t in TIMES], calculated_vrad_list=vr)
    with pytest.raises(ValueError, match=r"epochs \[2\]"):
        det.calc_partial_periodogram(partial_type="shape")
