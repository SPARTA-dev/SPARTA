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
# in-place wavelength mutation was removed: the fix must leave these unchanged.
GOLDEN_SHIFT = np.array([
    -0.236595548786, -0.274103449129, -0.054241104738, -0.271579752874,
    -0.11600031873, 0.692773586682, 0.674448731687, -0.227845559285,
    -0.061452377936, -0.170206489286, -0.251746235987, -0.218864896597,
    0.129640278, -0.185947595114, -0.270080863928, 0.049667558497,
    -0.350276766428, 0.625301315036, 0.79369598832,
])
GOLDEN_SHAPE = np.array([
    -0.159874821791, -0.051405776858, 0.022280581774, 0.069173322256,
    -0.061893425622, -0.125880491952, 0.031243117311, -0.225023433041,
    0.135581338745, 0.4858111466, 0.24296085895, 0.028768299842,
    -0.074752196309, -0.090863815946, -0.055156229176, 0.137577140206,
    -0.267697071988, -0.011293675649, -0.044850387649,
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
