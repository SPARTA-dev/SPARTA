"""Golden-master + primitive tests for the USuRPER spectra path.

USuRPER's spectral distance is the zero-lag (unshifted, pointwise) correlation
of resampled + normalized spectra -- no Fourier transform, no velocity scan.
This pins that path across the spectral_ops refactor that removed the Template
coupling. Golden captured 2026-07-22 before the refactor; the refactor
reproduced it bit-for-bit.
"""
import numpy as np
from types import SimpleNamespace
import pytest

from sparta import spectral_ops as so
from sparta.Auxil.TimeSeries import TimeSeries
from sparta.Auxil.PeriodicityDetector import PeriodicityDetector

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
    det.calc_USURPER(calc_unbiased_flag=True)
    return det


def test_usurper_matches_golden(usurper):
    power = np.asarray(usurper.results_power["USURPER"])
    # atol 1e-7: the distance matrix is now a BLAS Gram product (Stage 4
    # vectorization) whose summation order differs from the original per-pair
    # np.sum at the ~8th significant figure. Physics/peak unchanged.
    np.testing.assert_allclose(power, GOLDEN_USURPER, rtol=1e-6, atol=1e-7)


def test_usurper_recovers_period(usurper):
    freqs = np.asarray(usurper.results_frequency["USURPER"])
    power = np.asarray(usurper.results_power["USURPER"])
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
