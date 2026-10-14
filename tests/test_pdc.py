"""Golden-master tests for the scalar PDC periodogram.

Pins the current numeric output of the packaged PDC path so later refactors
(vectorizing USURPER_functions, dropping numba) can be proven behavior-
preserving. Golden values captured 2026-07-22 on the transitional stack
(numpy 1.23.5); see PROJECT_LOG.md.
"""
import numpy as np
import pytest

from sparta.Auxil.TimeSeries import TimeSeries
from sparta.Auxil.PeriodicityDetector import PeriodicityDetector
from sparta.USURPER.USURPER_functions import unbiased_u_centering

# Deterministic dataset: 12 irregular epochs, pure sine at P = 3 d.
TIMES = [0.0, 1.3, 2.1, 3.7, 5.2, 6.9, 8.4, 10.1, 12.6, 14.0, 17.3, 19.8]
PERIOD = 3.0
VALS = [float(np.sin(2 * np.pi * t / PERIOD)) for t in TIMES]

GOLDEN_POWER = np.array([
    -0.10348584931, -0.101106174198, -0.11169115221, -0.138116963059,
    -0.082663558562, 0.135678091263, 0.711136454507, -0.11632209148,
    0.124224461486, -0.131032238939, -0.122167008023, -0.090888138587,
    -0.04235436056, -0.149691177548, -0.151892935537, 0.125817084689,
    -0.171079940457, 0.682960042922, 0.044393994947,
])


@pytest.fixture(scope="module")
def pdc():
    ts = TimeSeries(size=len(TIMES), times=list(TIMES), vals=list(VALS))
    det = PeriodicityDetector(time_series=ts, freq_range=(0.05, 1.0),
                              periodogram_grid_resolution=20)
    det.calc_PDC(calc_unbiased_flag=True)
    return det


def test_pdc_periodogram_matches_golden(pdc):
    power = np.asarray(pdc.results_power["PDC_unbiased"])
    np.testing.assert_allclose(power, GOLDEN_POWER, rtol=1e-8, atol=1e-10)


def test_pdc_recovers_injected_period(pdc):
    freqs = np.asarray(pdc.results_frequency["PDC_unbiased"])
    power = np.asarray(pdc.results_power["PDC_unbiased"])
    peak = freqs[np.argmax(power)]
    assert peak == pytest.approx(0.35)   # nearest grid point to f = 1/3
    assert power.max() > 0.5             # clear, dominant peak


def test_pdc_biased_matches_golden():
    # The biased PDC path crashed under @njit (it took a Python object) until
    # Stage 4 vectorized calc_PDC. This pins the now-working output.
    ts = TimeSeries(size=len(TIMES), times=list(TIMES), vals=list(VALS))
    det = PeriodicityDetector(time_series=ts, freq_range=(0.05, 1.0),
                              periodogram_grid_resolution=20)
    det.calc_PDC(calc_biased_flag=True, calc_unbiased_flag=False)
    golden = np.array([
        0.166999289578, 0.161718905768, 0.151392714048, 0.115604741635,
        0.157741292807, 0.321436592725, 0.758153188588, 0.131493345769,
        0.318711174608, 0.129253575907, 0.135130104094, 0.170396683197,
        0.186017495755, 0.12021572498, 0.10820334488, 0.313639457286,
        0.099999809231, 0.728898342886, 0.238452656323,
    ])
    power = np.asarray(det.results_power["PDC_biased"])
    np.testing.assert_allclose(power, golden, rtol=1e-6, atol=1e-7)


def test_unbiased_u_centering_golden():
    m = np.array([[0., 1., 2., 3.],
                  [1., 0., 1., 2.],
                  [2., 1., 0., 1.],
                  [3., 2., 1., 0.]])
    expected = np.array([[0., -2/3, 1/3, 1/3],
                         [-2/3, 0., 1/3, 1/3],
                         [1/3, 1/3, 0., -2/3],
                         [1/3, 1/3, -2/3, 0.]])
    np.testing.assert_allclose(unbiased_u_centering(m, 4), expected, atol=1e-12)


# --- input validation (audited bug fixes) -----------------------------------

@pytest.mark.parametrize("n", [2, 3])
def test_unbiased_pdc_rejects_n_le_3(n):
    # U-centering's 1/(n(n-3)) and /(n-2) factors are singular for n <= 3:
    # n=2 used to give NaN power, n=3 rounding noise.
    ts = TimeSeries(size=n, times=TIMES[:n], vals=VALS[:n])
    det = PeriodicityDetector(time_series=ts, freq_range=(0.05, 1.0),
                              periodogram_grid_resolution=20)
    with pytest.raises(ValueError, match="n <= 3"):
        det.calc_PDC(calc_unbiased_flag=True)


def test_unbiased_u_centering_rejects_n_le_3():
    with pytest.raises(ValueError, match="n <= 3"):
        unbiased_u_centering(np.zeros((3, 3)), 3)


@pytest.mark.parametrize("f_min", [0.0, -0.1])
def test_frequency_grid_must_exclude_zero(f_min):
    ts = TimeSeries(size=len(TIMES), times=list(TIMES), vals=list(VALS))
    with pytest.raises(ValueError, match="1/day"):
        PeriodicityDetector(time_series=ts, freq_range=(f_min, 1.0),
                            periodogram_grid_resolution=20)


def test_frequency_grid_must_be_increasing():
    ts = TimeSeries(size=len(TIMES), times=list(TIMES), vals=list(VALS))
    with pytest.raises(ValueError, match="freq_range"):
        PeriodicityDetector(time_series=ts, freq_range=(1.0, 0.5),
                            periodogram_grid_resolution=20)


@pytest.mark.parametrize("kwargs", [{}, {"freq_range": (0.05, 1.0)},
                                    {"periodogram_grid_resolution": 20}])
def test_detector_requires_frequency_grid(kwargs):
    ts = TimeSeries(size=len(TIMES), times=list(TIMES), vals=list(VALS))
    with pytest.raises(TypeError, match="freq_range"):
        PeriodicityDetector(time_series=ts, **kwargs)


def test_pdc_accepts_float32_values():
    vals32 = list(np.asarray(VALS, dtype=np.float32))
    ts = TimeSeries(size=len(TIMES), times=list(TIMES), vals=vals32)
    det = PeriodicityDetector(time_series=ts, freq_range=(0.05, 1.0),
                              periodogram_grid_resolution=20)
    det.calc_PDC(calc_unbiased_flag=True)
    power = np.asarray(det.results_power["PDC_unbiased"])
    np.testing.assert_allclose(power, GOLDEN_POWER, atol=1e-5)


def test_gls_accepts_integer_values():
    ivals = [int(round(10 * v)) for v in VALS]
    ts = TimeSeries(size=len(TIMES), times=list(TIMES), vals=ivals)
    det = PeriodicityDetector(time_series=ts, freq_range=(0.05, 1.0),
                              periodogram_grid_resolution=20)
    _, power = det.calc_GLS()
    assert np.all(np.isfinite(power))


def test_timeseries_rejects_spectrum_ops_on_float32_scalars():
    ts = TimeSeries(times=[0.0, 1.0], vals=[np.float32(1.0), np.float32(2.0)])
    with pytest.raises(ValueError, match="spectrum time-series only"):
        ts.calc_rv_against_template(ts.vals, template=None)


def test_timeseries_size_must_match_times():
    with pytest.raises(ValueError, match="size"):
        TimeSeries(size=5, times=list(TIMES), vals=list(VALS))


def test_timeseries_size_defaults_to_len_times():
    assert TimeSeries(times=list(TIMES), vals=list(VALS)).size == len(TIMES)
