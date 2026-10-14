"""Golden-master tests for the scalar PDC periodogram.

Pins the current numeric output of the packaged PDC path so later refactors
(vectorizing the PDC kernels, now sparta.pdc; dropping numba) can be proven behavior-
preserving. Golden values captured 2026-07-22 on the transitional stack
(numpy 1.23.5).
"""
import numpy as np
import pytest

from sparta.timeseries import TimeSeries
from sparta.periodogram import PeriodicityDetector
from sparta.pdc import unbiased_u_centering

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


def test_observations_rejects_spectrum_ops_on_float32_scalars():
    from sparta.observations import Observations
    ts = TimeSeries(times=[0.0, 1.0], vals=[np.float32(1.0), np.float32(2.0)])
    with pytest.raises(ValueError, match="spectrum time-series only"):
        Observations(time_series=ts).calc_rv_against_template(template=None)


def test_timeseries_size_must_match_times():
    with pytest.raises(ValueError, match="size"):
        TimeSeries(size=5, times=list(TIMES), vals=list(VALS))


def test_timeseries_size_defaults_to_len_times():
    assert TimeSeries(times=list(TIMES), vals=list(VALS)).size == len(TIMES)


# The error-aware PDC distance (Binnenfeld, Shahaf & Zucker 2024, Eq. 3, in its
# published form -- deliberately without a sqrt(pi) on the x*erf(x) term).
# Golden values captured 2026-10-14; the peak stays at P = 3 d.
ERRORS = [0.1 * (1 + i % 3) for i in range(len(TIMES))]
GOLDEN_POWER_ERRORS = np.array([
    -0.083359353882, -0.084396542771, -0.108568005069, -0.125086976581,
    -0.103570947238, 0.113113821665, 0.716716842014, -0.114300252979,
    0.131395392636, -0.138188082812, -0.123696802047, -0.089531393859,
    -0.033136650209, -0.141505792898, -0.143289838346, 0.126395626848,
    -0.165025252659, 0.716770766716, 0.030133323429,
])


def test_error_aware_pdc_matches_golden():
    ts = TimeSeries(times=list(TIMES), vals=list(VALS), errors=list(ERRORS))
    det = PeriodicityDetector(time_series=ts, freq_range=(0.05, 1.0),
                              periodogram_grid_resolution=20)
    det.calc_PDC(calc_unbiased_flag=True, errors=True)
    np.testing.assert_allclose(np.asarray(det.results_power["PDC_unbiased"]),
                               GOLDEN_POWER_ERRORS, rtol=1e-8, atol=1e-10)


# --- the detector holds state; sparta.pdc is pure (Phase-4 decoupling) -------

def test_pdc_power_is_a_pure_function_of_arrays():
    from sparta import pdc
    freqs = np.arange(0.05, 1.0, 1 / 20)
    a = pdc.unbiased_u_centering(pdc.scalar_distance_matrix(VALS), len(TIMES))
    np.testing.assert_allclose(pdc.pdc_power(TIMES, freqs, a), GOLDEN_POWER,
                               rtol=1e-8, atol=1e-10)


def test_partial_power_needs_the_unbiased_statistic():
    from sparta import pdc
    d = pdc.scalar_distance_matrix(VALS)
    a, c = pdc.partial_centering(d, d[::-1, ::-1])
    with pytest.raises(ValueError, match="unbiased"):
        pdc.pdc_power(TIMES, [0.3], a, unbiased=False, control=c)


def test_error_aware_distance_needs_one_error_per_value():
    from sparta import pdc
    with pytest.raises(ValueError, match="one error per value"):
        pdc.error_aware_distance_matrix(VALS, ERRORS[:-1])


def test_pdc_on_stored_rvs_leaves_the_series_values_alone():
    # A spectrum series' PDC runs on calculated_vrad_list; it used to swap
    # time_series.vals for the RVs during the call.
    from types import SimpleNamespace
    specs = [SimpleNamespace(wv=[np.arange(10.0)], sp=[np.ones(10)]) for _ in TIMES]
    ts = TimeSeries(times=list(TIMES), vals=specs, calculated_vrad_list=list(VALS))
    det = PeriodicityDetector(time_series=ts, freq_range=(0.05, 1.0),
                              periodogram_grid_resolution=20)
    det.calc_PDC(calc_unbiased_flag=True)
    assert ts.vals is specs and all(v is s for v, s in zip(ts.vals, specs))
    np.testing.assert_allclose(det.results_power["PDC_unbiased"], GOLDEN_POWER,
                               rtol=1e-8, atol=1e-10)


def test_plot_with_values_leaves_period_truth_unset():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = TimeSeries(times=list(TIMES), vals=list(VALS))
    det = PeriodicityDetector(time_series=ts, freq_range=(0.05, 1.0),
                              periodogram_grid_resolution=20)
    det.calc_GLS()
    det.plot_periodograms(plot_vals=True)
    plt.close("all")
    assert det.period_truth is None
