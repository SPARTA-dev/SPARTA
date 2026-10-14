"""Tests for Observations.clean_time_series and the ReadSpec KEK stub.

The Observations object is built directly from a synthetic TimeSeries (no FITS
reading); per-epoch products such as RVs and CCFs are attached by hand so the
epoch filter can be checked against every array the object holds.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from sparta.Auxil.ReadSpec import ReadSpec
from sparta.Auxil.TimeSeries import TimeSeries
from sparta.Observations import Observations

TIMES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
VRADS = [-10.0, 0.0, np.nan, 5.0, 12.0, -3.0]


def _observations(vrads=()):
    specs = [SimpleNamespace(wv=[np.linspace(5000.0, 5010.0, 50)],
                             sp=[np.ones(50)], epoch=i)
             for i in range(len(TIMES))]
    ts = TimeSeries(times=list(TIMES), vals=specs,
                    calculated_vrad_list=list(vrads),
                    errors=[0.1 * (i + 1) for i in range(len(TIMES))])
    obs = Observations(time_series=ts)
    obs.bcv = [float(i) for i in range(len(TIMES))]
    obs.snr = [10.0 * i for i in range(len(TIMES))]
    obs.vels = np.arange(len(TIMES), dtype=float)
    obs.evels = 0.5 * np.arange(len(TIMES), dtype=float)
    obs.ccf_list = [f"ccf{i}" for i in range(len(TIMES))]
    return obs


def _epochs(obs):
    return [s.epoch for s in obs.time_series.vals]


def _assert_consistent(obs, kept):
    n = len(kept)
    assert obs.sample_size == obs.time_series.size == n
    assert obs.time_series.times == [TIMES[i] for i in kept]
    assert obs.time_list == obs.time_series.times
    assert _epochs(obs) == kept
    assert [s.epoch for s in obs.spec_list] == kept
    assert obs.bcv == [float(i) for i in kept]
    assert obs.snr == [10.0 * i for i in kept]
    np.testing.assert_array_equal(obs.vels, np.asarray(kept, float))
    np.testing.assert_array_equal(obs.evels, 0.5 * np.asarray(kept, float))
    assert obs.ccf_list == [f"ccf{i}" for i in kept]
    np.testing.assert_allclose(obs.time_series.errors,
                               [0.1 * (i + 1) for i in kept])


def test_clean_defaults_without_rvs_keeps_every_epoch():
    obs = _observations()
    obs.clean_time_series()
    _assert_consistent(obs, list(range(len(TIMES))))
    assert len(obs.time_series.calculated_vrad_list) == 0


def test_clean_drops_nan_rvs_from_every_array():
    obs = _observations(VRADS)
    obs.clean_time_series()
    kept = [0, 1, 3, 4, 5]
    _assert_consistent(obs, kept)
    assert list(obs.time_series.calculated_vrad_list) == [VRADS[i] for i in kept]


def test_clean_honors_zero_thresholds():
    obs = _observations(VRADS)
    obs.clean_time_series(max_vel=0)
    _assert_consistent(obs, [0, 1, 5])

    obs = _observations(VRADS)
    obs.clean_time_series(min_vel=0)
    _assert_consistent(obs, [1, 3, 4])

    obs = _observations()
    obs.clean_time_series(max_time=0)
    _assert_consistent(obs, [0])


def test_clean_combines_filters_and_sample_rate():
    obs = _observations(VRADS)
    obs.clean_time_series(sample_rate=2, max_vel=10)   # keeps 0, 2, 4 -> NaN, >10 out
    _assert_consistent(obs, [0])

    obs = _observations()
    obs.clean_time_series(sample_rate=-3)              # drops 0 and 3
    _assert_consistent(obs, [1, 2, 4, 5])


def test_clean_velocity_cut_without_rvs_raises():
    with pytest.raises(ValueError, match="calculated_vrad_list"):
        _observations().clean_time_series(max_vel=10)


def test_clean_rejects_zero_sample_rate():
    with pytest.raises(ValueError, match="sample_rate"):
        _observations().clean_time_series(sample_rate=0)


def test_kek_loader_is_not_implemented():
    with pytest.raises(NotImplementedError, match="KEK"):
        ReadSpec("KEK")._load_kek(None, 4900, 5150)


def test_clean_time_series_ignores_unaligned_vrad_list_and_rv_pairs():
    # vrad_list only gets entries for headers carrying VRAD, and a [min, max]
    # pair of RV rows is not one RV per epoch -- neither may block a time cut.
    obs = _observations()
    obs.vrad_list = [1.0, 2.0]
    obs.time_series.calculated_vrad_list = [np.zeros(len(TIMES)), np.ones(len(TIMES))]
    with pytest.warns(UserWarning, match="NaN filter"):
        obs.clean_time_series(max_time=2.5)
    _assert_consistent(obs, [0, 1, 2])
    assert obs.vrad_list == [1.0, 2.0]
