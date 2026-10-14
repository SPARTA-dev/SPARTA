"""Wavelength-axis helpers shared by the loaders (sparta.instruments._common)."""
import numpy as np
import pytest

from sparta.instruments._common import _ascending, _wavelength_window


def test_wavelength_window_matches_legacy_where_and_clips_partial_overlap():
    w = np.linspace(5000.0, 5100.0, 1001)
    window = _wavelength_window(w, 5010.0, 5020.0)
    assert window.start == np.where(w > 5010.0)[0][0]
    assert window.stop == np.where(w > 5020.0)[0][0]
    clipped = _wavelength_window(w, 5090.0, 6000.0)
    assert w[clipped][-1] == w[-1]


@pytest.mark.parametrize("lo, hi", [(6000.0, 6100.0), (4000.0, 4999.0), (5050.0, 5050.0)])
def test_wavelength_window_raises_valueerror_off_range(lo, hi):
    with pytest.raises(ValueError):
        _wavelength_window(np.linspace(5000.0, 5100.0, 11), lo, hi)


def test_ascending_flips_decreasing_and_rejects_non_monotonic():
    w, s = _ascending(np.array([3.0, 2.0, 1.0]), np.array([30.0, 20.0, 10.0]))
    np.testing.assert_array_equal(w, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(s, [10.0, 20.0, 30.0])
    with pytest.raises(ValueError):
        _ascending(np.array([1.0, 3.0, 2.0]))
