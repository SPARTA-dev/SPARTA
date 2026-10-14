"""PDC p-values as array atoms: :func:`sparta.pdc.pdc_pvalue` and its inverse.

Three layers, as in ``tests/test_atoms_unicor.py``:

* the atoms: the chi-square approximation of Shen et al. (2019),
  ``p = chi2_1.sf(n * power + 1)``, and its inverse, element by element as
  the comprehension it replaces (bit for bit);
* ``PeriodicityDetector.calc_pdc_pval`` calls exactly one of them -- spied
  through ``sparta.pdc``, the module ``sparta.periodogram`` reads them from;
* the atoms rebuild ``calc_pdc_pval`` exactly (``np.array_equal``) on the
  byte-identity baseline's series (``tests/_atoms_baseline.py``).

Times in days, frequencies in 1/day, power dimensionless.
"""
import numpy as np
import pytest
from scipy.stats import chi2

from sparta import pdc
from sparta.pdc import pdc_power_level, pdc_pvalue
from sparta.periodogram import PeriodicityDetector
from sparta.timeseries import TimeSeries

LEVELS = [0.1, 0.01, 1e-3, 1e-4]


def _baseline_detector():
    """The scalar series of ``_atoms_baseline._record_periodograms``, PDC computed."""
    rng = np.random.default_rng(5)
    times = np.sort(rng.uniform(0.0, 100.0, 30))
    vals = np.sin(2 * np.pi * times / 5.0) + 0.3 * rng.normal(size=times.size)
    det = PeriodicityDetector(time_series=TimeSeries(times=list(times), vals=list(vals)),
                              freq_range=(0.02, 1.0), periodogram_grid_resolution=40)
    det.calc_pdc()
    return det


# =============================================================================
# The atoms
# =============================================================================
def test_zero_power_is_the_chi2_tail_at_one():
    assert pdc_pvalue([0.0], 30)[0] == chi2.sf(1.0, 1)
    assert pdc_pvalue(0.0, 30) == pytest.approx(0.3173105, abs=1e-7)


def test_the_power_level_inverts_the_pvalue():
    power = np.linspace(0.0, 0.5, 11)
    np.testing.assert_allclose(pdc_power_level(pdc_pvalue(power, 17), 17), power,
                               rtol=0, atol=1e-12)


def test_the_pvalue_falls_with_power_and_with_n():
    p = pdc_pvalue([0.01, 0.05, 0.1], 30)
    assert np.all(np.diff(p) < 0)
    assert pdc_pvalue(0.05, 60) < pdc_pvalue(0.05, 30)


@pytest.mark.parametrize("n", [4, 30, 1000])
def test_the_atoms_equal_the_scalar_comprehension_bit_for_bit(n):
    rng = np.random.default_rng(1)
    power = np.r_[rng.normal(0.0, 0.05, 2000), -1.0 / n, 0.0]
    p = np.r_[rng.uniform(0.0, 1.0, 2000), LEVELS]
    assert pdc_pvalue(power, n).tobytes() == np.array(
        [chi2.sf(x * n + 1, 1) for x in power]).tobytes()
    assert pdc_power_level(p, n).tobytes() == np.array(
        [(chi2.isf(x, 1) - 1) / n for x in p]).tobytes()


@pytest.mark.parametrize("atom", [pdc_pvalue, pdc_power_level])
@pytest.mark.parametrize("n", [0, -3])
def test_the_atoms_need_a_positive_number_of_epochs(atom, n):
    with pytest.raises(ValueError, match="number of epochs"):
        atom([0.1], n)


# =============================================================================
# calc_pdc_pval calls one atom
# =============================================================================
def _spy(monkeypatch, name):
    calls = []
    real = getattr(pdc, name)

    def wrapper(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(pdc, name, wrapper)
    return calls


@pytest.mark.parametrize("inverse_sf, used", [(False, "pdc_pvalue"),
                                              (True, "pdc_power_level")])
def test_calc_pdc_pval_calls_one_atom(monkeypatch, inverse_sf, used):
    det = _baseline_detector()
    calls = {name: _spy(monkeypatch, name) for name in ("pdc_pvalue", "pdc_power_level")}
    det.calc_pdc_pval(LEVELS, inverse_sf=inverse_sf)
    assert {k: len(v) for k, v in calls.items()} == {
        name: int(name == used) for name in calls}
    (args, _), = calls[used]
    assert args[1] == 30


# =============================================================================
# The atoms rebuild calc_pdc_pval exactly
# =============================================================================
def test_the_atoms_rebuild_calc_pdc_pval_exactly():
    det = _baseline_detector()
    power = det.results_power["PDC_unbiased"]
    n = len(det.time_series.times)
    assert np.array_equal(det.calc_pdc_pval(power), pdc_pvalue(power, n))
    assert np.array_equal(det.calc_pdc_pval(LEVELS, inverse_sf=True),
                          pdc_power_level(LEVELS, n))


def test_calc_pdc_pval_still_takes_any_iterable():
    det = _baseline_detector()
    assert np.array_equal(det.calc_pdc_pval(iter(LEVELS), inverse_sf=True),
                          det.calc_pdc_pval(LEVELS, inverse_sf=True))
    assert det.calc_pdc_pval([]).shape == (0,)
