"""Correctness tests for the reimplemented CCF / RV extraction.

The legacy CrossCorrelateSpec could not recover non-negative radial velocities
(see KNOWN_TRAPS.md): it truncated the template to a shifted sub-grid and used
a constant (not per-lag) normalization, so only RV <~ 0 was recoverable. The
rewrite over spectral_ops.ccf_vs_velocity resamples template and signal onto a
common, equal-length log-lambda grid and scans symmetric velocity lags, so the
whole axis is valid. The golden master is physical: recovered RV must equal the
injected RV, for both signs.
"""
import numpy as np
import pytest

from sparta import spectral_ops as so
from sparta.UNICOR.Spectrum import Spectrum
from sparta.UNICOR.Template import Template
from sparta.UNICOR.CCF1d import CCF1d

C = 299792.458
WAVE = np.linspace(5480.0, 5520.0, 4000)


def _line(center):
    return 1.0 - 0.6 * np.exp(-0.5 * ((WAVE - center) / 0.12) ** 2)


@pytest.fixture(scope="module")
def template():
    return Template(template=Spectrum(wv=[WAVE.copy()], sp=[_line(5500.0)], name="t"))


@pytest.mark.parametrize("inj", [-10.0, -5.0, 0.0, 5.0, 10.0])
def test_rv_recovers_injected(template, inj):
    obs = Spectrum(wv=[WAVE.copy()], sp=[_line(5500.0 * (1 + inj / C))], name="o")
    ccf = CCF1d().CrossCorrelateSpec(spec_in=obs, template_in=template,
                                     dv=0.5, VelBound=50).CombineCCFs()
    rv = ccf.CorrCombined["RV"]
    rv = float(rv.value if hasattr(rv, "value") else rv)
    assert rv == pytest.approx(inj, abs=0.05)
    assert float(ccf.CorrCombined["peakCorr"]) > 0.99


@pytest.mark.parametrize("inj", [-8.0, 0.0, 8.0])
def test_ccf_vs_velocity_recovers_shift(inj):
    dv, n = 0.5, 2000
    u = np.arange(n) * dv

    def line(cv):
        return 1.0 - 0.6 * np.exp(-0.5 * ((u - n // 2 * dv - cv) / 3.0) ** 2)

    v, ccf = so.ccf_vs_velocity(line(0.0), line(inj), dv, vmax=40)
    assert v[np.argmax(ccf)] == pytest.approx(inj, abs=dv)
    assert ccf.max() > 0.999   # ~1; shifted signals lose a little edge overlap


def test_ccf_vs_velocity_rejects_mismatched_grids():
    with pytest.raises(ValueError):
        so.ccf_vs_velocity(np.ones(10), np.ones(12), 0.5, 5)
