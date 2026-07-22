"""Golden-master + physical tests for velocity-space broadening (spectral_ops).

The PYASL_* reference values were captured 2026-07-22 from PyAstronomy
(rotBroad, instrBroadGaussFast) on the fixed grid below, before PyAstronomy
was removed. spectral_ops reproduces them to <1e-4 in the line interior and
conserves equivalent width exactly, so the removal is behavior-preserving.
"""
import numpy as np
import pytest
from scipy.integrate import trapezoid

from sparta import spectral_ops as so

C = 299792.458
WL = np.linspace(5495.0, 5505.0, 4001)
LAM_C = WL.mean()
DV = C * (WL[1] - WL[0]) / LAM_C
U = (WL - LAM_C) / LAM_C * C
FLUX = 1.0 - 0.6 * np.exp(-0.5 * (U / 4.0) ** 2)   # absorption line, sigma=4 km/s
IC = 2000
IDX = np.arange(IC - 6, IC + 7)

# pyasl golden at IDX: rotBroad(epsilon=0.6, vsini=25 km/s); instrBroadGaussFast(R=20000).
PYASL_ROT = np.array([
    0.8366185907, 0.8365761103, 0.8365413683, 0.8365143635, 0.8364950951,
    0.8364835624, 0.8364797649, 0.8364837025, 0.8364953753, 0.8365147836,
    0.8365419281, 0.8365768096, 0.8366194292,
])
PYASL_GAU = np.array([
    0.6826464927, 0.6820725153, 0.6816021253, 0.6812357853, 0.6809738559,
    0.6808165949, 0.6807641574, 0.6808165949, 0.6809738559, 0.6812357853,
    0.6816021253, 0.6820725153, 0.6826464927,
])


def test_rotational_matches_pyasl_golden():
    rot = so.broaden_rotational(FLUX, DV, 25.0, 0.6)
    np.testing.assert_allclose(rot[IDX], PYASL_ROT, atol=1e-4)


def test_gaussian_matches_pyasl_golden():
    gau = so.broaden_gaussian(FLUX, DV, C / 20000.0)
    np.testing.assert_allclose(gau[IDX], PYASL_GAU, atol=1e-4)


def test_broadening_conserves_equivalent_width():
    ew_in = trapezoid(1 - FLUX, U)
    rot = so.broaden_rotational(FLUX, DV, 25.0, 0.6)
    gau = so.broaden_gaussian(FLUX, DV, C / 20000.0)
    np.testing.assert_allclose(trapezoid(1 - rot, U), ew_in, rtol=1e-6)
    np.testing.assert_allclose(trapezoid(1 - gau, U), ew_in, rtol=1e-6)


def test_rotational_kernel_properties():
    k = so.rotational_kernel(DV, 25.0, 0.6)
    assert k.sum() == pytest.approx(1.0)          # unit sum
    np.testing.assert_allclose(k, k[::-1])         # symmetric
    assert len(k) == 2 * int(25.0 / DV) + 1        # support [-vsini, vsini]
    assert (k >= 0).all()                          # non-negative profile
