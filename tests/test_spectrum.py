"""Regression tests for Spectrum preprocessing after the removed-API fixes.

`np.setdiff` (never a real function), `np.Inf`, and `scipy.signal.tukey` were
replaced by `np.setdiff1d`, `np.inf`, and `scipy.signal.windows.tukey`. The
SelectOrders(remove=True) path used to raise NameError outright.
"""
import numpy as np
import pytest
import astropy.units as u

from sparta.UNICOR.Spectrum import Spectrum


def _three_order_spec():
    w = [np.linspace(5000, 5010, 200),
         np.linspace(5010, 5020, 200),
         np.linspace(5020, 5030, 200)]
    s = [1 - 0.3 * np.exp(-0.5 * ((wi - wi.mean()) / 0.5) ** 2) for wi in w]
    return Spectrum(wv=[x.copy() for x in w], sp=[x.copy() for x in s], name="t")


def test_select_orders_remove():
    sp = _three_order_spec()
    sp.n_ord = 3
    sp.SelectOrders(orders=[1], remove=True)   # np.setdiff1d
    assert len(sp.wv) == 2 and len(sp.sp) == 2  # orders 0 and 2 kept


def test_cosine_bell_and_cosmics_run():
    sp = _three_order_spec()
    sp.ApplyCosineBell(alpha=0.3)              # signal.windows.tukey
    assert len(sp.sp) == 3
    sp.RemoveCosmics(sigma_upper=3)            # sigma_clip(sigma_lower=np.inf)
    assert len(sp.sp) == 3


def test_interpolate_log_grid_is_velocity_even():
    # InterpolateSpectrum now resamples via specutils (cubic spline). The log
    # method must still produce a velocity-even (uniform-in-log-lambda) grid and
    # preserve the line, since the CCF relies on it.
    w = np.linspace(5000.0, 5020.0, 500)
    s = 1 - 0.5 * np.exp(-0.5 * ((w - 5010.0) / 0.3) ** 2)
    sp = Spectrum(wv=[w.copy()], sp=[s.copy()], name="t")
    sp.InterpolateSpectrum(delta=0.5, InterpMethod="log")
    dlnw = np.diff(np.log(sp.wv[0]))
    np.testing.assert_allclose(dlnw, dlnw[0], rtol=1e-6)      # uniform in log-lambda
    assert np.isfinite(sp.sp[0]).all()                        # no edge NaNs
    assert sp.sp[0].min() == pytest.approx(0.5, abs=0.02)     # line depth preserved


def test_orders_are_specutils_spectrum1d():
    sp = _three_order_spec()
    orders = sp.orders
    assert len(orders) == 3                          # one Spectrum1D per order
    o0 = orders[0]
    assert o0.spectral_axis.unit == u.Angstrom
    assert o0.flux.unit == u.dimensionless_unscaled
    np.testing.assert_allclose(o0.spectral_axis.to_value(u.Angstrom), sp.wv[0])
    np.testing.assert_allclose(o0.flux.value, sp.sp[0])


def test_from_spectrum1d_roundtrip():
    sp = _three_order_spec()
    sp.name = "roundtrip"
    back = Spectrum.from_spectrum1d(sp.orders)       # SpectrumList -> Spectrum
    assert back.n_ord == 3 and back.name == "roundtrip"
    for orig_w, orig_s, w, s in zip(sp.wv, sp.sp, back.wv, back.sp):
        np.testing.assert_allclose(w, orig_w)
        np.testing.assert_allclose(s, orig_s)
    # a single Spectrum1D also builds a one-order Spectrum
    single = Spectrum.from_spectrum1d(sp.orders[0])
    assert single.n_ord == 1
