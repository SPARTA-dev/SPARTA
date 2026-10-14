"""Regression tests for Spectrum preprocessing after the removed-API fixes.

`np.setdiff` (never a real function), `np.Inf`, and `scipy.signal.tukey` were
replaced by `np.setdiff1d`, `np.inf`, and `scipy.signal.windows.tukey`. The
select_orders(remove=True) path used to raise NameError outright.
"""
import numpy as np
import pytest
import astropy.units as u

from sparta.spectrum import Spectrum


def _three_order_spec():
    w = [np.linspace(5000, 5010, 200),
         np.linspace(5010, 5020, 200),
         np.linspace(5020, 5030, 200)]
    s = [1 - 0.3 * np.exp(-0.5 * ((wi - wi.mean()) / 0.5) ** 2) for wi in w]
    return Spectrum(wv=[x.copy() for x in w], sp=[x.copy() for x in s], name="t")


def test_select_orders_remove():
    sp = _three_order_spec()
    sp.n_ord = 3
    sp.select_orders(orders=[1], remove=True)   # np.setdiff1d
    assert len(sp.wv) == 2 and len(sp.sp) == 2  # orders 0 and 2 kept


def test_cosine_bell_and_cosmics_run():
    sp = _three_order_spec()
    sp.apply_cosine_bell(alpha=0.3)              # signal.windows.tukey
    assert len(sp.sp) == 3
    sp.remove_cosmics(sigma_upper=3)            # sigma_clip(sigma_lower=np.inf)
    assert len(sp.sp) == 3


def test_interpolate_log_grid_is_velocity_even():
    # interpolate now resamples via specutils (cubic spline). The log
    # method must still produce a velocity-even (uniform-in-log-lambda) grid and
    # preserve the line, since the CCF relies on it.
    w = np.linspace(5000.0, 5020.0, 500)
    s = 1 - 0.5 * np.exp(-0.5 * ((w - 5010.0) / 0.3) ** 2)
    sp = Spectrum(wv=[w.copy()], sp=[s.copy()], name="t")
    sp.interpolate(delta=0.5, method="log")
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


def _log_grid_spec():
    w = np.linspace(5000.0, 5020.0, 2000)
    rng = np.random.default_rng(3)
    s = 1 - 0.5 * np.exp(-0.5 * ((w - 5010.0) / 0.3) ** 2) + rng.normal(0, 0.01, w.size)
    sp = Spectrum(wv=[w], sp=[s], name="t")
    sp.interpolate(delta=0.5, method="log")
    return sp


def test_remove_cosmics_keeps_the_log_grid_and_ccf_still_runs():
    from sparta.template import Template
    from sparta.ccf1d import CCF1d
    sp = _log_grid_spec()
    tmpl = Template(Spectrum(wv=[sp.wv[0].copy()], sp=[sp.sp[0].copy()], name="tm"))
    wave = sp.wv[0].copy()
    sp.sp[0][[300, 1200, 2000]] += 1.0                  # three cosmics
    sp.remove_cosmics(sigma_upper=5)
    np.testing.assert_array_equal(sp.wv[0], wave)       # grid untouched
    assert sp.Info["GridType"] == "log"
    ccf = CCF1d().cross_correlate(spectrum=sp, template=tmpl, vel_bound=20).combine()
    assert float(ccf.CorrCombined["RV"]) == pytest.approx(0.0, abs=0.05)


def test_remove_cosmics_replaces_a_cosmic_by_its_neighbours():
    sp = _log_grid_spec()
    clean = sp.sp[0].copy()
    sp.sp[0][1200] += 1.0
    sp.remove_cosmics(sigma_upper=5)
    w = sp.wv[0]
    expected = np.interp(w[1200], w[[1199, 1201]], clean[[1199, 1201]])
    assert sp.sp[0][1200] == pytest.approx(expected, abs=1e-12)
    others = np.arange(clean.size) != 1200
    np.testing.assert_array_equal(sp.sp[0][others], clean[others])


def test_remove_cosmics_keeps_zero_valued_and_below_median_pixels():
    # The old filter tested the truthiness of each masked value, which also
    # dropped every pixel whose flux was exactly 0.0; negative outliers (deep
    # lines, below the median) are never clipped -- the clip is one-sided.
    w = np.linspace(5000.0, 5010.0, 500)
    s = np.ones_like(w) + np.random.default_rng(0).normal(0, 0.01, w.size)
    s[100:105] = 0.0
    s[250] = -3.0
    sp = Spectrum(wv=[w.copy()], sp=[s.copy()], name="t")
    sp.remove_cosmics(sigma_upper=3)
    assert sp.sp[0].size == w.size
    np.testing.assert_array_equal(sp.wv[0], w)
    below = s < np.median(s)
    np.testing.assert_array_equal(sp.sp[0][below], s[below])


def _native_log_spec(n_ord=1, dv_native=1.0):
    from unicor import dsp
    waves = [dsp.log_lambda_grid(5000.0 + 100.0 * k, 5050.0 + 100.0 * k, dv_native)[0]
             for k in range(n_ord)]
    fluxes = [1 - 0.5 * np.exp(-0.5 * ((w - w.mean()) / 0.3) ** 2) for w in waves]
    return Spectrum(wv=waves, sp=fluxes, name="t")


def test_orders_round_trip_keeps_the_native_sampling():
    # A resampled spectrum sent through .orders and back used to claim its
    # current step as native: 10x too many independent bins.
    sp = _native_log_spec(n_ord=2).interpolate(delta=0.1, method="log")
    orders = sp.orders
    assert [o.meta["NativeStep"] for o in orders] == pytest.approx(sp.native_steps())
    back = Spectrum.from_spectrum1d(orders)
    assert back.native_steps() == pytest.approx(sp.native_steps())
    assert back.native_steps() == pytest.approx([1.0, 1.0], rel=1e-4)
    assert back.Info["GridType"] == "log"
    assert back.Info["GridDelta"] == sp.Info["GridDelta"]
    # Orders that disagree on their grid are taken as raw (to be resampled).
    orders[1].meta["GridType"] = "linear"
    assert Spectrum.from_spectrum1d(orders).Info["GridType"] == "raw"


def test_from_spectrum1d_without_sparta_meta_is_native():
    sp = _native_log_spec()
    back = Spectrum.from_spectrum1d(sp.orders)
    assert "NativeStep" not in back.Info
    assert back.native_steps() == pytest.approx([1.0], rel=1e-4)


def test_a_coarser_resampling_raises_the_native_step():
    # An intermediate coarser resampling sets the noise scale: the CCF must
    # not count the original (finer) pixels afterwards.
    sp = _native_log_spec()
    lam = np.median(sp.wv[0])
    sp.interpolate(delta=3.0)                  # linear, 3 x median step
    coarse = 3.0 * np.median(np.diff(_native_log_spec().wv[0])) / lam * 299792.458
    assert sp.native_steps() == pytest.approx([coarse], rel=1e-3)
    sp.interpolate(delta=0.25, method="log")   # finer: unchanged
    assert sp.native_steps() == pytest.approx([coarse], rel=1e-3)
    sp.interpolate(delta=5.0, method="log")    # coarser again
    assert sp.native_steps() == pytest.approx([5.0], rel=1e-3)


@pytest.mark.parametrize("steps, match", [([1.0], "2 orders"),
                                          ([1.0, 0.0], "finite and positive"),
                                          ([1.0, np.nan], "finite and positive")])
def test_native_steps_are_validated_against_the_orders(steps, match):
    sp = _native_log_spec(n_ord=2)
    sp.Info["NativeStep"] = steps
    with pytest.raises(ValueError, match=match):
        sp.native_steps()


def test_select_orders_keeps_native_steps_aligned():
    sp = _native_log_spec(n_ord=3)
    sp.Info["NativeStep"] = [1.0, 2.0, 3.0]
    sp.select_orders(orders=[1], remove=True)
    assert sp.native_steps() == [1.0, 3.0]


@pytest.mark.parametrize("orders", [[3], [-1], [0, 5]])
def test_select_orders_rejects_out_of_range_indices(orders):
    sp = _three_order_spec()
    with pytest.raises(ValueError, match="order indices"):
        sp.select_orders(orders=orders)
    assert len(sp.wv) == 3                      # untouched


def test_mismatched_wv_and_sp_raise_valueerror():
    with pytest.raises(ValueError, match="orders"):
        Spectrum(wv=[np.arange(3.0)], sp=[])


def test_default_spectra_do_not_share_their_order_lists():
    a, b = Spectrum(), Spectrum()
    a.wv.append(np.arange(3.0))
    assert b.wv == [] and b.n_ord == 0


def test_trim_removes_noisy_unexposed_margins_before_filtering():
    # NRES-like order: unexposed detector at both ends reads ~0 plus noise,
    # not exact zeros. Those margins must go before the continuum is removed,
    # or the filter turns the 0 -> continuum jump into a step at the edge.
    rng = np.random.default_rng(3)
    w = np.linspace(5120.0, 5230.0, 4000)
    flux = 400.0 + rng.normal(0.0, 5.0, w.size)
    pad = (w < 5131.0) | (w > 5224.0)
    flux[pad] = rng.normal(0.0, 0.2, pad.sum())
    spec = Spectrum(wv=[w.copy()], sp=[flux.copy()])
    spec.trim(clean_margins=True)
    kept = spec.wv[0]
    assert kept[0] >= 5130.5 and kept[-1] <= 5224.5
    assert kept[0] < 5131.5 and kept[-1] > 5223.5      # nothing exposed was lost
    assert np.all(np.asarray(spec.sp[0]) > 100.0)


def test_trim_keeps_the_last_exposed_pixel_and_handles_nans_first():
    # Margins are found after NaNs are removed, and the bounds are inclusive.
    w = np.arange(12, dtype=float) + 5000.0
    flux = np.array([np.nan, np.nan, 0, 0, 1, 2, 3, 4, 5, 0, 0, 0], dtype=float)
    spec = Spectrum(wv=[w], sp=[flux])
    spec.trim(clean_margins=True, margin_window=1, margin_fraction=0.1)
    np.testing.assert_array_equal(spec.sp[0], [1, 2, 3, 4, 5])


def test_interpolate_refuses_an_empty_order():
    spec = Spectrum(wv=[np.array([5000.0])], sp=[np.array([1.0])])
    with pytest.raises(ValueError, match="cannot be resampled"):
        spec.interpolate()
