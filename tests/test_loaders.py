"""Synthetic-FITS tests for the per-instrument ReadSpec loaders.

Each test writes a minimal file following the instrument's documented header
conventions (data-model URLs are cited in the loader docstrings) to tmp_path
and checks: the wavelength solution, float64 output, a float observation time,
trimming to [min_wv, max_wv], and the recorded frame attributes. HARPS and NRES
are pinned on real data in test_readspec.py.
"""
import warnings

import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

from sparta.Auxil.ReadSpec import (ReadSpec, _wavelength_window, _ascending,
                                   _iraf_multispec_wavelengths)
from sparta.Observations import Observations


def _write(tmp_path, hdus, name="spec.fits"):
    path = tmp_path / name
    fits.HDUList(hdus).writeto(path)
    return str(path)


def _load(survey, path, **kw):
    rs = ReadSpec(survey)
    rs.load_spectrum_from_fits(path=path, **kw)
    return rs


def _check_single_order(rs, min_wv, max_wv):
    assert rs.n_orders == 1
    assert rs.w.shape == rs.s.shape and rs.w.shape[0] == 1 and rs.w.size > 0
    assert rs.w.dtype == np.float64 and rs.s.dtype == np.float64
    assert np.all(np.diff(rs.w[0]) > 0)
    assert min_wv < rs.w[0, 0] and rs.w[0, -1] <= max_wv
    assert isinstance(rs.DATE_OBS, float)


def _table(columns, n):
    """One-row ESO-SDP-style binary table of array columns."""
    return fits.BinTableHDU.from_columns(
        [fits.Column(name=name, format=f"{n}D", array=np.asarray(values, float)[None, :])
         for name, values in columns.items()], name="SPECTRUM")


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# LAMOST (log-linear, vacuum, heliocentric)
# --------------------------------------------------------------------------
def _lamost(tmp_path, helio=True, vacuum=True, name="lamost.fits"):
    n = 3909
    data = np.zeros((5, n), dtype=np.float32)
    data[0] = np.arange(n)
    hdr = fits.Header()
    hdr.update({"COEFF0": 3.5682, "COEFF1": 1e-4, "CRVAL1": 3.5682, "CD1_1": 1e-4,
                "CRPIX1": 1, "HELIO": helio, "HELIO_RV": -22.47678, "VACUUM": vacuum,
                "DATE-OBS": "2015-03-11T20:30:51.01"})
    return _write(tmp_path, [fits.PrimaryHDU(data, hdr)], name)


def test_lamost_log_linear_solution(tmp_path):
    rs = _load("LAMOST", _lamost(tmp_path), min_wv=5000.0, max_wv=5100.0)
    _check_single_order(rs, 5000.0, 5100.0)
    i = rs.s[0].astype(int)       # flux row was set to the pixel index
    np.testing.assert_allclose(rs.w[0], 10 ** (3.5682 + 1e-4 * i), rtol=1e-13)
    np.testing.assert_allclose(rs.w[0, 1] / rs.w[0, 0], 10 ** 1e-4, rtol=1e-13)
    assert rs.DATE_OBS == pytest.approx(Time("2015-03-11T20:30:51.01").jd, abs=1e-9)
    assert (rs.time_system, rs.wavelength_frame, rs.specsys, rs.bcv) == \
        ("JD-UTC", "vacuum", "HELIOCEN", 0.0)


def test_lamost_rejects_uncorrected_and_off_window(tmp_path):
    with pytest.raises(NotImplementedError):
        _load("LAMOST", _lamost(tmp_path, helio=False), min_wv=5000.0, max_wv=5100.0)
    with pytest.raises(ValueError):
        _load("LAMOST", _lamost(tmp_path, name="b.fits"), min_wv=10000.0, max_wv=11000.0)


# --------------------------------------------------------------------------
# ELODIE (linear WCS, topocentric)
# --------------------------------------------------------------------------
def _elodie(tmp_path, crpix=1.0, name="elodie.fits"):
    n = 56001
    hdr = fits.Header()
    hdr.update({"CRVAL1": 4000.0, "CDELT1": 0.05, "CRPIX1": crpix, "SN": 55.0,
                "DATE-OBS": "1998-07-02T23:01:02"})
    return _write(tmp_path, [fits.PrimaryHDU(np.arange(n, dtype=np.float32), hdr)], name)


def test_elodie_linear_wcs_without_float32_drift(tmp_path):
    with pytest.warns(UserWarning, match="topocentric"):
        rs = _load("ELODIE", _elodie(tmp_path), min_wv=6700.0, max_wv=6800.0)
    _check_single_order(rs, 6700.0, 6800.0)
    pix = rs.s[0]                 # flux = 0-based pixel index
    np.testing.assert_allclose(rs.w[0], 4000.0 + 0.05 * pix, rtol=0, atol=1e-9)
    assert rs.DATE_OBS == pytest.approx(Time("1998-07-02T23:01:02").jd, abs=1e-9)
    assert (rs.time_system, rs.specsys, rs.bcv) == ("JD-UTC", "TOPOCENT", 0.0)


def test_elodie_honours_crpix(tmp_path):
    with pytest.warns(UserWarning):
        rs = _load("ELODIE", _elodie(tmp_path, crpix=11.0), min_wv=5000.0, max_wv=5010.0)
    np.testing.assert_allclose(rs.w[0], 4000.0 + 0.05 * (rs.s[0] - 10.0), atol=1e-9)


def test_observations_reads_elodie_times_as_floats(tmp_path):
    for k, date in enumerate(["1998-07-02T23:00:00", "1998-07-05T23:00:00"]):
        hdr = fits.Header()
        hdr.update({"CRVAL1": 4000.0, "CDELT1": 0.05, "SN": 50.0, "DATE-OBS": date})
        _write(tmp_path, [fits.PrimaryHDU(np.ones(56001, dtype=np.float32), hdr)], f"e{k}.fits")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        obs = Observations(survey="ELODIE", target_visits_lib=str(tmp_path),
                           min_wv=5150.0, max_wv=5300.0)
    times = sorted(obs.time_list)
    assert all(isinstance(t, float) for t in times)
    assert times[1] - times[0] == pytest.approx(3.0, abs=1e-9)
    assert obs.time_system == "JD-UTC"


def test_observations_refuses_mixed_wavelength_frames(tmp_path):
    _lamost(tmp_path, vacuum=True, name="a.fits")
    _lamost(tmp_path, vacuum=False, name="b.fits")
    with pytest.raises(ValueError, match="disagree"):
        Observations(survey="LAMOST", target_visits_lib=str(tmp_path),
                     min_wv=5000.0, max_wv=5100.0)


# --------------------------------------------------------------------------
# TRES (IRAF multispec, non-linear dispersion)
# --------------------------------------------------------------------------
def _wat2_cards(attributes):
    """Split a multispec WAT2 attribute string into 68-character cards, as IRAF does."""
    text = "wtype=multispec " + " ".join(f'spec{k} = "{a}"' for k, a in attributes.items())
    return {f"WAT2_{i // 68 + 1:03d}": text[i:i + 68] for i in range(0, len(text), 68)}


def _tres(tmp_path, attributes, n_pix=2304, name="tres.fits"):
    n_ord = len(attributes)
    data = np.tile(np.arange(n_pix, dtype=np.float32), (n_ord, 1)) + 1e4 * np.arange(n_ord)[:, None]
    hdr = fits.Header()
    hdr.update({"WCSDIM": 2, "CTYPE1": "MULTISPE", "CTYPE2": "MULTISPE",
                "CD1_1": 1.0, "CD2_2": 1.0, "LTM1_1": 1.0, "LTM2_2": 1.0,
                "WAT0_001": "system=multispec",
                "WAT1_001": "wtype=multispec label=Wavelength units=angstroms",
                "WS_BJD": 2458000.123456, "BCV": -12.5, "BUNIT": "adu"})
    hdr.update(_wat2_cards(attributes))
    return _write(tmp_path, [fits.PrimaryHDU(data, hdr)], name)


def _cheb_attr(ap, beam, centre, coeffs, n_pix=2304, wt=1.0, w0=0.0):
    c = " ".join(repr(float(x)) for x in coeffs)
    return (f"{ap} {beam} 2 {centre - 30.0!r} 0.026 {n_pix} 0. 10.0 20.0 "
            f"{wt!r} {w0!r} 1 {len(coeffs)} 1. {float(n_pix)!r} {c}")


TRES_ORDERS = {1: _cheb_attr(1, 90, 5050.0, [5050.0, 30.0, -0.25, 0.01]),
               2: _cheb_attr(2, 89, 5110.0, [5110.0, 30.5, -0.26, 0.011]),
               3: _cheb_attr(3, 88, 5170.0, [5170.0, 31.0, -0.27, 0.012])}


def test_tres_chebyshev_matches_specutils_iraf_reader(tmp_path):
    from specutils import SpectrumCollection
    path = _tres(tmp_path, TRES_ORDERS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oracle = SpectrumCollection.read(path, format="iraf").spectral_axis.to_value("Angstrom")
    rs = _load("TRES", path, min_wv=5100.0, max_wv=5120.0)
    _check_single_order(rs, 5100.0, 5120.0)
    order = int(rs.s[0, 0] // 1e4)                # flux encodes the order index
    assert order == 1                              # 5110 +- 30 A covers the window best
    pix = (rs.s[0] - 1e4 * order).astype(int)
    np.testing.assert_allclose(rs.w[0], oracle[order][pix], rtol=0, atol=1e-9)
    assert (rs.DATE_OBS, rs.bcv, rs.specsys) == (2458000.123456, -12.5, "TOPOCENT")


def test_tres_weight_zero_point_and_function_sum(tmp_path):
    n = 2304
    attr = (f"1 90 2 5000. 0.026 {n} 0. 10.0 20.0 "
            f"0.5 2.0 1 2 1. {float(n)!r} 5000. 30. "
            f"0.5 -2.0 2 2 1. {float(n)!r} 5000. 32.")
    rs = _load("TRES", _tres(tmp_path, {1: attr}), min_wv=4975.0, max_wv=5025.0)
    p = rs.s[0] + 1.0                              # 1-based physical pixel
    x = (p - (n + 1.0) / 2) / ((n - 1.0) / 2)
    expected = 0.5 * (2.0 + 5000.0 + 30.0 * x) + 0.5 * (-2.0 + 5000.0 + 32.0 * x)
    np.testing.assert_allclose(rs.w[0], expected, rtol=0, atol=1e-9)


def test_tres_spline_dispersion_is_not_implemented(tmp_path):
    attr = "1 90 2 5000. 0.026 2304 0. 10.0 20.0 1. 0. 3 2 1. 2304. 5000. 5010. 5020. 5030."
    with pytest.raises(NotImplementedError):
        _load("TRES", _tres(tmp_path, {1: attr}), min_wv=4975.0, max_wv=5025.0)



def test_tres_trimmed_image_counts_physical_pixels(tmp_path):
    # An image section (columns 101-2304: LTV1 = -100, LTM1_1 = 1) keeps the
    # physical-pixel dispersion functions; its logical pixel l is physical
    # l + 100. Order 3 has only 2200 valid physical pixels, i.e. 2100 logical
    # ones in the section -- not the first 2200.
    attrs = dict(TRES_ORDERS)
    attrs[3] = _cheb_attr(3, 88, 5170.0, [5170.0, 31.0, -0.27, 0.012], n_pix=2200)
    full = _tres(tmp_path, attrs, name="full.fits")
    with fits.open(full) as hdul:
        hdr = hdul[0].header.copy()
        data = hdul[0].data[:, 100:].copy()
    hdr["LTV1"] = -100.0
    trimmed = _write(tmp_path, [fits.PrimaryHDU(data, hdr)], "trimmed.fits")

    # specutils evaluates physical pixels 1..NAXIS1, ignoring LTV1 and nw, so
    # it can only check the untrimmed, full-length orders.
    from specutils import SpectrumCollection
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oracle = SpectrumCollection.read(_tres(tmp_path, TRES_ORDERS, name="oracle.fits"),
                                         format="iraf").spectral_axis.to_value("Angstrom")
    waves_full = _iraf_multispec_wavelengths(fits.getheader(full))
    waves_trim = _iraf_multispec_wavelengths(fits.getheader(trimmed))
    for line in (1, 2):
        np.testing.assert_allclose(waves_full[line], oracle[line - 1], rtol=0, atol=1e-9)
    for line in (1, 2, 3):
        np.testing.assert_array_equal(waves_trim[line], waves_full[line][100:])
    assert waves_trim[3].size == 2100

    rs = _load("TRES", trimmed, min_wv=5100.0, max_wv=5120.0)
    order = int(rs.s[0, 0] // 1e4)
    pix = (rs.s[0] - 1e4 * order).astype(int)       # flux encodes the physical pixel - 1
    np.testing.assert_allclose(rs.w[0], oracle[order][pix], rtol=0, atol=1e-9)


def test_multispec_log_linear_doppler_factor_divides_the_wavelength(tmp_path):
    # IRAF's multispec driver (sys/mwcs/wfmspec.x, wf_msp_fwd) evaluates
    # 10 ** (w1 + dw (p - 1)) and then divides the *wavelength* by (1 + z), as
    # does specutils' reader; the code used to divide the exponent instead.
    w1, dw, n = 3.70, 1.0e-5, 2304
    zs = {1: 0.0, 2: 3e-4}
    path = _tres(tmp_path, {k: f"{k} {90 - k} 1 {w1!r} {dw!r} {n} {z!r} 10.0 20.0"
                            for k, z in zs.items()})
    from specutils import SpectrumCollection
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oracle = SpectrumCollection.read(path, format="iraf").spectral_axis.to_value("Angstrom")
    waves = _iraf_multispec_wavelengths(fits.getheader(path))
    p = np.arange(1, n + 1)
    for k, z in zs.items():
        np.testing.assert_allclose(waves[k], 10 ** (w1 + dw * (p - 1)) / (1 + z), rtol=1e-13)
        np.testing.assert_allclose(waves[k], oracle[k - 1], rtol=1e-12)

# --------------------------------------------------------------------------
# ESO Phase-3 SDP: UVES and FEROS
# --------------------------------------------------------------------------
def _sdp(tmp_path, columns, header, tucd="em.wl;obs.atmos", name="sdp.fits"):
    n = len(next(iter(columns.values())))
    table = _table(columns, n)
    if tucd is not None:
        table.header["TUCD1"] = tucd
    return _write(tmp_path, [fits.PrimaryHDU(header=fits.Header(header)), table], name)


WAVE = np.linspace(4800.0, 5300.0, 5001)


def test_uves_topocentric_applies_qc_barycor(tmp_path):
    cols = {"WAVE": WAVE, "FLUX_REDUCED": np.ones_like(WAVE), "ERR_REDUCED": np.ones_like(WAVE),
            "BGFLUX_REDUCED": np.zeros_like(WAVE), "FLUX": 2 * np.ones_like(WAVE),
            "ERR": np.ones_like(WAVE)}
    hdr = {"SPECSYS": "TOPOCENT", "HIERARCH ESO QC VRAD BARYCOR": 12.3,
           "MJD-OBS": 58000.25, "SNR": 40.0}
    rs = _load("UVES", _sdp(tmp_path, cols, hdr), min_wv=4900.0, max_wv=5150.0)
    _check_single_order(rs, 4900.0, 5150.0)
    assert np.all(rs.s == 2.0)                     # flux-calibrated FLUX column
    assert (rs.bcv, rs.specsys, rs.wavelength_frame, rs.time_system, rs.DATE_OBS) == \
        (12.3, "TOPOCENT", "air", "MJD-UTC", 58000.25)


def test_uves_reduced_product_without_flux_column(tmp_path):
    cols = {"WAVE": WAVE, "FLUX_REDUCED": 3 * np.ones_like(WAVE), "ERR_REDUCED": np.ones_like(WAVE),
            "BGFLUX_REDUCED": np.zeros_like(WAVE)}
    hdr = {"SPECSYS": "BARYCENT", "MJD-OBS": 58000.25, "SNR": 40.0}
    rs = _load("UVES", _sdp(tmp_path, cols, hdr, tucd="em.wl"), min_wv=4900.0, max_wv=5150.0)
    assert np.all(rs.s == 3.0)
    assert (rs.bcv, rs.wavelength_frame) == (0.0, "vacuum")


@pytest.mark.parametrize("specsys, bcv", [("BARYCENT", 0.0), ("TOPOCENT", -7.5)])
def test_feros_applies_barycorr_only_when_topocentric(tmp_path, specsys, bcv):
    cols = {"WAVE": WAVE, "FLUX": np.ones_like(WAVE), "ERR": np.full_like(WAVE, np.nan)}
    hdr = {"SPECSYS": specsys, "HIERARCH ESO DRS BARYCORR": -7.5,
           "MJD-OBS": 58000.0, "MJD-END": 58000.02, "SNR": 30.0}
    rs = _load("FEROS", _sdp(tmp_path, cols, hdr), min_wv=4900.0, max_wv=5150.0)
    _check_single_order(rs, 4900.0, 5150.0)
    assert rs.bcv == bcv
    assert rs.DATE_OBS == pytest.approx(58000.01)
    assert (rs.time_system, rs.wavelength_frame) == ("MJD-UTC", "air")


def test_eso_sdp_without_specsys_is_refused(tmp_path):
    cols = {"WAVE": WAVE, "FLUX": np.ones_like(WAVE), "ERR": np.ones_like(WAVE)}
    hdr = {"HIERARCH ESO DRS BARYCORR": -7.5, "MJD-OBS": 58000.0, "MJD-END": 58000.02, "SNR": 30.0}
    with pytest.raises(ValueError, match="SPECSYS"):
        _load("FEROS", _sdp(tmp_path, cols, hdr), min_wv=4900.0, max_wv=5150.0)


# --------------------------------------------------------------------------
# APOGEE apVisit (decreasing vacuum wavelengths, observer frame)
# --------------------------------------------------------------------------
def _apogee(tmp_path, n=4096):
    wave = np.stack([np.linspace(16950.0, 16470.0, n),
                     np.linspace(16430.0, 15860.0, n),
                     np.linspace(15800.0, 15150.0, n)])       # chip a is the red one
    flux = (1.0 + 1e-4 * wave).astype(np.float32)              # flux is a function of wavelength
    mask = np.zeros((3, n), dtype=np.int16)
    mask[1, 100] = 1 << 12                                     # SIG_SKYLINE
    flux[1, 100] = -999.0
    hdr = fits.Header()
    hdr.update({"JD-MID": 2455813.69734, "BC": -8.92126133253, "VRAD": 4.80937})
    return _write(tmp_path, [fits.PrimaryHDU(header=hdr), fits.ImageHDU(flux),
                             fits.ImageHDU(np.ones_like(flux)), fits.ImageHDU(mask),
                             fits.ImageHDU(wave)], "apVisit.fits"), wave


def test_apogee_ascending_vacuum_with_bc(tmp_path):
    path, wave = _apogee(tmp_path)
    rs = _load("APOGEE", path, APOGEE_segment=1)
    assert rs.w.dtype == np.float64 and rs.s.dtype == np.float64
    assert np.all(np.diff(rs.w[0]) > 0)
    np.testing.assert_array_equal(rs.w[0], wave[1][::-1])
    np.testing.assert_allclose(rs.s[0], (1.0 + 1e-4 * rs.w[0]).astype(np.float32), rtol=1e-6)
    assert isinstance(rs.DATE_OBS, float)
    assert (rs.bcv, rs.specsys, rs.wavelength_frame, rs.time_system) == \
        (-8.92126133253, "TOPOCENT", "vacuum", "JD-UTC")


def test_apogee_masking_interpolates_and_holds_edges():
    rs = ReadSpec("APOGEE")
    rs.s = [np.array([9.0, 2.0, 99.0, 4.0, np.nan, 8.0, 7.0])]
    sky, telluric, other = 1 << 12, 1 << 13, 1 << 0
    flags = [sky, 0, telluric, 0, 0, other, sky | telluric]
    rs.bool_mask = [format(f, "016b") for f in flags]
    rs.APOGEE_masking()
    # pixel 0 and 6 are edge runs -> nearest good value; 2 and the NaN at 4
    # are interpolated in pixel index; bit 0 is not a sky/telluric flag.
    np.testing.assert_allclose(rs.s[0], [2.0, 2.0, 3.0, 4.0, 6.0, 8.0, 8.0])


def test_apogee_masking_refuses_other_surveys_and_fully_masked():
    with pytest.raises(ValueError):
        ReadSpec("HARPS").APOGEE_masking()
    rs = ReadSpec("APOGEE")
    rs.s = [np.ones(3)]
    rs.bool_mask = [format(1 << 13, "016b")] * 3
    with pytest.raises(ValueError, match="masked"):
        rs.APOGEE_masking()


# --------------------------------------------------------------------------
# CERES cubes: HARPS_15 / FEROS_15
# --------------------------------------------------------------------------
def _ceres(tmp_path, name="ceres.fits"):
    n = 2048
    cube = np.zeros((11, 3, n))
    for k, start in enumerate([5000.0, 5100.0, 5200.0]):
        cube[0, k] = np.linspace(start, start + 120.0, n)
        cube[3, k] = 10.0 * k + np.linspace(0.0, 1.0, n)
    hdr = fits.Header()
    hdr.update({"HIERARCH MBJD": 58000.5, "HIERARCH BARYCENTRIC CORRECTION (KM/S)": 3.2,
                "SNR": 80.0, "RV": 11.1})
    return _write(tmp_path, [fits.PrimaryHDU(cube, hdr)], name)


@pytest.mark.parametrize("survey, frame", [("HARPS_15", "vacuum"), ("FEROS_15", None)])
def test_ceres_picks_order_covering_window(tmp_path, survey, frame):
    rs = _load(survey, _ceres(tmp_path), min_wv=5150.0, max_wv=5215.0)
    _check_single_order(rs, 5150.0, 5215.0)
    assert np.all((rs.s[0] >= 10.0) & (rs.s[0] < 20.0))    # order 1 (5100-5220 A)
    assert (rs.DATE_OBS, rs.bcv, rs.vrad, rs.snr) == (58000.5, 0.0, 11.1, 80.0)
    assert (rs.time_system, rs.specsys, rs.wavelength_frame) == ("MBJD-UTC", "BARYCENT", frame)
    with pytest.raises(ValueError, match="no echelle order"):
        _load(survey, _ceres(tmp_path, "b.fits"), min_wv=6000.0, max_wv=6100.0)


# --------------------------------------------------------------------------
# CORALIE / SRG (linear WCS images)
# --------------------------------------------------------------------------
def _linear_image(tmp_path, extra, n=20001):
    hdr = fits.Header()
    hdr.update({"CRVAL1": 4000.0, "CDELT1": 0.1, "CRPIX1": 1.0, **extra})
    return _write(tmp_path, [fits.PrimaryHDU(np.arange(n, dtype=np.float32), hdr)])


def test_coralie_linear_wcs(tmp_path):
    path = _linear_image(tmp_path, {"HIERARCH ESO DRS BJD": 2455000.5,
                                    "HIERARCH ESO OBS TARG CORBAORB": 1.0})
    rs = _load("CORALIE", path, min_wv=4900.0, max_wv=5150.0)
    _check_single_order(rs, 4900.0, 5150.0)
    np.testing.assert_allclose(rs.w[0], 4000.0 + 0.1 * rs.s[0], atol=1e-9)
    assert (rs.bcv, rs.specsys) == (0.0, "BARYCENT")


@pytest.mark.parametrize("extra, airmass", [({"AIRMASS": 1.3}, 1.3), ({}, None)])
def test_srg_airmass_and_window(tmp_path, extra, airmass):
    path = _linear_image(tmp_path, {"BJD": 2455000.5, "TEMPCCD": -110.0, **extra})
    rs = _load("SRG", path, min_wv=5000.0, max_wv=5100.0)
    _check_single_order(rs, 5000.0, 5100.0)
    assert rs.air_mass == airmass


# --------------------------------------------------------------------------
# PEPSI and NRES-BANZAI (binary tables)
# --------------------------------------------------------------------------
def test_pepsi_table_rest_frame(tmp_path):
    w = np.linspace(4800.0, 5400.0, 60001)
    table = fits.BinTableHDU.from_columns([
        fits.Column(name="Arg", format="D", array=w),
        fits.Column(name="Fun", format="D", array=np.ones_like(w)),
        fits.Column(name="Var", format="D", array=np.ones_like(w))])
    hdr = fits.Header()
    hdr["JD-OBS"] = 2458000.75
    rs = _load("PEPSI", _write(tmp_path, [fits.PrimaryHDU(header=hdr), table]),
               min_wv=4900.0, max_wv=5150.0)
    _check_single_order(rs, 4900.0, 5150.0)
    assert (rs.specsys, rs.wavelength_frame, rs.DATE_OBS) == ("SOURCE", "air", 2458000.75)


def test_nres_banzai_table(tmp_path):
    n_ord, n = 68, 100
    fiber = np.repeat([1, 2], n_ord)
    wave = np.tile(np.linspace(5000.0, 5010.0, n), (2 * n_ord, 1)) + 10.0 * np.arange(2 * n_ord)[:, None]
    table = fits.BinTableHDU.from_columns([
        fits.Column(name="fiber", format="J", array=fiber),
        fits.Column(name="wavelength", format=f"{n}D", array=wave),
        fits.Column(name="normflux", format=f"{n}D", array=np.ones_like(wave))])
    hdr = fits.Header()
    hdr.update({"SCIFIBER": 2, "BARYCORR": 1500.0, "TCORR": 2459000.1})
    rs = _load("NRES-BANZAI", _write(tmp_path, [fits.PrimaryHDU(header=hdr), table]))
    assert len(rs.w) == len(rs.s) == 68
    np.testing.assert_array_equal(rs.w[0], wave[n_ord])          # first science-fiber row
    assert (rs.bcv, rs.DATE_OBS, rs.time_system, rs.wavelength_frame) == \
        (1.5, 2459000.1, "BJD-TDB", "vacuum")
