"""TRES loader (IRAF multispec, non-linear dispersion) on synthetic FITS."""
import warnings

import numpy as np
import pytest
from astropy.io import fits

from fits_builders import check_single_order, load, write
from sparta.instruments._iraf import _iraf_multispec_wavelengths


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
    return write(tmp_path, [fits.PrimaryHDU(data, hdr)], name)


def _cheb_attr(ap, beam, centre, coeffs, n_pix=2304, wt=1.0, w0=0.0):
    c = " ".join(repr(float(x)) for x in coeffs)
    return (f"{ap} {beam} 2 {centre - 30.0!r} 0.026 {n_pix} 0. 10.0 20.0 "
            f"{wt!r} {w0!r} 1 {len(coeffs)} 1. {float(n_pix)!r} {c}")


TRES_ORDERS = {1: _cheb_attr(1, 90, 5050.0, [5050.0, 30.0, -0.25, 0.01]),
               2: _cheb_attr(2, 89, 5110.0, [5110.0, 30.5, -0.26, 0.011]),
               3: _cheb_attr(3, 88, 5170.0, [5170.0, 31.0, -0.27, 0.012])}


def _specutils_axes(path):
    from specutils import SpectrumCollection
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SpectrumCollection.read(path, format="iraf").spectral_axis.to_value("Angstrom")


def test_tres_chebyshev_matches_specutils_iraf_reader(tmp_path):
    path = _tres(tmp_path, TRES_ORDERS)
    oracle = _specutils_axes(path)
    spec = load("TRES", path, min_wv=5100.0, max_wv=5120.0)
    check_single_order(spec, 5100.0, 5120.0)
    flux = spec.flux[0]
    order = int(flux[0] // 1e4)                    # flux encodes the order index
    assert order == 1                              # 5110 +- 30 A covers the window best
    pix = (flux - 1e4 * order).astype(int)
    np.testing.assert_allclose(spec.wave[0], oracle[order][pix], rtol=0, atol=1e-9)
    assert (spec.time, spec.bcv, spec.specsys) == (2458000.123456, -12.5, "TOPOCENT")


def test_tres_weight_zero_point_and_function_sum(tmp_path):
    n = 2304
    attr = (f"1 90 2 5000. 0.026 {n} 0. 10.0 20.0 "
            f"0.5 2.0 1 2 1. {float(n)!r} 5000. 30. "
            f"0.5 -2.0 2 2 1. {float(n)!r} 5000. 32.")
    spec = load("TRES", _tres(tmp_path, {1: attr}), min_wv=4975.0, max_wv=5025.0)
    p = spec.flux[0] + 1.0                         # 1-based physical pixel
    x = (p - (n + 1.0) / 2) / ((n - 1.0) / 2)
    expected = 0.5 * (2.0 + 5000.0 + 30.0 * x) + 0.5 * (-2.0 + 5000.0 + 32.0 * x)
    np.testing.assert_allclose(spec.wave[0], expected, rtol=0, atol=1e-9)


def test_tres_spline_dispersion_is_not_implemented(tmp_path):
    attr = "1 90 2 5000. 0.026 2304 0. 10.0 20.0 1. 0. 3 2 1. 2304. 5000. 5010. 5020. 5030."
    with pytest.raises(NotImplementedError):
        load("TRES", _tres(tmp_path, {1: attr}), min_wv=4975.0, max_wv=5025.0)


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
    trimmed = write(tmp_path, [fits.PrimaryHDU(data, hdr)], "trimmed.fits")

    # specutils evaluates physical pixels 1..NAXIS1, ignoring LTV1 and nw, so
    # it can only check the untrimmed, full-length orders.
    oracle = _specutils_axes(_tres(tmp_path, TRES_ORDERS, name="oracle.fits"))
    waves_full = _iraf_multispec_wavelengths(fits.getheader(full))
    waves_trim = _iraf_multispec_wavelengths(fits.getheader(trimmed))
    for line in (1, 2):
        np.testing.assert_allclose(waves_full[line], oracle[line - 1], rtol=0, atol=1e-9)
    for line in (1, 2, 3):
        np.testing.assert_array_equal(waves_trim[line], waves_full[line][100:])
    assert waves_trim[3].size == 2100

    spec = load("TRES", trimmed, min_wv=5100.0, max_wv=5120.0)
    flux = spec.flux[0]
    order = int(flux[0] // 1e4)
    pix = (flux - 1e4 * order).astype(int)         # flux encodes the physical pixel - 1
    np.testing.assert_allclose(spec.wave[0], oracle[order][pix], rtol=0, atol=1e-9)


def test_multispec_log_linear_doppler_factor_divides_the_wavelength(tmp_path):
    # IRAF's multispec driver (sys/mwcs/wfmspec.x, wf_msp_fwd) evaluates
    # 10 ** (w1 + dw (p - 1)) and then divides the *wavelength* by (1 + z), as
    # does specutils' reader; the code used to divide the exponent instead.
    w1, dw, n = 3.70, 1.0e-5, 2304
    zs = {1: 0.0, 2: 3e-4}
    path = _tres(tmp_path, {k: f"{k} {90 - k} 1 {w1!r} {dw!r} {n} {z!r} 10.0 20.0"
                            for k, z in zs.items()})
    oracle = _specutils_axes(path)
    waves = _iraf_multispec_wavelengths(fits.getheader(path))
    p = np.arange(1, n + 1)
    for k, z in zs.items():
        np.testing.assert_allclose(waves[k], 10 ** (w1 + dw * (p - 1)) / (1 + z), rtol=1e-13)
        np.testing.assert_allclose(waves[k], oracle[k - 1], rtol=1e-12)
