"""Tests for Template: its constructors, the hardened PHOENIX download path
(no network required), and the model operations."""
import io
import os
import stat

import numpy as np
import pytest
from astropy.io import fits

from sparta import template as template_module
from sparta.spectrum import Spectrum
from sparta.template import Template, phoenix_spectrum


def test_download_disabled_raises_without_network():
    # A template whose files are not cached locally must raise (not silently
    # reach the network) when allow_download=False.
    with pytest.raises(FileNotFoundError):
        Template.from_phoenix(9999, 9.99, 0, 0, wave_range=(5000, 5100), allow_download=False)


def test_download_is_opt_in_by_default(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(template_module, "download_file", no_network)
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        Template.from_phoenix(9999, 9.99, 0, 0, wave_range=(5000, 5100))


def test_download_url_is_https_and_correct():
    url = template_module._phoenix_url(*template_module._phoenix_filename(5500, 4.5, 0.0, 0.0))
    assert url.startswith("https://")
    assert "/data/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/" in url
    assert url.endswith("lte05500-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")


@pytest.mark.parametrize("feh, alpha, z_dir, tail", [
    (-1.0, 0.2, "Z-1.0.Alpha=+0.20/", "lte05500-4.50-1.0.Alpha=+0.20."),
    (-0.5, 0.0, "Z-0.5", "lte05500-4.50-0.5."),
    (0.5, 0.0, "Z+0.5/", "lte05500-4.50+0.5."),
])
def test_phoenix_names_carry_one_metallicity_sign(feh, alpha, z_dir, tail):
    # Negative [M/H] with alpha used to produce "Z--1.0" / "-4.50--1.0".
    p, fname = template_module._phoenix_filename(5500.0, 4.5, feh, alpha)
    assert p.endswith(z_dir)
    assert fname.startswith(tail)
    assert "--" not in p + fname


def test_from_phoenix_cuts_the_bundled_model_to_the_range():
    t = Template.from_phoenix(5800, 3.5, 0.5, 0.0, wave_range=(5000.0, 5010.0))
    w, s = t.model.wv, t.model.sp
    assert w.shape == s.shape and w.shape[0] == 1          # one order, (1, n)
    assert 5000.0 < w[0, 0] and w[0, -1] <= 5010.0

    # air=True divides the vacuum grid by n_air (Morton 2000: n - 1 = 2.79e-4
    # at 5000 A) before the range is applied: the same pixel, shorter.
    air = Template.from_phoenix(5800, 3.5, 0.5, 0.0, wave_range=(5000.0, 5010.0), air=True)
    vac = Template.from_phoenix(5800, 3.5, 0.5, 0.0, wave_range=(4990.0, 5020.0))
    assert air.model.wv[0, 0] > 5000.0
    k = np.flatnonzero(vac.model.sp[0] == air.model.sp[0, 0])
    assert k.size == 1
    assert vac.model.wv[0, k[0]] / air.model.wv[0, 0] == pytest.approx(1 + 2.79e-4, abs=2e-6)


@pytest.mark.parametrize("wave_range", [(5100.0, 5000.0), (5000.0, 1e9)])
def test_from_phoenix_refuses_a_range_off_the_grid(wave_range):
    with pytest.raises(ValueError, match="wave_range"):
        Template.from_phoenix(5800, 3.5, 0.5, 0.0, wave_range=wave_range)


@pytest.mark.parametrize("air", [False, True])
def test_phoenix_spectrum_is_the_array_behind_from_phoenix(air):
    wave, flux = phoenix_spectrum(5800, 3.5, 0.5, 0.0, wave_range=(5000.0, 5010.0), air=air)
    assert wave.ndim == flux.ndim == 1 and wave.shape == flux.shape
    assert 5000.0 < wave[0] and wave[-1] <= 5010.0
    model = Template.from_phoenix(5800, 3.5, 0.5, 0.0, wave_range=(5000.0, 5010.0),
                                  air=air).model
    assert np.array_equal(model.wv, wave.reshape(1, -1))
    assert np.array_equal(model.sp, flux.reshape(1, -1))
    assert model.sp.dtype == flux.dtype


def test_phoenix_spectrum_is_vacuum_unless_air_is_asked_for():
    vac, f_vac = phoenix_spectrum(5800, 3.5, 0.5, wave_range=(4990.0, 5020.0))
    air, f_air = phoenix_spectrum(5800, 3.5, 0.5, wave_range=(5000.0, 5010.0), air=True)
    k = np.flatnonzero(f_vac == f_air[0])
    assert k.size == 1
    # Morton (2000): n_air - 1 = 2.79e-4 at 5000 A.
    assert vac[k[0]] / air[0] == pytest.approx(1 + 2.79e-4, abs=2e-6)


def test_phoenix_spectrum_never_downloads_by_default(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(template_module, "download_file", no_network)
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        phoenix_spectrum(9999, 9.99, wave_range=(5000.0, 5100.0))
    with pytest.raises(ValueError, match="wave_range"):
        phoenix_spectrum(5800, 3.5, 0.5, wave_range=(5100.0, 5000.0))


# --- the PHOENIX cache: atomic, verified writes; no network (download_file is mocked)

_MODEL = template_module._phoenix_filename(9000, 4.0, 0.0, 0.0)


def _fits_bytes():
    buf = io.BytesIO()
    fits.PrimaryHDU(np.arange(1.0, 9.0, dtype=np.float32)).writeto(buf)
    return buf.getvalue()


@pytest.fixture
def bundled(tmp_path, monkeypatch):
    """The package's own models: an empty, existing directory."""
    root = tmp_path / "bundled"
    root.mkdir()
    monkeypatch.setattr(template_module, "TEMPLATE_FILES_PATH", str(root))
    return root


@pytest.fixture
def cache(tmp_path, monkeypatch, bundled):
    """The user cache, under SPARTA_CACHE_DIR; not created: the download must create it."""
    monkeypatch.setenv("SPARTA_CACHE_DIR", str(tmp_path / "user"))
    return tmp_path / "user" / "phoenix"


@pytest.fixture
def fake_download(tmp_path, monkeypatch):
    """Stand-in for astropy's download_file: writes ``content`` to a temp file
    outside the cache (as astropy does) and records every call."""
    fetch_dir = tmp_path / "fetch"
    fetch_dir.mkdir()
    state = {"calls": [], "content": _fits_bytes(), "fetched": []}

    def download_file(url, **kwargs):
        state["calls"].append((url, kwargs))
        fetched = fetch_dir / f"download{len(state['calls'])}"
        fetched.write_bytes(state["content"])
        state["fetched"].append(fetched)
        return str(fetched)

    monkeypatch.setattr(template_module, "download_file", download_file)
    return state


def test_download_is_installed_whole_over_default_tls(cache, fake_download):
    path = template_module._cached_phoenix_file(*_MODEL, allow_download=True)
    assert path == os.path.join(os.path.realpath(cache), _MODEL[1])
    assert open(path, "rb").read() == fake_download["content"]
    assert os.listdir(cache) == [_MODEL[1]]                  # no temporary left over
    assert not any(f.exists() for f in fake_download["fetched"])

    (url, kwargs), = fake_download["calls"]
    assert url == template_module._phoenix_url(*_MODEL)
    assert url.startswith("https://phoenix.astro.physik.uni-goettingen.de/data/")
    assert "ssl_context" not in kwargs and "allow_insecure" not in kwargs
    assert kwargs["cache"] is False

    assert template_module._cached_phoenix_file(*_MODEL, allow_download=True) == path
    assert len(fake_download["calls"]) == 1                  # now served from the cache


@pytest.mark.parametrize("previous", [None, b"SIMPLE  = old partial file"])
def test_an_interrupted_copy_never_reaches_the_final_path(cache, fake_download, monkeypatch,
                                                          previous):
    final = cache / _MODEL[1]
    if previous is not None:
        cache.mkdir(parents=True)
        final.write_bytes(previous)

    def copy_half_then_interrupt(src, dst):
        dst.write(src.read(100))
        raise KeyboardInterrupt

    monkeypatch.setattr(template_module.shutil, "copyfileobj", copy_half_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        template_module._cached_phoenix_file(*_MODEL, allow_download=True)

    if previous is None:
        assert not final.exists()
        assert os.listdir(cache) == []
    else:
        assert final.read_bytes() == previous
        assert os.listdir(cache) == [_MODEL[1]]
    assert not any(f.exists() for f in fake_download["fetched"])


def test_a_download_that_is_not_fits_is_not_installed(cache, fake_download):
    fake_download["content"] = b"<html>502 Bad Gateway</html>"
    with pytest.raises(OSError, match="does not open as FITS"):
        template_module._cached_phoenix_file(*_MODEL, allow_download=True)
    assert os.listdir(cache) == []
    assert not any(f.exists() for f in fake_download["fetched"])


@pytest.mark.parametrize("corrupt", [b"", b"not a FITS file", _fits_bytes()[:2880]],
                         ids=["empty", "foreign", "truncated"])
def test_a_corrupt_cached_file_is_reported_not_trusted(cache, fake_download, corrupt):
    cache.mkdir(parents=True)
    final = cache / _MODEL[1]
    final.write_bytes(corrupt)

    with pytest.raises(OSError, match="corrupt or truncated") as info:
        template_module._cached_phoenix_file(*_MODEL, allow_download=False)
    assert _MODEL[1] in str(info.value)
    assert final.read_bytes() == corrupt                     # reported, not deleted
    assert fake_download["calls"] == []

    template_module._cached_phoenix_file(*_MODEL, allow_download=True)
    assert final.read_bytes() == fake_download["content"]
    assert os.listdir(cache) == [_MODEL[1]]


@pytest.mark.parametrize("teff, logg, feh, alpha", [
    (np.nan, 4.5, 0.0, 0.0), (np.inf, 4.5, 0.0, 0.0), (-5500, 4.5, 0.0, 0.0), (0, 4.5, 0.0, 0.0),
    (5500, np.nan, 0.0, 0.0), (5500, -1.0, 0.0, 0.0), (5500, 4.5, np.inf, 0.0),
    (5500, 4.5, 0.0, -np.inf), (5500, 4.5, 0.0, np.nan), ("../../x", 4.5, 0.0, 0.0),
])
def test_bad_model_parameters_raise_before_any_io(monkeypatch, teff, logg, feh, alpha):
    def no_io(*args, **kwargs):
        raise AssertionError("cache or network touched")

    monkeypatch.setattr(template_module, "_cached_phoenix_file", no_io)
    monkeypatch.setattr(template_module, "download_file", no_io)
    with pytest.raises(ValueError, match="PHOENIX teff"):
        template_module._phoenix_filename(teff, logg, feh, alpha)
    with pytest.raises(ValueError, match="PHOENIX teff"):
        phoenix_spectrum(teff, logg, feh, alpha, wave_range=(5000.0, 5100.0),
                         allow_download=True)


@pytest.mark.parametrize("name", ["../escape.fits", "/etc/passwd", "sub/model.fits",
                                  "..", ".", ""])
def test_a_cache_path_cannot_leave_the_cache_dir(cache, fake_download, name):
    with pytest.raises(ValueError, match="directly inside"):
        template_module._cached_phoenix_file("/HiResFITS/", name, allow_download=True)
    assert fake_download["calls"] == []
    assert not cache.exists()


def test_from_mask_marks_the_windows_with_negative_weights(tmp_path):
    mask = tmp_path / "toy.mas"
    mask.write_text("# start end weight\n"
                    "5001.00 5001.10 0.5\n"
                    "5004.00 5004.20 1.0\n")
    t = Template.from_mask(str(mask), wave_range=(5000.0, 5006.0))
    w, s = t.model.wv[0], t.model.sp[0]
    assert t.model.wv.shape == t.model.sp.shape == (1, w.size)
    assert 5000.0 < w[0] and w[-1] <= 5006.0
    first = (w > 5001.00) & (w < 5001.10)
    second = (w > 5004.00) & (w < 5004.20)
    assert first.any() and second.any()
    np.testing.assert_array_equal(s[first], -0.5)
    np.testing.assert_array_equal(s[second], -1.0)
    np.testing.assert_array_equal(s[~(first | second)], 0.0)


def test_from_arrays_wraps_one_order():
    w = np.linspace(5000.0, 5100.0, 50)
    t = Template.from_arrays(w, np.ones_like(w))
    assert t.model.wv.shape == t.model.sp.shape == (1, 50)
    np.testing.assert_array_equal(t.model.wv[0], w)
    with pytest.raises(ValueError, match="1-D"):
        Template.from_arrays(w, np.ones(49))
    with pytest.raises(ValueError, match="1-D"):
        Template.from_arrays(np.vstack([w, w]), np.ones((2, 50)))


def test_the_constructor_takes_a_spectrum():
    spec = Spectrum(wv=[np.linspace(5000.0, 5010.0, 30)], sp=[np.ones(30)])
    assert Template(spec).model is spec
    with pytest.raises(TypeError, match="from_arrays"):
        Template(np.ones(30))


def test_broadening_keeps_every_order():
    # Both broadening methods used to keep only order 0, leaving model.wv with
    # N orders and model.sp with one -- the next CCF then raised IndexError.
    import numpy as np
    from sparta.spectrum import Spectrum
    from unicor import log_lambda_grid
    waves = [log_lambda_grid(lo, lo + 50.0, 0.5)[0] for lo in (5000.0, 5100.0)]
    sps = [1.0 - 0.5 * np.exp(-0.5 * ((w - w.mean()) / 0.1) ** 2) for w in waves]
    t = Template(Spectrum(wv=waves, sp=sps, name="two-order"))
    t.broaden_instrumental(resolution=50_000)
    t.broaden_rotational(vsini=10.0, epsilon=0.6)
    assert len(t.model.sp) == 2
    for w, s, s0 in zip(t.model.wv, t.model.sp, sps):
        assert len(s) == len(w)
        assert s.min() > s0.min()          # each order was actually broadened


def _phoenix_like_order():
    """A linear-lambda order across 5000 A with PHOENIX HiRes's step change
    (0.006 A below, 0.01 A above: 0.36 vs 0.60 km/s) and the same narrow
    line (sigma 0.5 km/s) 10 A to each side of it."""
    from unicor import C_KMS
    wave = np.concatenate([np.arange(4980.0, 5000.0, 0.006),
                           np.arange(5000.0, 5020.0 + 1e-9, 0.01)])
    flux = np.ones_like(wave)
    for centre in (4990.0, 5010.0):
        flux -= 0.5 * np.exp(-0.5 * (C_KMS * np.log(wave / centre) / 0.5) ** 2)
    return wave, flux


def _line_sigma_kms(wave, flux, centre, half_width_kms=60.0):
    """Velocity second moment (km/s) of the absorption line at ``centre``."""
    from unicor import C_KMS
    x = C_KMS * np.log(np.asarray(wave) / centre)
    near = np.abs(x) < half_width_kms
    depth = 1.0 - np.asarray(flux)[near]
    return np.sqrt(np.sum(depth * x[near] ** 2) / np.sum(depth))


def test_instrumental_broadening_has_the_requested_width_across_a_step_change():
    # One median velocity step per order broadened the 0.60 km/s side of a
    # PHOENIX order spanning 5000 A by ~1.67x the requested width; the kernel
    # now acts on a uniform ln(lambda) grid.
    from unicor import C_KMS
    wave, flux = _phoenix_like_order()
    t = Template.from_arrays(wave, flux).broaden_instrumental(resolution=50_000)
    sigma_r = C_KMS / 50_000 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    expected = np.hypot(0.5, sigma_r)
    for centre in (4990.0, 5010.0):
        assert _line_sigma_kms(t.model.wv[0], t.model.sp[0], centre) == pytest.approx(expected, rel=0.01)


def test_rotational_broadening_is_the_same_on_both_sides_of_a_step_change():
    wave, flux = _phoenix_like_order()
    t = Template.from_arrays(wave, flux).broaden_rotational(vsini=20.0, epsilon=0.6)
    blue, red = (_line_sigma_kms(t.model.wv[0], t.model.sp[0], c) for c in (4990.0, 5010.0))
    assert blue == pytest.approx(red, rel=0.01)
    assert blue > 5.0                      # actually broadened (intrinsic 0.5 km/s)


def test_a_non_log_order_keeps_its_grid_and_the_round_trip_is_lossless():
    # Zero width isolates the ln(lambda) round trip: on a resolved line (sigma
    # 2 km/s on 0.36-0.60 km/s pixels) it must give the input back.
    from unicor import C_KMS
    wave = np.concatenate([np.arange(4980.0, 5000.0, 0.006),
                           np.arange(5000.0, 5020.0 + 1e-9, 0.01)])
    flux = 1.0 - 0.5 * np.exp(-0.5 * (C_KMS * np.log(wave / 5010.0) / 2.0) ** 2)
    t = Template.from_arrays(wave, flux).broaden_instrumental(resolution=np.inf)
    assert np.array_equal(t.model.wv[0], wave)
    assert isinstance(t.model.sp, np.ndarray) and t.model.sp.shape == (1, wave.size)
    np.testing.assert_allclose(t.model.sp[0], flux, rtol=0, atol=1e-4)


def test_broadening_leaves_a_log_lambda_order_on_its_grid():
    # Already velocity-even: broadened where it is, exactly as the kernel alone.
    from unicor import broaden_gaussian, dv_from_wave, log_lambda_grid, C_KMS
    wave = log_lambda_grid(5000.0, 5050.0, 0.5)[0]
    flux = 1.0 - 0.5 * np.exp(-0.5 * ((wave - 5025.0) / 0.1) ** 2)
    t = Template.from_arrays(wave, flux).broaden_instrumental(resolution=50_000)
    assert np.array_equal(t.model.wv[0], wave)
    assert np.array_equal(t.model.sp[0], broaden_gaussian(flux, dv_from_wave(wave), C_KMS / 50_000))


def test_save_template_model_round_trips_every_order(tmp_path):
    import numpy as np
    from sparta.spectrum import Spectrum
    waves = [np.linspace(5000.0, 5010.0, 30), np.linspace(5100.0, 5120.0, 45)]
    sps = [np.linspace(0.5, 1.0, 30), np.linspace(1.0, 0.7, 45)]
    t = Template(Spectrum(wv=waves, sp=sps, name="two-order"))
    target = t.save_template_model(title="t", path=str(tmp_path))
    with np.load(target) as f:
        for I in range(2):
            np.testing.assert_array_equal(f[f"wv_{I}"], waves[I])
            np.testing.assert_array_equal(f[f"sp_{I}"], sps[I])


def test_cut_multiorder_like_keeps_native_steps_aligned():
    # Cutting a one-order template into the observation's orders multiplies
    # the order count; a recorded native step must follow.
    import numpy as np
    from unicor import dsp
    from sparta.spectrum import Spectrum
    wave, _ = dsp.log_lambda_grid(4990.0, 5110.0, 0.5)
    model = Spectrum(wv=[wave], sp=[np.ones_like(wave)])
    model.interpolate(delta=0.1, method="log")
    tmpl = Template(model)
    obs = Spectrum(wv=[np.linspace(5000.0, 5040.0, 100), np.linspace(5060.0, 5100.0, 100)],
                   sp=[np.ones(100), np.ones(100)])
    tmpl.cut_multiorder_like(obs)
    assert len(tmpl.model.wv) == 2
    assert tmpl.model.native_steps() == pytest.approx([0.5, 0.5], rel=1e-3)


def test_integrate_spec_records_the_coarser_sampling():
    import numpy as np
    from unicor import dsp
    from sparta.spectrum import Spectrum
    wave, _ = dsp.log_lambda_grid(4990.0, 5010.0, 0.5)
    model = Spectrum(wv=[wave, wave], sp=[np.ones_like(wave), np.ones_like(wave)])
    model.Info["NativeStep"] = [0.5, 0.5]
    tmpl = Template(model)
    tmpl.integrate_spec(4)
    assert tmpl.model.native_steps() == pytest.approx([2.0], rel=1e-3)


_C_KMS = 299792.458


def test_doppler_defaults_to_cz_and_offers_the_relativistic_factor():
    import numpy as np
    from astropy import units as u
    w = np.linspace(5000.0, 5100.0, 50)
    t = Template.from_arrays(w, np.ones_like(w))
    b = 150.0 / _C_KMS
    np.testing.assert_allclose(t.doppler(150.0), w[None, :] * (1 + b), rtol=1e-14)
    np.testing.assert_allclose(t.doppler(150.0, velocity_convention="cz"),
                               t.doppler(150.0), rtol=0)
    np.testing.assert_allclose(t.doppler(150.0, velocity_convention="relativistic"),
                               w[None, :] * np.sqrt((1 + b) / (1 - b)), rtol=1e-14)
    np.testing.assert_allclose(t.doppler(150e3 * u.m / u.s, w), w * (1 + b), rtol=1e-14)
    with pytest.raises(ValueError, match="convention"):
        t.doppler(150.0, velocity_convention="optical")


def test_barycentric_correction_stays_cz():
    # The BERV composes as lambda * (1 + bcv/c), independent of the RV
    # convention (see Spectrum.barycentric_correction).
    import numpy as np
    from astropy import units as u
    from sparta.spectrum import Spectrum
    w = np.linspace(5000.0, 5100.0, 50)
    spec = Spectrum(wv=[w.copy()], sp=[np.ones_like(w)], bcv=25.0)
    spec.barycentric_correction()
    np.testing.assert_allclose(spec.wv[0], w * (1 + 25.0 / _C_KMS), rtol=1e-14)
    assert spec.bcv == 0.0 * u.km / u.s


@pytest.mark.parametrize("wave, flux, match", [
    (np.array([5000.0, 5000.01, 5000.01, 5000.03]), np.ones(4), "strictly increasing"),
    (np.array([5000.03, 5000.02, 5000.0]), np.ones(3), "strictly increasing"),
    (np.array([5000.0]), np.ones(1), "strictly increasing"),
    (np.linspace(5000.0, 5010.0, 1001), np.r_[np.ones(500), np.nan, np.ones(500)], "NaN/inf"),
])
def test_broadening_a_non_log_order_rejects_bad_input_clearly(wave, flux, match):
    with pytest.raises(ValueError, match=match):
        Template.from_arrays(wave, flux).broaden_instrumental(resolution=50_000)


def test_broadening_refuses_a_grid_blown_up_by_one_close_pixel_pair():
    # One near-duplicate pixel would set the ln(lambda) step for the whole
    # order: 1e-4 A inside a 0.01 A grid means ~100x the pixels.
    wave = np.linspace(5000.0, 5010.0, 1001)
    wave = np.insert(wave, 501, wave[500] + 1e-4)
    with pytest.raises(ValueError, match="finest"):
        Template.from_arrays(wave, np.ones_like(wave)).broaden_instrumental(resolution=50_000)


# --- where the models live: bundled models are read in place, downloads go
# to the user cache, never into the package

def test_downloads_go_to_the_user_cache_not_the_package(cache, bundled, fake_download):
    path = template_module._cached_phoenix_file(*_MODEL, allow_download=True)
    assert os.path.dirname(path) == os.path.realpath(cache)
    assert os.listdir(bundled) == []


def test_a_bundled_model_is_read_without_touching_the_user_cache(bundled, monkeypatch):
    (bundled / _MODEL[1]).write_bytes(_fits_bytes())
    monkeypatch.setattr(template_module, "BUNDLED_PHOENIX_FILES", frozenset({_MODEL[1]}))

    def no_user_cache():
        raise AssertionError("user cache consulted for a bundled model")

    monkeypatch.setattr(template_module, "_user_cache_dir", no_user_cache)
    path = template_module._cached_phoenix_file(*_MODEL, allow_download=False)
    assert path == os.path.join(os.path.realpath(bundled), _MODEL[1])


def test_a_model_in_the_user_cache_is_found_offline(cache, fake_download):
    cache.mkdir(parents=True)
    (cache / _MODEL[1]).write_bytes(_fits_bytes())
    path = template_module._cached_phoenix_file(*_MODEL, allow_download=False)
    assert path == os.path.join(os.path.realpath(cache), _MODEL[1])
    assert fake_download["calls"] == []


def test_a_missing_model_names_both_places_and_creates_nothing(cache, bundled):
    with pytest.raises(FileNotFoundError) as info:
        template_module._cached_phoenix_file(*_MODEL, allow_download=False)
    message = str(info.value)
    assert str(bundled) in message and str(cache) in message
    assert "allow_download=True" in message
    assert not cache.parent.exists()                         # a lookup creates no directory


def test_a_corrupt_bundled_model_is_reported_and_never_replaced(bundled, cache, fake_download,
                                                                monkeypatch):
    (bundled / _MODEL[1]).write_bytes(b"not a FITS file")
    monkeypatch.setattr(template_module, "BUNDLED_PHOENIX_FILES", frozenset({_MODEL[1]}))
    with pytest.raises(OSError, match="bundled"):
        template_module._cached_phoenix_file(*_MODEL, allow_download=True)
    assert (bundled / _MODEL[1]).read_bytes() == b"not a FITS file"
    assert fake_download["calls"] == []


def test_the_user_cache_follows_sparta_cache_dir_then_xdg_then_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARTA_CACHE_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "x"))
    assert template_module._user_cache_dir() == str(tmp_path / "s" / "phoenix")
    monkeypatch.delenv("SPARTA_CACHE_DIR")
    assert template_module._user_cache_dir() == str(tmp_path / "x" / "sparta" / "phoenix")
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert template_module._user_cache_dir() == os.path.expanduser(
        os.path.join("~", ".cache", "sparta", "phoenix"))
    assert not (tmp_path / "s").exists() and not (tmp_path / "x").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_a_downloaded_model_is_readable_by_every_user(cache, fake_download):
    # A shared cache must not hold files only their downloader can read
    # (the temporary file is created 0600).
    path = template_module._cached_phoenix_file(*_MODEL, allow_download=True)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o644


def test_a_file_left_in_the_package_by_an_older_version_is_not_read(bundled, cache,
                                                                   fake_download):
    # Versions before 0.3.0 downloaded into template_files/; only the shipped
    # models are read there now, so a stray (here truncated) file is ignored.
    (bundled / _MODEL[1]).write_bytes(_fits_bytes()[:2880])
    with pytest.raises(FileNotFoundError, match="not one of the models bundled"):
        template_module._cached_phoenix_file(*_MODEL, allow_download=False)
    path = template_module._cached_phoenix_file(*_MODEL, allow_download=True)
    assert os.path.dirname(path) == os.path.realpath(cache)


def test_the_shipped_models_are_exactly_the_bundled_set():
    shipped = {f for f in os.listdir(template_module.TEMPLATE_FILES_PATH) if f.endswith(".fits")}
    assert shipped == set(template_module.BUNDLED_PHOENIX_FILES)


def test_the_user_cache_expands_home_and_refuses_a_relative_sparta_cache_dir(monkeypatch):
    monkeypatch.setenv("SPARTA_CACHE_DIR", "~/models")
    assert template_module._user_cache_dir() == os.path.join(
        os.path.expanduser("~"), "models", "phoenix")
    monkeypatch.setenv("SPARTA_CACHE_DIR", "models")
    with pytest.raises(ValueError, match="absolute"):
        template_module._user_cache_dir()


@pytest.mark.parametrize("xdg", ["", "relative/cache"])
def test_an_empty_or_relative_xdg_cache_home_is_ignored(monkeypatch, xdg):
    monkeypatch.delenv("SPARTA_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", xdg)
    assert template_module._user_cache_dir() == os.path.join(
        os.path.expanduser("~"), ".cache", "sparta", "phoenix")
