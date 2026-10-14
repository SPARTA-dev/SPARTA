"""Tests for Template: its constructors, the hardened PHOENIX download path
(no network required), and the model operations."""
import numpy as np
import pytest

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
