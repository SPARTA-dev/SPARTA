"""Tests for the hardened PHOENIX download path (no network required)."""
import pytest

from sparta.template import Template


def test_download_disabled_raises_without_network():
    # A template whose files are not cached locally must raise (not silently
    # reach the network) when allow_download=False.
    with pytest.raises(FileNotFoundError):
        Template(temp=9999, log_g=9.99, metal=0, alpha=0,
                 min_val=5000, max_val=5100, allow_download=False)


def test_download_is_opt_in_by_default(monkeypatch):
    import sys
    template_module = sys.modules["sparta.template"]   # the package exports the class

    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(template_module, "download_file", no_network)
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        Template(temp=9999, log_g=9.99, metal=0, alpha=0, min_val=5000, max_val=5100)


def test_download_url_is_https_and_correct():
    t = Template.__new__(Template)
    t.server_url = "https://phoenix.astro.physik.uni-goettingen.de/data/"
    p, fname = t.create_PHOENIX_fname(5500, 4.5, 0.0, 0.0)
    url = t.server_url + p.strip("/") + "/" + fname
    assert url.startswith("https://")
    assert "/data/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/" in url
    assert url.endswith("lte05500-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")


@pytest.mark.parametrize("metal, alpha, z_dir, tail", [
    (-1.0, 0.2, "Z-1.0.Alpha=+0.20/", "lte05500-4.50-1.0.Alpha=+0.20."),
    (-0.5, 0.0, "Z-0.5", "lte05500-4.50-0.5."),
    (0.5, 0.0, "Z+0.5/", "lte05500-4.50+0.5."),
])
def test_phoenix_names_carry_one_metallicity_sign(metal, alpha, z_dir, tail):
    # Negative [M/H] with alpha used to produce "Z--1.0" / "-4.50--1.0".
    p, fname = Template.__new__(Template).create_PHOENIX_fname(5500.0, 4.5, metal, alpha)
    assert p.endswith(z_dir)
    assert fname.startswith(tail)
    assert "--" not in p + fname


def test_phxreff_is_read_on_its_own():
    # A copy-paste bug tested 'PHXMASS' before reading 'PHXREFF'.
    import numpy as np
    w = np.linspace(5000.0, 5100.0, 50)
    t = Template(wavelengths=w, spectrum=np.ones_like(w), PHXREFF=7.0e10)
    assert t.PHXREFF == 7.0e10


def test_broadening_keeps_every_order():
    # Both broadening methods used to keep only order 0, leaving model.wv with
    # N orders and model.sp with one -- the next CCF then raised IndexError.
    import numpy as np
    from sparta.spectrum import Spectrum
    from unicor import log_lambda_grid
    waves = [log_lambda_grid(lo, lo + 50.0, 0.5)[0] for lo in (5000.0, 5100.0)]
    sps = [1.0 - 0.5 * np.exp(-0.5 * ((w - w.mean()) / 0.1) ** 2) for w in waves]
    t = Template(template=Spectrum(wv=waves, sp=sps, name="two-order"))
    t.GaussianBroadening(resolution=50_000)
    t.RotationalBroadening(vsini=10.0, epsilon=0.6)
    assert len(t.model.sp) == 2
    for w, s, s0 in zip(t.model.wv, t.model.sp, sps):
        assert len(s) == len(w)
        assert s.min() > s0.min()          # each order was actually broadened


def test_save_template_model_round_trips_every_order(tmp_path):
    import numpy as np
    from sparta.spectrum import Spectrum
    waves = [np.linspace(5000.0, 5010.0, 30), np.linspace(5100.0, 5120.0, 45)]
    sps = [np.linspace(0.5, 1.0, 30), np.linspace(1.0, 0.7, 45)]
    t = Template(template=Spectrum(wv=waves, sp=sps, name="two-order"))
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
    model.InterpolateSpectrum(delta=0.1, InterpMethod="log")
    tmpl = Template(template=model)
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
    tmpl = Template(template=model)
    tmpl.integrate_spec(4)
    assert tmpl.model.native_steps() == pytest.approx([2.0], rel=1e-3)


_C_KMS = 299792.458


def test_doppler_defaults_to_cz_and_offers_the_relativistic_factor():
    import numpy as np
    from astropy import units as u
    w = np.linspace(5000.0, 5100.0, 50)
    t = Template(wavelengths=w, spectrum=np.ones_like(w))
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
    # convention (see Spectrum.BarycentricCorrection).
    import numpy as np
    from astropy import units as u
    from sparta.spectrum import Spectrum
    w = np.linspace(5000.0, 5100.0, 50)
    spec = Spectrum(wv=[w.copy()], sp=[np.ones_like(w)], bcv=25.0)
    spec.BarycentricCorrection()
    np.testing.assert_allclose(spec.wv[0], w * (1 + 25.0 / _C_KMS), rtol=1e-14)
    assert spec.bcv == 0.0 * u.km / u.s
