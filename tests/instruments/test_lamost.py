"""LAMOST low-resolution loader (log-linear, vacuum, heliocentric) on synthetic FITS."""
import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

from fits_builders import check_single_order, load, write
from sparta.Observations import Observations


def _lamost(tmp_path, helio=True, vacuum=True, name="lamost.fits"):
    n = 3909
    data = np.zeros((5, n), dtype=np.float32)
    data[0] = np.arange(n)
    hdr = fits.Header()
    hdr.update({"COEFF0": 3.5682, "COEFF1": 1e-4, "CRVAL1": 3.5682, "CD1_1": 1e-4,
                "CRPIX1": 1, "HELIO": helio, "HELIO_RV": -22.47678, "VACUUM": vacuum,
                "DATE-OBS": "2015-03-11T20:30:51.01"})
    return write(tmp_path, [fits.PrimaryHDU(data, hdr)], name)


def test_lamost_log_linear_solution(tmp_path):
    spec = load("LAMOST", _lamost(tmp_path), min_wv=5000.0, max_wv=5100.0)
    check_single_order(spec, 5000.0, 5100.0)
    w = spec.wave[0]
    i = spec.flux[0].astype(int)      # flux row was set to the pixel index
    np.testing.assert_allclose(w, 10 ** (3.5682 + 1e-4 * i), rtol=1e-13)
    np.testing.assert_allclose(w[1] / w[0], 10 ** 1e-4, rtol=1e-13)
    assert spec.time == pytest.approx(Time("2015-03-11T20:30:51.01").jd, abs=1e-9)
    assert (spec.time_system, spec.wavelength_frame, spec.specsys, spec.bcv) == \
        ("JD-UTC", "vacuum", "HELIOCEN", 0.0)
    assert np.isnan(spec.snr) and np.isnan(spec.air_mass)


def test_lamost_rejects_uncorrected_and_off_window(tmp_path):
    with pytest.raises(NotImplementedError):
        load("LAMOST", _lamost(tmp_path, helio=False), min_wv=5000.0, max_wv=5100.0)
    with pytest.raises(ValueError):
        load("LAMOST", _lamost(tmp_path, name="b.fits"), min_wv=10000.0, max_wv=11000.0)


@pytest.mark.filterwarnings("ignore:loader not verified")
def test_observations_refuses_mixed_wavelength_frames(tmp_path):
    _lamost(tmp_path, vacuum=True, name="a.fits")
    _lamost(tmp_path, vacuum=False, name="b.fits")
    with pytest.raises(ValueError, match="disagree"):
        Observations(survey="LAMOST", target_visits_lib=str(tmp_path),
                     min_wv=5000.0, max_wv=5100.0)
