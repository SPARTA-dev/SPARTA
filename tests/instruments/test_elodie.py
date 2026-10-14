"""ELODIE archive 'spec' loader (linear WCS, topocentric) on synthetic FITS."""
import warnings

import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

from fits_builders import check_single_order, load, write
from sparta.Observations import Observations


def _elodie(tmp_path, crpix=1.0, name="elodie.fits"):
    n = 56001
    hdr = fits.Header()
    hdr.update({"CRVAL1": 4000.0, "CDELT1": 0.05, "CRPIX1": crpix, "SN": 55.0,
                "DATE-OBS": "1998-07-02T23:01:02"})
    return write(tmp_path, [fits.PrimaryHDU(np.arange(n, dtype=np.float32), hdr)], name)


def test_elodie_linear_wcs_without_float32_drift(tmp_path):
    with pytest.warns(UserWarning, match="topocentric"):
        spec = load("ELODIE", _elodie(tmp_path), min_wv=6700.0, max_wv=6800.0)
    check_single_order(spec, 6700.0, 6800.0)
    pix = spec.flux[0]                # flux = 0-based pixel index
    np.testing.assert_allclose(spec.wave[0], 4000.0 + 0.05 * pix, rtol=0, atol=1e-9)
    assert spec.time == pytest.approx(Time("1998-07-02T23:01:02").jd, abs=1e-9)
    assert (spec.time_system, spec.specsys, spec.bcv, spec.snr) == ("JD-UTC", "TOPOCENT", 0.0, 55.0)


def test_elodie_honours_crpix(tmp_path):
    with pytest.warns(UserWarning, match="topocentric"):
        spec = load("ELODIE", _elodie(tmp_path, crpix=11.0), min_wv=5000.0, max_wv=5010.0)
    np.testing.assert_allclose(spec.wave[0], 4000.0 + 0.05 * (spec.flux[0] - 10.0), atol=1e-9)


def test_observations_reads_elodie_times_as_floats(tmp_path):
    for k, date in enumerate(["1998-07-02T23:00:00", "1998-07-05T23:00:00"]):
        hdr = fits.Header()
        hdr.update({"CRVAL1": 4000.0, "CDELT1": 0.05, "SN": 50.0, "DATE-OBS": date})
        write(tmp_path, [fits.PrimaryHDU(np.ones(56001, dtype=np.float32), hdr)], f"e{k}.fits")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        obs = Observations(survey="ELODIE", target_visits_lib=str(tmp_path),
                           min_wv=5150.0, max_wv=5300.0)
    times = sorted(obs.time_list)
    assert all(isinstance(t, float) for t in times)
    assert times[1] - times[0] == pytest.approx(3.0, abs=1e-9)
    assert obs.time_system == "JD-UTC"
