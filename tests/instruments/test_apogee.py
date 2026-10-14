"""APOGEE apVisit loader (decreasing vacuum wavelengths, observer frame) on synthetic FITS."""
import numpy as np
import pytest
from astropy.io import fits

from fits_builders import load, write
from sparta.instruments.apogee import APOGEE_masking

SKY, TELLURIC, OTHER = 1 << 12, 1 << 13, 1 << 0


def _apogee(tmp_path, n=4096):
    wave = np.stack([np.linspace(16950.0, 16470.0, n),
                     np.linspace(16430.0, 15860.0, n),
                     np.linspace(15800.0, 15150.0, n)])       # chip a is the red one
    flux = (1.0 + 1e-4 * wave).astype(np.float32)              # flux is a function of wavelength
    mask = np.zeros((3, n), dtype=np.int16)
    mask[1, 100] = SKY
    flux[1, 100] = -999.0
    hdr = fits.Header()
    hdr.update({"JD-MID": 2455813.69734, "BC": -8.92126133253, "VRAD": 4.80937})
    return write(tmp_path, [fits.PrimaryHDU(header=hdr), fits.ImageHDU(flux),
                            fits.ImageHDU(np.ones_like(flux)), fits.ImageHDU(mask),
                            fits.ImageHDU(wave)], "apVisit.fits"), wave


def test_apogee_ascending_vacuum_with_bc(tmp_path):
    path, wave = _apogee(tmp_path)
    spec = load("APOGEE", path, segment=1)
    w, s = spec.wave[0], spec.flux[0]
    assert w.dtype == np.float64 and s.dtype == np.float64
    assert np.all(np.diff(w) > 0)
    np.testing.assert_array_equal(w, wave[1][::-1])
    # the sky-flagged -999 pixel is interpolated over
    np.testing.assert_allclose(s, (1.0 + 1e-4 * w).astype(np.float32), rtol=1e-6)
    assert spec.extras["pixmask"][-101] == SKY
    assert isinstance(spec.time, float)
    assert (spec.bcv, spec.specsys, spec.wavelength_frame, spec.time_system, spec.vrad) == \
        (-8.92126133253, "TOPOCENT", "vacuum", "JD-UTC", 4.80937)


def test_apogee_masking_interpolates_and_holds_edges():
    flux = np.array([9.0, 2.0, 99.0, 4.0, np.nan, 8.0, 7.0])
    flags = np.array([SKY, 0, TELLURIC, 0, 0, OTHER, SKY | TELLURIC], dtype=np.int16)
    # pixel 0 and 6 are edge runs -> nearest good value; 2 and the NaN at 4
    # are interpolated in pixel index; bit 0 is not a sky/telluric flag.
    np.testing.assert_allclose(APOGEE_masking(flux, flags), [2.0, 2.0, 3.0, 4.0, 6.0, 8.0, 8.0])
    assert flux[0] == 9.0                          # the input is not modified


def test_apogee_masking_refuses_fully_masked():
    with pytest.raises(ValueError, match="masked"):
        APOGEE_masking(np.ones(3), np.full(3, TELLURIC, dtype=np.int16))
