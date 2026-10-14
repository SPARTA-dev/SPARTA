"""NRES-BANZAI table loader on synthetic FITS (the e91 product is pinned on the bundled
TOI677 files in test_bundled_data.py and test_golden_instruments.py)."""
import numpy as np
from astropy.io import fits

from fits_builders import load, write


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
    spec = load("NRES-BANZAI", write(tmp_path, [fits.PrimaryHDU(header=hdr), table]))
    assert len(spec.wave) == len(spec.flux) == n_ord
    np.testing.assert_array_equal(spec.wave[0], wave[n_ord])     # first science-fiber row
    np.testing.assert_array_equal(spec.wave[-1], wave[-1])
    assert (spec.bcv, spec.time, spec.time_system, spec.wavelength_frame) == \
        (1.5, 2459000.1, "BJD-TDB", "vacuum")
