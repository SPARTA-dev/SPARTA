"""CORALIE s1d loader (linear WCS image) on synthetic FITS."""
import numpy as np

from fits_builders import check_single_order, linear_image, load


def test_coralie_linear_wcs(tmp_path):
    path = linear_image(tmp_path, {"HIERARCH ESO DRS BJD": 2455000.5,
                                   "HIERARCH ESO OBS TARG CORBAORB": 1.0})
    spec = load("CORALIE", path, min_wv=4900.0, max_wv=5150.0)
    check_single_order(spec, 4900.0, 5150.0)
    np.testing.assert_allclose(spec.wave[0], 4000.0 + 0.1 * spec.flux[0], atol=1e-9)
    assert (spec.bcv, spec.specsys, spec.time) == (0.0, "BARYCENT", 2455000.5)
