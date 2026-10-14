"""SRG loader (linear WCS image) on synthetic FITS."""
import numpy as np
import pytest

from fits_builders import check_single_order, linear_image, load


@pytest.mark.parametrize("extra, airmass", [({"AIRMASS": 1.3}, 1.3), ({}, np.nan)])
def test_srg_airmass_and_window(tmp_path, extra, airmass):
    path = linear_image(tmp_path, {"BJD": 2455000.5, "TEMPCCD": -110.0, **extra})
    spec = load("SRG", path, min_wv=5000.0, max_wv=5100.0)
    check_single_order(spec, 5000.0, 5100.0)
    np.testing.assert_equal(spec.air_mass, airmass)
