"""PEPSI table loader (stellar rest frame) on synthetic FITS."""
import numpy as np
from astropy.io import fits

from fits_builders import check_single_order, load, write


def test_pepsi_table_rest_frame(tmp_path):
    w = np.linspace(4800.0, 5400.0, 60001)
    table = fits.BinTableHDU.from_columns([
        fits.Column(name="Arg", format="D", array=w),
        fits.Column(name="Fun", format="D", array=np.ones_like(w)),
        fits.Column(name="Var", format="D", array=np.ones_like(w))])
    hdr = fits.Header()
    hdr["JD-OBS"] = 2458000.75
    spec = load("PEPSI", write(tmp_path, [fits.PrimaryHDU(header=hdr), table]),
                min_wv=4900.0, max_wv=5150.0)
    check_single_order(spec, 4900.0, 5150.0)
    assert (spec.specsys, spec.wavelength_frame, spec.time) == ("SOURCE", "air", 2458000.75)
