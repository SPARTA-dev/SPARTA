"""Golden master for the bundled HARPS (TOI193) and NRES (TOI677) data.

The reference values in ``golden/instruments.json`` were produced by the
legacy ``Auxil/ReadSpec.py`` reader (commit 0a5eb9c) before it was split into
per-instrument modules; every refactor of the loaders or of ``Observations``
must reproduce them bit for bit.

Spectra are pinned by SHA-256 hashes, not arrays, to keep the golden file
small. The hash is taken over the float64 *values* order by order (each order
prefixed by its int64 length), so it does not depend on the container -- a
2-D (1, N) array and a list of 1-D arrays hash the same -- but it does catch a
changed value, pixel count, order split or dtype.

"Unknown" scalars are stored as ``null``: the legacy sentinels (snr -1,
air_mass / vrad ``[]``) and their replacements (NaN, None) both map to it.

Regenerate only on purpose, after confirming a change in the reference is
intended, with ``_record_loader`` / ``_record_observations`` /
``_record_unicor`` below.
"""
import hashlib
import json
import math
import os
import pathlib

import numpy as np
import pytest

from sparta import instruments
from sparta.observations import Observations

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "examples" / "data"
GOLDEN_PATH = pathlib.Path(__file__).resolve().parent / "golden" / "instruments.json"

DATASETS = {
    # survey: (directory, wavelength window in Angstrom as passed to the loader)
    "HARPS": ("TOI193", {"min_wv": 4990.0, "max_wv": 5090.0}),   # TIRAVEL_Quickstart
    "NRES": ("TOI677", {"min_wv": 5150.0, "max_wv": 5300.0}),    # ignored by NRES
}
# UNICOR_quickstart.ipynb, run on two of the NRES epochs.
UNICOR_FILES = ("lscnrs01-fa09-20190522-0037-e91.fits",
                "lscnrs01-fa09-20190616-0018-e91.fits")


def _files(survey):
    return sorted((DATA / DATASETS[survey][0]).glob("*.fits"))


HAVE_DATA = all(_files(s) for s in DATASETS)
pytestmark = pytest.mark.skipif(not HAVE_DATA, reason="bundled example data not present")


def _golden():
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def hash_orders(orders):
    """SHA-256 over float64 values, order by order, each prefixed by its length."""
    h = hashlib.sha256()
    for order in orders:
        a = np.asarray(order)
        if a.ndim != 1 or a.dtype != np.float64:
            raise TypeError(f"expected 1-D float64 orders, got ndim={a.ndim} dtype={a.dtype}")
        h.update(np.int64(a.size).tobytes())
        h.update(np.ascontiguousarray(a, dtype="<f8").tobytes())
    return h.hexdigest()


def _known(x):
    """A scalar, or None for any 'unknown' marker (legacy -1 / [] or NaN / None)."""
    if x is None or (isinstance(x, list) and not x):
        return None
    x = float(x)
    return None if math.isnan(x) else x


def _known_snr(x):
    return None if x == -1 else _known(x)


def _load(survey, path, window):
    """The loader under test, as a plain dict."""
    out = instruments.get(survey).load(path, **window)
    return {"w": out.wave, "s": out.flux, "time": out.time, "bcv": out.bcv, "snr": out.snr,
            "air_mass": out.air_mass, "vrad": out.vrad, "time_system": out.time_system,
            "wavelength_frame": out.wavelength_frame, "specsys": out.specsys}


def _record_loader(survey, path):
    out = _load(survey, path, DATASETS[survey][1])
    return {"n_orders": len(out["w"]),
            "w_sha256": hash_orders(out["w"]), "s_sha256": hash_orders(out["s"]),
            "time": float(out["time"]), "bcv": float(out["bcv"]),
            "snr": _known_snr(out["snr"]), "air_mass": _known(out["air_mass"]),
            "vrad": _known(out["vrad"]),
            "time_system": out["time_system"], "wavelength_frame": out["wavelength_frame"],
            "specsys": out["specsys"]}


def _record_observations(survey, directory):
    obs = Observations(survey=survey, target_visits_lib=str(directory), **DATASETS[survey][1])
    epochs = {}
    for k, path in enumerate(obs.file_list):
        sp = obs.spec_list[k]
        epochs[os.path.basename(path)] = {
            "time": float(obs.time_list[k]), "bcv": float(obs.bcv[k]),
            "snr": _known_snr(obs.snr[k]), "air_mass": _known(obs.air_mass[k]),
            "wv_sha256": hash_orders(sp.wv), "sp_sha256": hash_orders(sp.sp),
            "spec_bcv_kms": float(sp.bcv.to_value("km/s"))}
    return {"first_time": obs.first_time, "time_system": obs.time_system,
            "wavelength_frame": obs.wavelength_frame, "sample_size": obs.sample_size,
            "n_vrad": len(obs.vrad_list), "epochs": epochs}


def _record_unicor(directory):
    """RV / eRV (km/s) per epoch, exactly as UNICOR_quickstart.ipynb computes them."""
    from sparta.template import Template
    obs = Observations(survey="NRES", target_visits_lib=str(directory))
    obs.select_orders([10 + i for i in np.arange(40)], remove=False)
    obs.preprocess(n_trim=100, clean_margins=True, remove_nans=True, delta=0.5,
                   cosmic_sigma=3, filter_lowcut=4, filter_highcut=0.15, alpha=0.3)
    template = Template.from_phoenix(5800, 3.5, 0.5, 0.0, wave_range=(4650, 7500),
                                     air=False, allow_download=False)
    template.integrate_spec(integration_ratio=3)
    template.model.interpolate(delta=1)
    template.broaden_rotational(vsini=6, epsilon=0.5)
    template.broaden_instrumental(resolution=53000)
    template.cut_multiorder_like(obs.spec_list[0], margins=150)
    template.model.preprocess(n_trim=10, clean_margins=False, remove_nans=False, delta=1,
                              cosmic_sigma=3, filter_lowcut=4, filter_highcut=0.15, alpha=0.3)
    obs.calc_rv_against_template(template, dv=0.05, vel_bound=[-50, 100], errors_per_order=False,
                                 combine_ccfs=True)
    return {os.path.basename(p): {"rv": float(v), "erv": float(e)}
            for p, v, e in zip(obs.file_list, obs.vels, obs.evels)}


def _unicor_dir(tmp_path):
    for name in UNICOR_FILES:
        (tmp_path / name).symlink_to(DATA / "TOI677" / name)
    return tmp_path


def _cases():
    if not HAVE_DATA:
        return []
    return [(s, p) for s in DATASETS for p in _files(s)]


@pytest.mark.parametrize("survey, path", _cases(), ids=lambda x: getattr(x, "name", x))
def test_loader_matches_golden(survey, path):
    assert _record_loader(survey, path) == _golden()["loaders"][survey][path.name]


def test_golden_covers_every_bundled_file():
    golden = _golden()["loaders"]
    for survey in DATASETS:
        assert sorted(golden[survey]) == [p.name for p in _files(survey)]
    assert (len(golden["HARPS"]), len(golden["NRES"])) == (32, 12)


@pytest.mark.parametrize("survey", list(DATASETS))
def test_observations_matches_golden(survey):
    got = _record_observations(survey, DATA / DATASETS[survey][0])
    assert got == _golden()["observations"][survey]


def test_unicor_rvs_match_golden(tmp_path):
    got = _record_unicor(_unicor_dir(tmp_path))
    want = _golden()["unicor"]["NRES"]
    assert sorted(got) == sorted(want)
    for name in want:
        np.testing.assert_allclose([got[name]["rv"], got[name]["erv"]],
                                   [want[name]["rv"], want[name]["erv"]], rtol=0, atol=1e-9)
