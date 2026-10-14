"""The public ``import sparta`` surface, pinned.

A later refactor renames and regroups the package; this makes every change to
what users import a deliberate edit of the lists below rather than a side
effect. Checked in a fresh interpreter, so submodules that other tests import
cannot leak into ``dir(sparta)``.

Note the current shadowing: ``sparta.Observations`` and ``sparta.TIRAVEL`` are
the classes, not the submodules of the same name.
"""
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
_PATH = os.pathsep.join(str(ROOT / p) for p in ("src", "packages/unicor/src",
                                                  "packages/todcor/src"))

# name -> "class <module>.<qualname>" or "module <name>"
TOP_LEVEL = {
    "Auxil": "module sparta.Auxil",
    "CCF1d": "class sparta.UNICOR.CCF1d.CCF1d",
    "CCF2d": "class sparta.TODCOR.CCF2d.CCF2d",
    "Observations": "class sparta.Observations.Observations",
    "TIRAVEL": "class sparta.TIRAVEL.TIRAVEL.TIRAVEL",
    "TODCOR": "module sparta.TODCOR",
    "TimeSeries": "class sparta.Auxil.TimeSeries.TimeSeries",
    "UNICOR": "module sparta.UNICOR",
    "USURPER": "module sparta.USURPER",
    "instruments": "module sparta.instruments",
}

# The deep import paths the examples and docs use: module -> names that must
# resolve there (existence, not an exhaustive list).
DEEP = {
    "sparta.UNICOR": ["CCF1d", "Spectrum", "Template"],
    "sparta.UNICOR.Spectrum": ["Spectrum", "resample_flux"],
    "sparta.UNICOR.Template": ["Template"],
    "sparta.UNICOR.CCF1d": ["CCF1d"],
    "sparta.TODCOR": ["CCF2d"],
    "sparta.TODCOR.CCF2d": ["CCF2d"],
    "sparta.TIRAVEL.tiravel_ops": ["align_spectra", "alignment", "coadd", "common_grid",
                                   "correlation_matrix", "normalize_rows",
                                   "rest_frame_wavelengths"],
    "sparta.Auxil.TimeSeries": ["TimeSeries"],
    "sparta.Auxil.PeriodicityDetector": ["PeriodicityDetector", "is_numeric_scalar"],
    "sparta.USURPER.USURPER_functions": ["calc_PDC", "calc_PDC_unbiased",
                                         "calc_pdc_distance_matrix", "get_wv_grid",
                                         "inner_prod", "norm_z",
                                         "resample_spec_to_common_grid",
                                         "unbiased_u_centering"],
    "sparta.instruments": ["Component", "Instrument", "LoadedSpectrum", "MODULES", "Product",
                           "REGISTRY", "Recommended", "Simulation", "Truth", "get",
                           "render_markdown"],
    "sparta.instruments.simulated": ["Component", "Simulation", "Truth", "component_spectra",
                                     "component_templates", "simulate", "upper_envelope"],
}

_PROBE = r"""
import importlib, inspect, json, sys
import sparta

def kind(obj):
    if inspect.ismodule(obj):
        return "module " + obj.__name__
    if inspect.isclass(obj):
        return "class " + obj.__module__ + "." + obj.__qualname__
    return type(obj).__name__

top = {n: kind(getattr(sparta, n)) for n in dir(sparta) if not n.startswith("_")}
deep = {m: [n for n in json.loads(sys.argv[1])[m] if hasattr(importlib.import_module(m), n)]
        for m in json.loads(sys.argv[1])}
print(json.dumps({"top": top, "deep": deep}))
"""


def _probe():
    env = dict(os.environ, PYTHONPATH=_PATH + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run([sys.executable, "-c", _PROBE, json.dumps(DEEP)],
                          capture_output=True, text=True, env=env, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_import_sparta_surface_is_pinned():
    probe = _probe()
    assert probe["top"] == TOP_LEVEL
    assert probe["deep"] == DEEP
