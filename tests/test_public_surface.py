"""The public ``import sparta`` surface, pinned.

Every change to what users import must be a deliberate edit of the lists
below rather than a side effect. Checked in a fresh interpreter, so
submodules that other tests import cannot leak into ``dir(sparta)``.

The layout is flat: one module per concern (``sparta.spectrum``,
``sparta.ccf1d``, ...), its class re-exported at the top level. The class and
its module differ in case only for ``Observations``/``observations`` and
``TIRAVEL``/``tiravel``.
"""
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
_PATH = os.pathsep.join(str(ROOT / p) for p in ("src", "packages/unicor/src",
                                                  "packages/todcor/src"))

ALL = ["Spectrum", "Template", "Observations", "TimeSeries", "CCF1d", "CCF2d",
       "TIRAVEL", "PeriodicityDetector", "instruments"]

# name -> "class <module>.<qualname>" or "module <name>"
TOP_LEVEL = {
    "CCF1d": "class sparta.ccf1d.CCF1d",
    "CCF2d": "class sparta.ccf2d.CCF2d",
    "Observations": "class sparta.observations.Observations",
    "PeriodicityDetector": "class sparta.periodogram.PeriodicityDetector",
    "Spectrum": "class sparta.spectrum.Spectrum",
    "TIRAVEL": "class sparta.tiravel.TIRAVEL",
    "Template": "class sparta.template.Template",
    "TimeSeries": "class sparta.timeseries.TimeSeries",
    "ccf1d": "module sparta.ccf1d",
    "ccf2d": "module sparta.ccf2d",
    "instruments": "module sparta.instruments",
    "observations": "module sparta.observations",
    "pdc": "module sparta.pdc",
    "periodogram": "module sparta.periodogram",
    "spectrum": "module sparta.spectrum",
    "template": "module sparta.template",
    "timeseries": "module sparta.timeseries",
    "tiravel": "module sparta.tiravel",
    "tiravel_ops": "module sparta.tiravel_ops",
}

# The deep import paths the examples and docs use: module -> names that must
# resolve there (existence, not an exhaustive list).
DEEP = {
    "sparta.spectrum": ["Spectrum", "resample_flux"],
    "sparta.template": ["Template"],
    "sparta.ccf1d": ["CCF1d"],
    "sparta.ccf2d": ["CCF2d"],
    "sparta.tiravel": ["TIRAVEL"],
    "sparta.tiravel_ops": ["align_spectra", "alignment", "coadd", "common_grid",
                           "correlation_matrix", "normalize_rows",
                           "rest_frame_wavelengths"],
    "sparta.timeseries": ["TimeSeries", "is_numeric_scalar"],
    "sparta.periodogram": ["PeriodicityDetector"],
    "sparta.pdc": ["double_centering", "error_aware_distance_matrix", "get_wv_grid",
                   "inner_prod", "norm_z", "partial_centering", "pdc_power",
                   "phase_distance_matrix", "resample_spec_to_common_grid",
                   "scalar_distance_matrix", "spectral_distance_matrix",
                   "unbiased_u_centering"],
    "sparta.observations": ["Observations"],
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
print(json.dumps({"top": top, "deep": deep, "all": sparta.__all__,
                  "version": sparta.__version__}))
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
    assert probe["all"] == ALL
    assert isinstance(probe["version"], str) and probe["version"]
