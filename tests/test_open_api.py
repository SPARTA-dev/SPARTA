"""The open surface: docs/open_api.md, the lazy front door and its import cost.

docs/open_api.md lists, in tables, every atom a user can build an analysis
from and where it imports from. These tests hold it to the code:

* every row's names import from the module the row states, and every
  documented callable or class has a docstring;
* every name ``sparta`` serves lazily (``sparta._LAZY_EXPORTS``) is in the
  document -- a row importing from ``sparta``, or the objects named in the
  "Not on the surface, on purpose" paragraph;
* the lazy table covers every public atom of SPARTA's atom modules, and no
  ``unicor``/``todcor`` name (those are separate packages);
* ``from sparta import X`` is the very object ``sparta.<module>.X``;
* ``import sparta`` -- and importing the atoms -- loads neither matplotlib
  nor specutils (checked in a fresh interpreter, so other tests' imports
  cannot leak in).
"""
import importlib
import inspect
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

import sparta

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "open_api.md"
_PATH = os.pathsep.join(str(ROOT / p) for p in ("src", "packages/unicor/src",
                                                  "packages/todcor/src"))

# SPARTA's atom modules: every public function and class each defines is on
# the front door. The PHOENIX reader is the one atom of sparta.template (whose
# Template is an object, not an atom).
ATOM_MODULES = ("sparta.preprocess", "sparta.pdc", "sparta.tiravel_ops")
EXTRA_ATOMS = {"phoenix_spectrum": "sparta.template"}
HEAVY = ("matplotlib", "specutils")


def _rows():
    '''(names, module) of every table row in docs/open_api.md.'''
    rows = []
    for line in DOC.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("|") or len(cells) < 2 or not cells[0].startswith("`"):
            continue
        modules = re.findall(r"`([^`]+)`", cells[1])
        assert len(modules) == 1, f"row must name one module: {line}"
        rows.append((re.findall(r"`([^`]+)`", cells[0]), modules[0]))
    return rows


def _not_on_surface():
    '''The names in the "Not on the surface, on purpose" paragraph.'''
    text = DOC.read_text(encoding="utf-8")
    match = re.search(r"\*\*Not on the surface, on purpose[^\n]*(?:\n(?!\n).*)*", text)
    assert match, "docs/open_api.md has no 'Not on the surface, on purpose' paragraph"
    return set(re.findall(r"`([^`]+)`", match.group(0)))


def _resolve(name, path):
    module = importlib.import_module(path)
    return module if path == f"sparta.{name}" else getattr(module, name)


def _public_atoms(path):
    module = importlib.import_module(path)
    return [n for n, o in vars(module).items()
            if not n.startswith("_") and (inspect.isfunction(o) or inspect.isclass(o))
            and o.__module__ == path]


# =============================================================================
# The document
# =============================================================================
def test_document_has_tables():
    assert len(_rows()) >= 40


@pytest.mark.parametrize("names, module", _rows(), ids=lambda v: str(v))
def test_every_row_imports_from_its_module(names, module):
    mod = importlib.import_module(module)
    for name in names:
        assert hasattr(mod, name), f"`{name}` does not import from `{module}`"


@pytest.mark.parametrize("names, module", _rows(), ids=lambda v: str(v))
def test_every_documented_atom_has_a_docstring(names, module):
    mod = importlib.import_module(module)
    for name in names:
        obj = getattr(mod, name)
        if callable(obj) or inspect.ismodule(obj):
            assert (inspect.getdoc(obj) or "").strip(), f"{module}.{name} has no docstring"


def test_every_lazy_export_is_documented():
    on_sparta = {n for names, module in _rows() if module == "sparta" for n in names}
    missing = set(sparta._LAZY_EXPORTS) - on_sparta - _not_on_surface()
    assert not missing, f"lazy exports missing from docs/open_api.md: {sorted(missing)}"


def test_rows_importing_from_sparta_are_lazy_exports():
    on_sparta = {n for names, module in _rows() if module == "sparta" for n in names}
    assert on_sparta <= set(sparta._LAZY_EXPORTS)


# =============================================================================
# The lazy table
# =============================================================================
def test_lazy_table_covers_every_public_atom():
    for path in ATOM_MODULES:
        served = {n for n, p in sparta._LAZY_EXPORTS.items() if p == path}
        assert served == set(_public_atoms(path)), path
    for name, path in EXTRA_ATOMS.items():
        assert sparta._LAZY_EXPORTS[name] == path


def test_atom_modules_do_not_share_a_name():
    seen = {}
    for path in ATOM_MODULES + tuple(set(EXTRA_ATOMS.values())):
        for name in _public_atoms(path):
            assert name not in seen, f"{name} is defined in {seen.get(name)} and {path}"
            seen[name] = path


def test_lazy_table_serves_only_sparta():
    assert all(p == "sparta" or p.startswith("sparta.") for p in sparta._LAZY_EXPORTS.values())
    for package in ("unicor", "todcor"):
        reexported = set(importlib.import_module(package).__all__) & set(sparta._LAZY_EXPORTS)
        assert not reexported, f"sparta re-exports {package} names: {sorted(reexported)}"


def test_all_and_dir_are_the_lazy_table():
    assert sparta.__all__ == list(sparta._LAZY_EXPORTS)
    assert dir(sparta) == sorted(sparta._LAZY_EXPORTS)


@pytest.mark.parametrize("name", list(sparta._LAZY_EXPORTS))
def test_from_sparta_import_is_the_module_object(name):
    expected = _resolve(name, sparta._LAZY_EXPORTS[name])
    namespace = {}
    exec(f"from sparta import {name}", namespace)
    assert namespace[name] is expected
    assert getattr(sparta, name) is expected


def test_plain_import_still_reaches_every_public_submodule():
    # Before the front door, ``import sparta`` imported everything, so
    # ``sparta.pdc.pdc_power`` worked after a bare ``import sparta``; it still
    # does, lazily, in a fresh interpreter.
    statement = ("import sparta; "
                 "assert sparta.pdc.pdc_power is __import__('sparta.pdc', fromlist=['x']).pdc_power; "
                 "assert sparta.spectrum.Spectrum is sparta.Spectrum; "
                 "assert sparta.tiravel_ops.build_tiravel is sparta.build_tiravel; "
                 "assert sparta.instruments.simulated.Simulation is not None")
    env = dict(os.environ, PYTHONPATH=_PATH + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run([sys.executable, "-c", statement], capture_output=True, text=True,
                          env=env, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError, match="no attribute 'lag_axis'"):
        sparta.lag_axis
    assert not hasattr(sparta, "no_such.module")      # not a ModuleNotFoundError
    with pytest.raises(ImportError):
        exec("from sparta import no_such_atom", {})


# =============================================================================
# Import cost
# =============================================================================
_COST = r"""
import json, sys, time
start = time.perf_counter()
exec(sys.argv[1])
print(json.dumps({"seconds": time.perf_counter() - start, "modules": len(sys.modules),
                  "heavy": sorted(m for m in json.loads(sys.argv[2]) if m in sys.modules)}))
"""

IMPORTS = [
    "import sparta",
    "from sparta import resample_to_log_grid, pdc_power, build_tiravel",
    "from sparta import " + ", ".join(
        n for n, p in sparta._LAZY_EXPORTS.items() if p not in ("sparta.instruments",
                                                               "sparta.plotting")),
]


@pytest.mark.parametrize("statement", IMPORTS, ids=["sparta", "atoms", "objects"])
def test_import_does_not_load_the_plotting_or_specutils_stack(statement):
    env = dict(os.environ, PYTHONPATH=_PATH + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run([sys.executable, "-c", _COST, statement, json.dumps(HEAVY)],
                          capture_output=True, text=True, env=env, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    cost = json.loads(proc.stdout.strip().splitlines()[-1])
    print(f"{statement[:60]}: {cost['seconds']:.3f} s, {cost['modules']} modules")
    assert cost["heavy"] == [], f"{statement!r} imported {cost['heavy']}"
