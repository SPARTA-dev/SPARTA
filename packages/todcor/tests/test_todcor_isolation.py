"""Dependency isolation: `todcor` depends on numpy + unicor only (scipy via unicor).

The point of the standalone package is that it installs and imports without
SPARTA's heavier stack. Two independent checks:

(a) Import every ``todcor`` submodule in a FRESH subprocess and assert that no
    forbidden top-level module ends up in ``sys.modules`` (a subprocess, so a
    sibling test that imported astropy cannot pollute the check).
(b) AST-scan every file under ``src/todcor/`` for forbidden imports, so a lazy
    import inside a function cannot slip past (a).
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PKGS = Path(__file__).resolve().parents[2]
_SRC = _PKGS / "todcor" / "src" / "todcor"
# The fresh interpreter must find the checkout whether or not it is installed.
_PATH = os.pathsep.join(str(p) for p in (_SRC.parent, _PKGS / "unicor" / "src"))
_SUBMODULES = ["estimator", "simulate"]
FORBIDDEN = ("sparta", "astropy", "specutils", "pandas", "matplotlib")


def test_no_forbidden_modules_imported_at_runtime():
    imports = "; ".join(f"import todcor.{m}" for m in _SUBMODULES)
    code = (
        f"import sys; import todcor; {imports}; "
        f"bad = sorted({{k for k in sys.modules if k.split('.')[0] in {FORBIDDEN!r}}}); "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    env = dict(os.environ, PYTHONPATH=_PATH + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, (
        f"forbidden modules imported by todcor: {proc.stdout.strip()} {proc.stderr.strip()}"
    )


@pytest.mark.parametrize("path", sorted(_SRC.rglob("*.py")), ids=lambda p: p.name)
def test_no_forbidden_imports_in_source(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if a.name.split(".")[0] in FORBIDDEN]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            if (node.module or "").split(".")[0] in FORBIDDEN:
                offenders.append(node.module)
    assert not offenders, f"{path.name} imports forbidden modules: {offenders}"
