"""Dependency isolation: `unicor` depends on numpy + scipy only.

The point of the standalone package is that it installs and imports without
SPARTA's heavier stack. Two independent checks:

(a) Import every ``unicor`` submodule in a FRESH subprocess and assert that no
    forbidden top-level module ends up in ``sys.modules`` (a subprocess, so a
    sibling test that imported astropy cannot pollute the check).
(b) AST-scan every file under ``src/unicor/`` for forbidden imports, so a lazy
    import inside a function cannot slip past (a).
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "unicor"
_SUBMODULES = ["dsp", "ccf"]
FORBIDDEN = ("sparta", "astropy", "specutils", "pandas", "matplotlib")


def test_no_forbidden_modules_imported_at_runtime():
    imports = "; ".join(f"import unicor.{m}" for m in _SUBMODULES)
    code = (
        f"import sys; import unicor; {imports}; "
        f"bad = sorted({{k for k in sys.modules if k.split('.')[0] in {FORBIDDEN!r}}}); "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"forbidden modules imported by unicor: {proc.stdout.strip()} {proc.stderr.strip()}"
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
