"""Dependency isolation: ``sparta.tiravel_ops`` needs numpy + unicor only.

The TIRAVEL measure is array-level, like the standalone ``unicor`` and
``todcor`` packages (see ``packages/*/tests/test_*_isolation.py``), so it can
move into one of them unchanged. Two independent checks:

(a) Load the module file in a FRESH subprocess that has only ``unicor`` on its
    path -- not through ``sparta/__init__``, which imports the whole stack --
    and assert that no forbidden top-level module ends up in ``sys.modules``.
(b) AST-scan the file: every import must be numpy, unicor or the stdlib
    ``dataclasses`` (for ``TiravelResult``, as in ``unicor.ccf``), so a lazy
    import inside a function cannot slip past (a).
"""
import ast
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
_FILE = ROOT / "src" / "sparta" / "tiravel_ops.py"
_UNICOR = ROOT / "packages" / "unicor" / "src"
ALLOWED = ("numpy", "unicor", "dataclasses")
FORBIDDEN = ("sparta", "astropy", "specutils", "pandas", "matplotlib", "todcor")


def test_no_forbidden_modules_imported_at_runtime():
    code = (
        "import importlib.util, sys; "
        f"spec = importlib.util.spec_from_file_location('tiravel_ops', {str(_FILE)!r}); "
        "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); "
        f"bad = sorted({{k for k in sys.modules if k.split('.')[0] in {FORBIDDEN!r}}}); "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    env = dict(os.environ, PYTHONPATH=str(_UNICOR))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, cwd=_UNICOR)
    assert proc.returncode == 0, (
        f"forbidden modules imported by tiravel_ops: {proc.stdout.strip()} {proc.stderr.strip()}"
    )


def test_only_numpy_and_unicor_are_imported_in_source():
    tree = ast.parse(_FILE.read_text(), filename=str(_FILE))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "tiravel_ops must not import relatively from sparta"
            imported.append(node.module or "")
    offenders = [m for m in imported if m.split(".")[0] not in ALLOWED]
    assert not offenders, f"tiravel_ops imports beyond numpy + unicor: {offenders}"
