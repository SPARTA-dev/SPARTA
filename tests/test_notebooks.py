"""Smoke test: every notebook under examples/ executes without an error.

Each notebook re-runs its analyses, so the test is marked ``slow`` and the
default ``pytest`` run deselects it; run it with ``pytest -m slow``. Every
notebook runs in its own folder (the data paths are relative to it), on a fresh
kernel whose PYTHONPATH puts this checkout's sources first, whatever is
installed in the environment. The notebooks are not modified.
"""
import os
import pathlib
import time

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
SOURCES = (ROOT / "src", ROOT / "packages" / "unicor" / "src", ROOT / "packages" / "todcor" / "src")
NOTEBOOKS = sorted(p for p in EXAMPLES.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts)
# Seconds. The examples are meant to run well under two minutes each; a hung or
# runaway cell fails at this limit (nbclient's timeout is per cell), and a
# notebook that exceeds it overall fails the runtime check.
BUDGET_S = 120


def _kernel_env():
    env = dict(os.environ)
    paths = [str(p) for p in SOURCES]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def test_examples_are_found():
    assert len(NOTEBOOKS) >= 10, [str(p) for p in NOTEBOOKS]


@pytest.mark.slow
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: str(p.relative_to(EXAMPLES)))
def test_notebook_executes(path):
    notebook = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(notebook, timeout=BUDGET_S,
                                     resources={"metadata": {"path": str(path.parent)}})
    start = time.perf_counter()
    client.execute(env=_kernel_env())
    elapsed = time.perf_counter() - start
    assert elapsed < BUDGET_S, f"{path.name} took {elapsed:.0f} s (budget {BUDGET_S} s)"
