"""The object layer, byte for byte (see tests/_atoms_baseline.py).

Exact comparison, no tolerance: each case is the SHA-256 of (shape, float64
bytes) of an object output on a seeded input. Moving a step into a function
is a text operation and must leave every hash in place; a deliberate numeric
change re-blesses in its own commit (``python tests/_atoms_baseline.py
--write``) and says what moved.

The hashes hold only on the environment that wrote them, so the module skips
when the running Python, numpy, scipy, astropy or specutils differs from the
recorded one. The 1e-9 goldens (test_golden_pipeline.py,
test_golden_instruments.py) remain the cross-platform anchor.
"""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _atoms_baseline import BASELINE_PATH, compute, environment  # noqa: E402

with open(BASELINE_PATH) as _f:
    BASELINE = json.load(_f)

if BASELINE["versions"] != environment():
    pytest.skip(f"byte-identity baseline recorded on {BASELINE['versions']}, running on "
                f"{environment()}; the 1e-9 goldens still apply. Re-bless on purpose with "
                "`python tests/_atoms_baseline.py --write`.", allow_module_level=True)


@pytest.fixture(scope="module")
def computed():
    return compute()


@pytest.mark.parametrize("case", sorted(BASELINE["cases"]))
def test_object_output_is_bit_identical_to_the_baseline(case, computed):
    got, want = computed[case], BASELINE["cases"][case]
    assert got["sha256"] == want["sha256"], (
        f"{case}: bytes changed (sum {want['sum']!r} -> {got['sum']!r}). A move is a text "
        "operation; a deliberate numeric change re-blesses in its own commit.")


def test_every_computed_case_is_baselined(computed):
    assert sorted(computed) == sorted(BASELINE["cases"])
