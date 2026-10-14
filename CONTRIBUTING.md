# Contributing to SPARTA

Thank you for helping. This page covers the layout, the development install,
the tests, and the rules the code base keeps.

## Layout

The repository holds three installable packages:

| Path | Package (import) | What it is |
| --- | --- | --- |
| `packages/unicor/` | `sparta-unicor` (`unicor`) | the 1-D cross-correlation engine and the shared log-λ DSP layer (`unicor.dsp`) |
| `packages/todcor/` | `sparta-todcor` (`todcor`) | the 2-D (SB2) engine, on top of `unicor` |
| `src/sparta/` | `SPARTA` (`sparta`) | the object layer: `Spectrum`, `Template`, `CCF1d`, `CCF2d`, `TIRAVEL`, `Observations`, `TimeSeries`, `PeriodicityDetector`, `sparta.pdc`, `sparta.instruments` |

Each engine has its own `README.md`, `pyproject.toml` and `tests/`. SPARTA's
tests are in `tests/` (with `tests/instruments/` for the loaders and
`tests/golden/` for the pinned reference values); `docs/instruments.md` and
`docs/api.md` are generated; the tutorials are in `examples/`.

## Development install

From the repository root, in a fresh environment (Python ≥ 3.10):

```bash
pip install -r requirements.txt
```

This installs all three packages in editable mode, with their test extras, in
one pip call so the engines resolve from the checkout. Each package declares
its own runtime dependencies in its `pyproject.toml`; do not add a dependency
without discussing it first.

## Running the tests

```bash
pytest                       # all three packages, from the repository root
pytest packages/unicor       # one engine
pytest tests/test_ccf1d.py   # one file
pytest -m slow               # the slow tests: executing the example notebooks
```

The root configuration puts `src/` and both `packages/*/src/` on the path, so
the tests always run against this checkout, whatever is installed. The engines'
suites also run standalone from their own directories
(`cd packages/todcor && pytest`).

Every change must leave all suites green. A failing test is diagnosed, not
loosened: do not widen a tolerance or skip a test to get green.

## Engine isolation

`unicor` depends on numpy and scipy only; `todcor` on numpy and `unicor` only;
`sparta.tiravel_ops` on numpy and `unicor` only. None of them may import
`sparta`, astropy, specutils, pandas or matplotlib — not even lazily inside a
function. The isolation tests (`packages/*/tests/test_*_isolation.py`,
`tests/test_tiravel_ops_isolation.py`) check this twice: by importing the
modules in a fresh subprocess, and by scanning their source for imports.
Array-level numerics belong in the engines; `Spectrum` / `Template` / units
handling belongs in `sparta`.

## The golden master

Two test files pin the numbers the user-facing workflows produce:

- `tests/test_golden_pipeline.py` → `tests/golden/pipeline.json`: UNICOR and
  TODCOR RVs on seeded simulations, TIRAVEL on the bundled HARPS spectra, and
  the GLS, PDC, error-aware PDC, USuRPER and partial periodograms; compared at
  1e-9 relative.
- `tests/test_golden_instruments.py` → `tests/golden/instruments.json`: what
  the HARPS and NRES loaders and `Observations` produce from the bundled
  files, pinned bit for bit (spectra by hash).

A refactor must reproduce them unchanged. A change that alters results on
purpose (a bug fix, a better estimator) regenerates the reference
deliberately:

```bash
python tests/test_golden_pipeline.py --write
```

(the instrument reference is rebuilt with the `_record_*` helpers in its test
file). Commit the regenerated JSON with the code change, and say in the commit
message which numbers moved, by how much, and why. Never regenerate to make an
unexplained difference go away.

The bisector span is deliberately not pinned: its formula is known to be
incorrect and is under review.

## Adding an instrument

1. Write `src/sparta/instruments/<name>.py`: an `Instrument` (resolving power
   and native step, with `references` and `notes` saying where they come
   from), one loader per archive product returning a `LoadedSpectrum`
   (wavelengths in Å, the time and its system, `bcv` in km/s, SPECSYS and the
   air/vacuum frame), and a `PRODUCTS` tuple of `Product`s. State what is
   documented and mark what is not (`UNVERIFIED:` in a comment, `None` for an
   unknown frame). Shared helpers live in `_common.py`, `_eso.py`, `_iraf.py`
   and `_ceres.py`.
2. Register the module in `MODULES` in `src/sparta/instruments/__init__.py`.
3. Add `tests/instruments/test_<name>.py`, building synthetic FITS files with
   `tests/instruments/fits_builders.py`: the wavelength solution, the frame,
   the time, the barycentric correction, and the failure modes.
4. Keep `verified=False` until the loader has been checked against real
   archive files; an unverified product warns on every load. Set
   `verified=True` only after that check, and, where the files can be
   bundled, pin them in the golden master.
5. Regenerate the reference page (a test fails if it is stale):

   ```bash
   PYTHONPATH=src:packages/unicor/src:packages/todcor/src python -m sparta.instruments > docs/instruments.md
   ```

## The API reference

`docs/api.md` is rendered from the docstrings of the public API of `sparta`,
`unicor` and `todcor`. After changing a public signature or docstring,
regenerate it (a test fails if it is stale):

```bash
python docs/_generate_api.py --write
```

## Code style

- **Docstrings in numpydoc form** on every public function, class and method:
  a one-line summary, then `Parameters`, `Returns` (and `Raises` where it
  raises), with the **units stated** for every physical quantity.
- **Units at the boundaries.** Velocities in km/s, wavelengths in Å, times in
  days, frequencies in 1/day. Public functions accept plain numbers in these
  units (and astropy Quantities where they already do); convert once at the
  boundary and keep plain floats inside, never `Quantity` in an inner loop.
  Say which frame (air or vacuum) a wavelength array is in.
- **Vectorize** the inner kernels with numpy broadcasting where that stays
  readable; a clear loop beats a dense one-liner elsewhere.
- **No silent failures.** No bare `except:`; validate inputs with real
  exceptions (`ValueError`, `TypeError`), not `assert`. A result that cannot
  be computed is NaN with a warning, or an error, never a plausible number.
- **No shims.** When an API changes, change every caller and record the old →
  new name in `CHANGELOG.md`; no aliases, no deprecated wrappers, no dead
  code.
- **No network by default.** Anything that downloads must be opt-in
  (`allow_download=True`).
- **Comments say why**, not what.
