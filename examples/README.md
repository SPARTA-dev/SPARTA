# SPARTA examples

Short, executed notebooks that teach SPARTA step by step. Each opens with
"What you'll learn", explains *why* before the code, checks its claims with
printed numbers, and runs offline in seconds (synthetic data have fixed seeds;
PHOENIX templates come from the bundled cache, never the network).

## The series

| Notebook | What it covers |
|---|---|
| [`00_getting_started`](00_getting_started.ipynb) | What SPARTA does; loading the bundled NRES and HARPS data with `Observations`; what a `Spectrum` holds; what `sparta.instruments` records per spectrograph; units. |
| [`01_spectra_and_log_lambda`](01_spectra_and_log_lambda.ipynb) | Why a Doppler shift is a translation on a log-λ grid; `unicor.log_lambda_grid` and `Spectrum.interpolate`; native vs resampled pixels (`native_steps`); air vs vacuum. |
| [`02_radial_velocity_unicor`](02_radial_velocity_unicor.ipynb) | The CCF by hand and with `unicor`; RV and its uncertainty; combining orders (Zucker 2003); error bars checked on a simulated star; RVs of TOI-677 (NRES) with a PHOENIX template; `velocity_convention`. |
| [`03_double_lined_todcor`](03_double_lined_todcor.ipynb) | TODCOR (`CCF2d`) on a simulated double-lined binary: the (v1, v2) map, amplitudes, errors, a whole orbit, and the window/taper/continuum choices. |
| [`04_template_from_data_tiravel`](04_template_from_data_tiravel.ipynb) | TIRAVEL: a template built from the observations; the alignment measure ρ; simulated data, then HARPS spectra of TOI-193. |
| [`05_finding_periods`](05_finding_periods.ipynb) | GLS, PDC (with and without error bars), USuRPER and the partial (shift/shape) periodograms; local p-values. |
| [`06_end_to_end`](06_end_to_end.ipynb) | A simulated season of HARPS-like spectra → RVs → periodogram → orbit fit, with the 1-day alias. |
| [`07_usurper_concept`](07_usurper_concept.ipynb) | USuRPER without RVs or a template: spectra as points on a unit sphere; a moving star, a temperature change with no motion, and an unresolved twin binary found at half its period. |

## Deep dives

[`todcor/`](todcor/): the mathematics of TODCOR on plain arrays with the
standalone `todcor` package (`01_method`, `02_single_window_fit`,
`03_multiwindow_combination`). See [`todcor/README.md`](todcor/README.md).

## Data

- `data/TOI677/`: 12 NRES spectra (67 echelle orders each) of TOI-677.
- `data/TOI193/`: 32 HARPS archive 1-D spectra of TOI-193 (LTT 9779).

## Running

From the repository root, install SPARTA and its two engines (editable), plus
Jupyter:

```bash
pip install -r requirements.txt
pip install jupyterlab
```

Then start Jupyter in this folder (the notebooks read `data/` relative to it):

```bash
cd examples
jupyter lab
```

or execute one from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace 00_getting_started.ipynb
```

`pytest -m slow tests/test_notebooks.py` (from the repository root) executes
every notebook here as a smoke test.
