# SPARTA examples

Short, executed notebooks that teach SPARTA step by step. Each opens with
"What you'll learn", explains *why* before the code, checks its claims with
printed numbers, and runs offline in seconds (synthetic data have fixed seeds;
PHOENIX templates come from the bundled cache, never the network).

## Quickstarts

[`quickstart/`](quickstart/): one capability each, in about ten cells, with the
high-level objects (`Observations`, `Template`, `CCF1d`, `CCF2d`, `TIRAVEL`,
`PeriodicityDetector`) and their plot methods. Minimal prose; the series below
explains the why.

| Notebook | What it does |
|---|---|
| [`q01_rvs_nres`](quickstart/q01_rvs_nres.ipynb) | RVs of TOI-677 (NRES): recommended orders and preprocessing, a broadened PHOENIX template, `calc_rv_against_template`, the RV curve, `CCF1d.plot`, `output='loglike'`. |
| [`q02_sb2_todcor`](quickstart/q02_sb2_todcor.ipynb) | A simulated double-lined binary: `calc_rv_against_two_templates`, the R and ln L maps (`CCF2d.plot`), velocities and amplitudes against `obs.truth`. |
| [`q03_tiravel_harps`](quickstart/q03_tiravel_harps.ipynb) | HARPS TOI-193: RVs against an air-frame PHOENIX template, `build_tiravel_template` (ρ, the template), RVs against the TIRAVEL template. |
| [`q04_periodograms`](quickstart/q04_periodograms.ipynb) | `PeriodicityDetector`: GLS, PDC and error-aware PDC with p-value levels on an RV series; USuRPER and the partial shift/shape periodograms on spectra; `plot_periodograms`. |
| [`q05_simulated_data`](quickstart/q05_simulated_data.ipynb) | `instruments.Simulation` like HARPS and like NRES: seeds, noise models, `as_product`, barycentric correction, `obs.truth`; the instrument registry. |

## The series

The series is *open*: every step is a visible call to an atom — plain arrays
in, arrays or a small result out — from `sparta` (preprocessing, PDC,
TIRAVEL), `unicor` or `todcor`; [`docs/open_api.md`](../docs/open_api.md)
lists them with their units. The objects appear only in each notebook's last
cell, which checks that they give the same numbers and points to the matching
quickstart.

| Notebook | What it covers |
|---|---|
| [`00_getting_started`](00_getting_started.ipynb) | What SPARTA does; the bundled NRES and HARPS data loaded as arrays with the `sparta.instruments` loaders; what one epoch holds (orders, time, barycentric correction, native step); units. |
| [`01_spectra_and_log_lambda`](01_spectra_and_log_lambda.ipynb) | Why a Doppler shift is a translation on a log-λ grid; `unicor.log_lambda_grid` / `log_grid_step` and `resample_to_log_grid`; native vs resampled pixels (`coarsest_native_step`, `effective_bins`); air vs vacuum with `phoenix_spectrum`. |
| [`02_radial_velocity_unicor`](02_radial_velocity_unicor.ipynb) | The CCF by hand, then `unicor.ccf_order` / `measure_rv`; RV and its uncertainty; combining orders (Zucker 2003) and the log-likelihood; error bars checked on a simulated star; the RVs of TOI-677 (NRES) rebuilt from atoms only, equal to `Observations`; `velocity_convention`. |
| [`03_double_lined_todcor`](03_double_lined_todcor.ipynb) | TODCOR from the `todcor` atoms (`prepare_window`, `window_stats`, `joint_fit`, `window_noise`) on a simulated two-order double-lined binary: the R and ln L maps, amplitudes, Cramér–Rao errors, pooling orders, a whole orbit, and the window/taper/continuum choices; equal to `CCF2d` / `Observations`. |
| [`04_template_from_data_tiravel`](04_template_from_data_tiravel.ipynb) | TIRAVEL step by step (`common_grid`, `align_spectra`, `normalize_rows`, `correlation_matrix`, `alignment`, `coadd`, then `build_tiravel`): the alignment measure ρ on simulated data, then the HARPS spectra of TOI-193 with an air-frame PHOENIX template. |
| [`05_finding_periods`](05_finding_periods.ipynb) | GLS, PDC (with and without error bars), USuRPER and the partial (shift/shape) periodograms from the `sparta.pdc` atoms; local p-values with `pdc_pvalue` / `pdc_power_level`. |
| [`06_end_to_end`](06_end_to_end.ipynb) | A simulated season of HARPS-like spectra → RVs (`ccf_order`) → periodogram → orbit fit, with the 1-day alias. |
| [`07_usurper_concept`](07_usurper_concept.ipynb) | USuRPER without RVs or a template: spectra as points on a unit sphere; a moving star, a temperature change with no motion, and an unresolved twin binary found at half its period. |
| [`08_partial_pdc_concept`](08_partial_pdc_concept.ipynb) | Partial distance correlation: separating Doppler shifts from shape changes; a toy on three variables; a planet around an active star split by the shift and shape periodograms; their limits and local p-values. |

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

Then start Jupyter in this folder (the notebooks read `data/` relative to it;
the quickstarts, `../data/`):

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
