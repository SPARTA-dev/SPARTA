# SPARTA examples

Executed notebooks that run offline in seconds (synthetic data have fixed
seeds; PHOENIX templates come from the bundled cache, never the network). They
come in two kinds, next to the reference pages in [`docs/`](../docs/index.md):

| | Where | What for |
| --- | --- | --- |
| **How-tos** | [`howto/`](howto/) | Getting a task done with the objects: short recipes, minimal prose. Start here. |
| **Tutorials** | [`tutorials/`](tutorials/) | How each method works and why: built step by step from the atoms (plain arrays in, arrays out), with every claim checked by a printed number. |
| **Reference** | [`docs/`](../docs/index.md) | The conventions, formulas, result keys and known limitations, stated once. |

Which notebook covers what:

| Task | Atoms | Object | How-to | Tutorial |
| --- | --- | --- | --- | --- |
| read archive spectra | `sparta.instruments` loaders | `Observations` | 01 | 01 |
| simulate spectra | `sparta.instruments.Simulation` | `Observations` | 05 | 03–06, 09 |
| prepare a spectrum | `sparta.preprocess` | `Spectrum` | 01 | 02, 03 |
| RV of a single-lined star (CCF) | `unicor` | `CCF1d` | 01 | 03 |
| RVs of both stars of a double-lined binary | `todcor` | `CCF2d` | 02 | 04 |
| a template built from the observations | `sparta.tiravel_ops` | `TIRAVEL` | 03 | 05 |
| periodograms: GLS, PDC, USuRPER, partial | `sparta.pdc` | `PeriodicityDetector` | 04 | 06–08 |

## How-tos

[`howto/`](howto/): one task each, in about ten cells, with the objects
(`Observations`, `Template`, `CCF1d`, `CCF2d`, `TIRAVEL`,
`PeriodicityDetector`) and their plot methods.

| Notebook | What it does |
| --- | --- |
| [`01_measure_rvs`](howto/01_measure_rvs.ipynb) | RVs of TOI-677 (NRES): recommended orders and preprocessing, a broadened PHOENIX template, `calc_rv_against_template`, the RV curve, `CCF1d.plot`, `output='loglike'`. |
| [`02_double_lined_binary`](howto/02_double_lined_binary.ipynb) | A simulated double-lined binary: `calc_rv_against_two_templates`, the R and ln L maps (`CCF2d.plot`), velocities and amplitudes against `obs.truth`. |
| [`03_template_from_data`](howto/03_template_from_data.ipynb) | HARPS TOI-193: RVs against an air-frame PHOENIX template, `build_tiravel_template` (ρ, the template), RVs against the TIRAVEL template. |
| [`04_find_periods`](howto/04_find_periods.ipynb) | `PeriodicityDetector`: GLS, PDC and error-aware PDC with p-value levels on an RV series; USuRPER and the partial shift/shape periodograms on spectra; `plot_periodograms`. |
| [`05_simulate_data`](howto/05_simulate_data.ipynb) | `instruments.Simulation` like HARPS and like NRES: seeds, noise models, `as_product`, barycentric correction, `obs.truth`; the instrument registry. |

[`howto/todcor_detailed/`](howto/todcor_detailed/README.md): the standalone
`todcor` engine on plain arrays, for using TODCOR outside SPARTA. Longer than
the other how-tos: they derive the fit as they go (`01_method`,
`02_single_window_fit`, `03_multiwindow_combination`).

## Tutorials

[`tutorials/`](tutorials/): every step is a visible call to an atom from
`sparta` (preprocessing, PDC, TIRAVEL), `unicor` or `todcor`;
[`docs/open_api.md`](../docs/open_api.md) lists them with their units. The
objects appear only in each notebook's last cell, which checks that they give
the same numbers and points to the matching how-to.

| Notebook | What it covers |
| --- | --- |
| [`01_reading_spectra`](tutorials/01_reading_spectra.ipynb) | The bundled NRES and HARPS data loaded as arrays with the `sparta.instruments` loaders; what one epoch holds (orders, time, barycentric correction, native step); units. |
| [`02_log_lambda`](tutorials/02_log_lambda.ipynb) | Why a Doppler shift is a translation on a log-λ grid; `unicor.log_lambda_grid` / `log_grid_step` and `resample_to_log_grid`; native vs resampled pixels (`coarsest_native_step`, `effective_bins`); air vs vacuum with `phoenix_spectrum`. |
| [`03_unicor`](tutorials/03_unicor.ipynb) | The CCF by hand, then `unicor.ccf_order` / `measure_rv`; RV and its uncertainty; combining orders (Zucker 2003) and the log-likelihood; error bars checked on a simulated star; the RVs of TOI-677 (NRES) rebuilt from atoms only; `velocity_convention`. |
| [`04_todcor`](tutorials/04_todcor.ipynb) | TODCOR from the `todcor` atoms (`prepare_window`, `window_stats`, `joint_fit`, `window_noise`) on a simulated two-order double-lined binary: the R and ln L maps, amplitudes, Cramér–Rao errors, pooling orders, a whole orbit, and the window/taper/continuum choices. |
| [`05_tiravel`](tutorials/05_tiravel.ipynb) | TIRAVEL step by step (`common_grid`, `align_spectra`, `normalize_rows`, `correlation_matrix`, `alignment`, `coadd`, then `build_tiravel`): the alignment measure ρ on simulated data, then the HARPS spectra of TOI-193 with an air-frame PHOENIX template. |
| [`06_pdc`](tutorials/06_pdc.ipynb) | GLS and PDC of a scalar series from the `sparta.pdc` atoms: distance matrices, U-centring, the phase distances; local p-values with `pdc_pvalue` / `pdc_power_level`; the error-aware PDC for very unequal error bars. |
| [`07_usurper`](tutorials/07_usurper.ipynb) | USuRPER without RVs or a template: spectra as points on a unit sphere; a moving star, a temperature change with no motion, and an unresolved twin binary found at half its period. |
| [`08_partial_pdc`](tutorials/08_partial_pdc.ipynb) | Partial distance correlation: separating Doppler shifts from shape changes; a toy on three variables; a planet around an active star split by the shift and shape periodograms; their limits and local p-values. |
| [`09_end_to_end`](tutorials/09_end_to_end.ipynb) | A simulated season of HARPS-like spectra → RVs (`ccf_order`) → periodogram → orbit fit, with the 1-day alias. |

## Data

- `data/TOI677/`: 12 NRES spectra (67 echelle orders each) of TOI-677.
- `data/TOI193/`: 32 HARPS archive 1-D spectra of TOI-193 (LTT 9779).

## Running

With SPARTA installed ([README](../README.md#installation)), add Jupyter and
open this folder:

```bash
pip install jupyterlab
jupyter lab examples
```

Each notebook runs from its own folder and reads the data from `../data/`.
To execute one from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace examples/howto/01_measure_rvs.ipynb
```

`pytest -m slow tests/test_notebooks.py` (from the repository root) executes
every notebook here as a smoke test.
