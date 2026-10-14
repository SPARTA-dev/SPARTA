# TODCOR deep dives

Three executed notebooks on TODCOR, the two-component (double-lined, SB2)
velocity engine, the counterpart of UNICOR. They build the method up on plain
arrays using only the standalone [`todcor`](../../packages/todcor) package.
Everything is synthetic, so each notebook is self-contained (no external data),
and every quantitative claim is a live numerical check.

For TODCOR on SPARTA objects (`CCF2d` with `Spectrum` and `Template`, and a
whole SB2 orbit through `Observations.calc_rv_against_two_templates`), start
with [`../03_double_lined_todcor.ipynb`](../03_double_lined_todcor.ipynb).

The canonical written spec is [`todcor_method.md`](../../packages/todcor/docs/todcor_method.md).

1. **`01_method.ipynb`**: the method derived in the open: the log-λ shift
   theorem, the six **sufficient statistics** and the χ² quadratic form, the
   closed-form 2×2 weighted-least-squares amplitude fit, profile vs marginal
   likelihood, the Parseval identity, and why the even-N Nyquist mode is dropped.
2. **`02_single_window_fit.ipynb`**: one window end-to-end: `composite_flux` →
   `todcor_map` → read the peak → `model_flux` reconstruction → `crlb_from_map`
   errors, with the CRLB validated against a Monte Carlo.
3. **`03_multiwindow_combination.ipynb`**: the sufficient-statistic **addition**:
   pooling windows of one epoch via `combine_stats` (shared velocities /
   amplitudes) tightens and de-correlates the joint fit, and the boundary
   (within one epoch, shared velocity grid, **not** across epochs).

## Running

The notebooks need only `todcor` (plus matplotlib). From the repository root,
`pip install -r requirements.txt` installs it with SPARTA (editable). Then, from
this directory:

```bash
jupyter nbconvert --to notebook --execute --inplace 01_method.ipynb
```

## Conventions

- Velocities and the log-λ shift `s` in km/s; ζ in s/km; wavelengths in Angstrom;
  kernel widths as FWHM.
- Uniform log-λ (constant velocity step `dv`) grid; `np.fft.rfftfreq(N, d=dv)[1:]`
  (DC bin dropped, which also removes the continuum), positive frequencies only.
- Build-by-hand first, library call second, printed agreement check third; fixed
  seeds throughout.

## Reference

Zucker, S. & Mazeh, T. 1994, *ApJ*, 420, 806 — "Study of spectroscopic binaries
with TODCOR. I. A new two-dimensional correlation algorithm."
