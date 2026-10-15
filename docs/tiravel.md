# TIRAVEL: a template from the observations

TIRAVEL (Zucker & Mazeh 2006) builds a template out of the spectra
themselves, for stars no library model fits well (peculiar, active, fast
rotators). This page is the reference for what SPARTA implements, how to read
its outputs, and what is left to build.
[Tutorial 05](../examples/tutorials/05_tiravel.ipynb) builds it step by step;
[how-to 03](../examples/howto/03_template_from_data.ipynb) is the short
recipe.

## Status

| | |
| --- | --- |
| **Implemented** | The fixed-velocity step: given one trial velocity per epoch, the alignment measure ρ, the co-addition weights and the template. Multi-order epochs, pooled into one correlation matrix. |
| **Not implemented** | The optimisation of ρ over the velocities, which is the rest of the method: SPARTA does not *measure* velocities with TIRAVEL (see [Development](#development)). No uncertainties on TIRAVEL velocities. No relativistic velocity convention ('cz' only). |
| **Verified** | Against Zucker & Mazeh (2006) in the 2026-07-23 rewrite and the 2026-10-14 scientific review: the exact de-shift, ρ, the principal eigenpair (`eigh`), unsquared weights. Each defect the rewrite fixed has a regression test (e.g. `test_alignment_picks_the_principal_eigenpair`, `test_weights_favour_the_higher_snr_epochs`, `test_rho_is_invariant_to_a_global_velocity_offset`). |
| **Tests** | `tests/test_tiravel.py`, `tests/test_atoms_tiravel.py` (atoms = class, bit for bit), `tests/test_tiravel_ops_isolation.py`; golden values in `tests/test_golden_pipeline.py`. |

The pieces:

| Layer | Name | Does |
| --- | --- | --- |
| atoms (`sparta.tiravel_ops`) | `rest_frame_wavelengths`, `common_grid`, `align_spectra`, `normalize_rows`, `correlation_matrix`, `alignment`, `coadd` | one step each, on plain arrays ([open_api.md](open_api.md#tiravel--a-template-from-the-observations-spartatiravel_ops)) |
| | `build_tiravel` → `TiravelResult` | the whole fixed-velocity step |
| object (`sparta.tiravel`) | `TIRAVEL.build_template` | the same on a list of `Spectrum`; returns a `Template` |
| driver | `Observations.build_tiravel_template(velocities)` | the same on every epoch; sets `obs.tiravel` and `obs.tiravel_template` |

## Basics

### The measure

For K epochs at trial velocities v_k: shift each spectrum to rest,
λ / (1 + v_k/c), and resample all onto one common grid; subtract each order's
mean per epoch and scale each epoch to unit length over all orders; form the
K × K correlation matrix R (unit diagonal). With λ_max its largest
eigenvalue,

$$\rho = \frac{\lambda_{\max} - 1}{K - 1},$$

from 0 (the spectra share nothing) to 1 (identical up to scale). The
eigenvector of λ_max holds the weights, and the template is the weighted sum
of the normalised epochs. It is mean-free with an arbitrary scale, to which
a correlation is blind.

### What ρ says, and what it does not

- **Noise keeps ρ below 1.** At the right velocities each epoch is the common
  spectrum s plus its own noise σ, so
  1 − ρ ≈ σ²/var(s) ∝ (S/N)⁻². In tutorial 05, (1 − ρ)(S/N)² stays at
  16.4–16.9 for S/N 25, 50 and 100 (ρ = 0.974, 0.993, 0.998). The absolute ρ
  therefore says as much about the S/N as about the velocities.
- **ρ is largest at the right velocities.** Scaling the true velocity curve by
  s, ρ was 0.434 at s = 0 and peaked at s = 1 (tutorial 05). That is what an
  optimiser would use.
- **Only differences matter.** Adding a constant to every velocity leaves R
  and ρ unchanged and shifts the template by that constant, so RVs measured
  against it move by it too (+5.000 km/s for +5 km/s, tutorial 05). The
  template's rest frame is the zero point of the velocities you supply.

### The weights

- **Equal S/N gives equal weights**, 1/√K (0.1825–0.1827 for K = 30).
- **`n_eff`**, the participation ratio 1/Σw⁴, is roughly the number of epochs
  that contribute. The template's noise is about 1/√n_eff of one epoch's:
  5.38 measured against √n_eff = 5.48 in tutorial 05.
- **A negative or near-zero weight is a warning.** That epoch anti-correlates
  with, or is unrelated to, the rest: a bad spectrum or a wrong trial velocity.
  `TIRAVEL.plot_weights` shows them per epoch.
- **Maximum likelihood holds strictly** for a rank-1 model with equal,
  independent noise on the normalised rows. Unit-normalising epochs of
  different S/N only approximates that.

### Grids, orders and frames

- The common grid is the reference epoch's rest-frame grid (epoch 0 by
  default, so it is not resampled), clipped to the range every de-shifted
  epoch covers, less `margin` pixels (10) at each end. Epochs that do not
  overlap raise.
- The shifted epochs are resampled with cubic splines. Linear interpolation
  smooths the lines enough to lower the very correlations being measured.
- Orders are mean-subtracted separately and normalised jointly, so they count
  by line content, not flux scale, and one weight vector describes the whole
  spectrum.
- The template inherits the data's wavelength frame (air for HARPS) and
  their preprocessing. NaN flux, non-finite velocities and epochs with
  different numbers of orders raise.

## Using it

The bootstrap: rough RVs against any reasonable template, a template from the
data at those velocities, then the RVs again against it.

```python
obs.calc_rv_against_template(library_template)        # rough RVs (km/s)
obs.build_tiravel_template(obs.vels)                  # rho, weights, template
print(obs.tiravel.rho, obs.tiravel.diagnostics["n_eff"])
obs.calc_rv_against_template(obs.tiravel_template)    # RVs against the data's own template
```

On the HARPS spectra of TOI-193 (tutorial 05): ρ = 0.9973 with all 32 epochs
contributing; the TIRAVEL and PHOENIX templates correlate at 0.895; the RVs
against them differ by +0.00 m/s on average with 2.3 m/s rms, against an RV
rms of 14.1 m/s.

- **With UNICOR**, the template works as it is: UNICOR drops the pixels a
  template does not cover.
- **With TODCOR**, the template also works as it is: `CCF2d` removes each
  template's median and fits a free offset per order, so the template's
  continuum level does not matter, and it drops the pixels a template does
  not cover.
- **Check** `rho`, `diagnostics['n_eff']` against K, and the weights, before
  trusting the template ([good practices](good_practices.md#tiravel)).

## Development

### The velocity optimiser

The missing half of the method: find the velocities that maximise ρ. The plan
is a fixed-point iteration:

1. build the template at the current velocities;
2. measure every epoch's RV against it (K correlations);
3. repeat until ρ stops rising, and report the converged velocities.

This costs K correlations per pass, against the K² of a direct maximisation of
ρ over K velocities. Before it can ship it needs:

- **a zero point.** ρ is blind to a common offset, so the velocities must be
  anchored, for example to their starting mean or to a library template;
- **a convergence criterion**, and a test that it converges from rough
  starting RVs on simulated data with a known truth;
- **velocity uncertainties.** None are defined yet. The RV errors measured
  against the final template treat it as noiseless, while it carries about
  1/√n_eff of one epoch's noise and is built from the same epochs.

### Other open items

- **Weighting epochs of different S/N** beyond the rank-1, equal-noise
  approximation of the weights.
- **The relativistic convention**, if the RV engines' option is to carry
  through.

## References

Zucker, S. & Mazeh, T. 2006, [MNRAS, 371, 1513](https://ui.adsabs.harvard.edu/abs/2006MNRAS.371.1513Z/abstract)
(astro-ph/0607293).
