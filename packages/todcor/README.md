# todcor

Standalone Python engine for **Fourier-domain two-component (TODCOR) radial
velocities** of double-lined spectroscopic binaries — the array-level core of
[SPARTA](https://github.com/SPARTA-dev/SPARTA)'s TODCOR. Its only dependencies are
numpy and [`unicor`](https://github.com/SPARTA-dev/SPARTA/tree/master/packages/unicor) (for the shared log-λ grid and broadening
primitives; it brings in scipy).

`todcor` fits the radial velocities *and* relative amplitudes of both
components of an SB2 simultaneously. It is TODCOR (Zucker & Mazeh 1994, ApJ
420, 806) recast as a Gaussian profile likelihood: on a uniform log-λ grid a
Doppler shift is a translation, so in the Fourier domain it is a phase factor —
the templates are transformed once and reused at every trial velocity. The
amplitudes enter linearly and are solved in closed form (2×2 weighted least
squares) at each `(v1, v2)` node; profiling them out gives the classic TODCOR
map. Spectral windows (orders) of **one epoch** pool by adding their sufficient
statistics, and the Cramér–Rao bound comes from the curvature of the map.

Inputs and outputs are plain floats and arrays: velocities and the log-λ shift
`s` in km/s, ζ (the Fourier conjugate) in s/km, kernel widths as FWHM in km/s.
Velocities are in the `'cz'` Doppler convention (`λ_obs/λ_rest = 1 + v/c`; IAU
2000, IRAF fxcor) unless you pass `velocity_convention='relativistic'`
(`√((1 + v/c)/(1 − v/c))`) to `window_stats` / `todcor_map` and the forward
model; the two differ by ≈ v²/2c (37.5 m/s at 150 km/s). `dv` must be the
grid's actual step, `c` times its ln λ step (`todcor.log_grid_step(wave)`).

## Install

The PyPI name is `sparta-todcor`, and it needs `sparta-unicor`. The import name
is `todcor`.

```bash
# from a SPARTA checkout (pass both, so pip uses the local unicor)
pip install ./packages/unicor ./packages/todcor

# straight from GitHub
pip install \
  "sparta-unicor @ git+https://github.com/SPARTA-dev/SPARTA#subdirectory=packages/unicor" \
  "sparta-todcor @ git+https://github.com/SPARTA-dev/SPARTA#subdirectory=packages/todcor"
```

## Quick start

A self-contained example: two synthetic templates, a noisy composite at known
velocities, and the fit. It runs in under a second.

```python
import numpy as np
import todcor

rng = np.random.default_rng(1)
dv, R = 1.0, 60000.0                                # grid step (km/s), resolving power

# Two rest-frame templates (continuum 1) with random absorption lines, on a fine
# log-lambda grid that covers the observed range plus the velocity shifts,
# broadened to the instrument's resolution (FWHM c/R).
t_wave, _ = todcor.log_lambda_grid(4990.0, 5110.0, 0.1)
def template(n_lines):
    centres = rng.uniform(4992.0, 5108.0, n_lines)
    depths = rng.uniform(0.1, 0.6, n_lines)
    lines = np.sum(depths[:, None] * np.exp(-0.5 * ((t_wave - centres[:, None]) / 0.04) ** 2), axis=0)
    return 1.0 + todcor.broaden_gaussian_residual(-lines, 0.1, todcor.C_KMS / R)
t1, t2 = template(80), template(80)

# The observation: both components shifted and diluted, sampled on the analysis
# grid, plus white noise of 0.01 per pixel.
wave, _ = todcor.log_lambda_grid(5000.0, 5100.0, dv)
sigma = 0.01
observed = todcor.composite_flux(wave, [(t_wave, t1), (t_wave, t2)],
                                 velocities=(25.3, -41.7), amplitudes=(0.7, 0.3))
observed += rng.normal(0.0, sigma, wave.size)

# Residuals (flux - 1) on the analysis grid. The templates go there at rest;
# the observation is already broadened, so it is only tapered, like them.
taper = todcor.tukey_window(wave.size)
g1 = (todcor.resample(t_wave, t1, wave) - 1.0) * taper
g2 = (todcor.resample(t_wave, t2, wave) - 1.0) * taper
data = (observed - 1.0) * taper

v1 = v2 = np.arange(-80.0, 80.5, 0.5)               # trial velocities (km/s)
res = todcor.todcor_map(data, g1, g2, todcor.log_grid_step(wave), v1, v2, sigma=sigma,
                        nonneg=True)
crlb = todcor.crlb_from_map(res["logL_profile"], v1, v2)
best = res["best"]
print(f"grid peak: v1 = {best['v1']:.1f}, v2 = {best['v2']:.1f} km/s, "
      f"a1 = {best['a1']:.3f}, a2 = {best['a2']:.3f}, chi2_red = {best['chi2_red']:.2f}")
print("refined:   v1 = {:.3f} +- {:.3f}, v2 = {:.3f} +- {:.3f} km/s".format(
      crlb["peak_refined"][0], crlb["sigma_v1"], crlb["peak_refined"][1], crlb["sigma_v2"]))
```

```text
grid peak: v1 = 25.5, v2 = -41.5 km/s, a1 = 0.702, a2 = 0.296, chi2_red = 1.01
refined:   v1 = 25.301 +- 0.012, v2 = -41.647 +- 0.030 km/s
```

The injected values are v1 = 25.3, v2 = −41.7 km/s, a1 = 0.7, a2 = 0.3. The
observation is never broadened here: real data already carry the instrumental
profile, so only the templates are matched to the resolution. `sigma` is the
per-pixel noise of the observed flux; with it right, `chi2_red` ≈ 1. The
quadratic refinement of `crlb_from_map` needs a map step that resolves the
peak; SPARTA's `CCF2d` re-evaluates a finer local map around it instead (see
the method notes).

## Two maps: correlation or log-likelihood

`todcor_map` and `profile_from_stats` return both surfaces over `(v1, v2)`:

- `res["R"]` — the two-dimensional correlation of Zucker & Mazeh (1994),
  `R² = 1 − χ²_min/SD` (the fraction of the data's power the best-fit
  amplitude pair explains), in [0, 1] and free of the noise scale;
- `res["logL_profile"]` (and `"logL_marginal"`) — the Gaussian log-likelihood
  `−χ²/2`, which `crlb_from_map` needs for the errors.

They are related by `logL_profile = −(SD/2)(1 − R²)` and share their peak, so
`best` (now with `best["R"]`) is the same either way. The keyword
`output='ccf'` (default) or `output='loglike'` only selects which of the two is
`res["map"]`, the primary surface; any other value raises `ValueError`. The
derivation and the reduction to the one-dimensional CCF are in the method notes.

Several windows (orders) of the same epoch, evaluated on one shared velocity
grid, pool by adding their statistics:

```python
stats = [todcor.window_stats(d, g1, g2, dv, v1, v2, sigma=s) for d, g1, g2, dv, s in windows]
res = todcor.profile_from_stats(todcor.combine_stats(stats), v1, v2, nonneg=True)
```

## Learn the method

- [`docs/todcor_method.md`](https://github.com/SPARTA-dev/SPARTA/blob/master/packages/todcor/docs/todcor_method.md):
  the derivation (sufficient statistics, the χ² quadratic form, the
  two-dimensional correlation `R`, profile vs marginal amplitudes, Parseval, the even-N Nyquist drop, window combination,
  the noise budget and its assumptions).
- SPARTA's [`examples/todcor/`](https://github.com/SPARTA-dev/SPARTA/tree/master/examples/todcor)
  notebooks 01–03 walk through it numerically with this package alone.
  SPARTA's `CCF2d` and `Observations.calc_rv_against_two_templates` run it on
  multi-order spectra.

## Test

```bash
pip install "./packages/todcor[test]"
pytest packages/todcor
```

## Reference

Zucker, S. & Mazeh, T. 1994, ApJ, 420, 806 — "Study of spectroscopic binaries
with TODCOR. I. A new two-dimensional correlation algorithm."
