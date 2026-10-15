# todcor

Standalone Python engine for **two-component (TODCOR) radial
velocities** of double-lined spectroscopic binaries — the array-level core of
[SPARTA](https://github.com/SPARTA-dev/SPARTA)'s TODCOR. Its only dependencies are
numpy and [`unicor`](https://github.com/SPARTA-dev/SPARTA/tree/master/packages/unicor) (for the shared log-λ grid and broadening
primitives; it brings in scipy).

`todcor` fits the radial velocities *and* relative amplitudes of both
components of an SB2 simultaneously. It is TODCOR (Zucker & Mazeh 1994, ApJ
420, 806) recast as a Gaussian profile likelihood over the observed pixels of
each window (echelle order). On a uniform log-λ grid a Doppler shift is a
translation: each template is resampled onto the window's grid extended past
both ends by the largest trial shift, and shifted by a Fourier phase factor,
so nothing wraps around and no taper is needed. A free continuum offset per
window and the amplitudes enter linearly and are solved in closed form at
each `(v1, v2)` node; profiling them out gives the classic TODCOR map.
Windows of **one epoch** pool by adding their sufficient statistics (shared
amplitudes) or with one amplitude pair per window, and the Cramér–Rao bound
comes from the curvature of the map.

Inputs and outputs are plain floats and arrays: velocities and the log-λ shift
`s` in km/s, wavelengths in Å, kernel widths as FWHM in km/s.
Velocities are in the `'cz'` Doppler convention (`λ_obs/λ_rest = 1 + v/c`; IAU
2000, IRAF fxcor) unless you pass `velocity_convention='relativistic'`
(`√((1 + v/c)/(1 − v/c))`) to `window_stats` / `todcor_map` and the forward
model; the two differ by ≈ v²/2c (37.5 m/s at 150 km/s). `dv` must be the
grid's actual step, `c` times its ln λ step (`todcor.log_grid_step(wave)`).

## Install

TODCOR is installed from a clone of the SPARTA repository (it is not on
PyPI), together with UNICOR, which it needs. The package name is
`sparta-todcor`; the import name is `todcor`.

```bash
git clone https://github.com/SPARTA-dev/SPARTA.git
cd SPARTA
pip install ./packages/unicor ./packages/todcor   # both in one call, so pip uses the local unicor
```

For all of SPARTA, see the [main README](../../README.md#installation).

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

# The window: the observed pixels, and both templates resampled onto that grid
# extended past each end by the largest trial velocity (no wrap, no taper).
v1 = v2 = np.arange(-80.0, 80.5, 0.5)               # trial velocities (km/s)
win = todcor.prepare_window(wave, observed, t_wave, t1, t_wave, t2, v_max=80.0)
res = todcor.todcor_map(win.data_g, win.g1, win.g2, win.dv, v1, v2, sigma=sigma,
                        nonneg=True)
crlb = todcor.crlb_from_map(res["logL_profile"], v1, v2)
best = res["best"]
print(f"grid peak: v1 = {best['v1']:.1f}, v2 = {best['v2']:.1f} km/s, "
      f"a1 = {best['a1']:.3f}, a2 = {best['a2']:.3f}, chi2_red = {best['chi2_red']:.2f}")
print("refined:   v1 = {:.3f} +- {:.3f}, v2 = {:.3f} +- {:.3f} km/s".format(
      crlb["peak_refined"][0], crlb["sigma_v1"], crlb["peak_refined"][1], crlb["sigma_v2"]))
```

```text
grid peak: v1 = 25.5, v2 = -41.5 km/s, a1 = 0.701, a2 = 0.296, chi2_red = 1.06
refined:   v1 = 25.295 +- 0.012, v2 = -41.644 +- 0.029 km/s
```

The injected values are v1 = 25.3, v2 = −41.7 km/s, a1 = 0.7, a2 = 0.3. The
observation is never broadened here: real data already carry the instrumental
profile, so only the templates are matched to the resolution. The templates
must cover the window ± `v_max`; observed pixels they do not reach are
dropped. `sigma` is the per-pixel noise of the observed flux; with it right,
`chi2_red` ≈ 1 at the peak (1.005 here at the refined velocities; the grid
node above is up to 0.25 km/s off the peak, hence 1.06). `crlb_from_map` reads the peak from a degree-6 polynomial over
±4 nodes, so the map step must resolve the peak; SPARTA's `CCF2d` re-evaluates
a finer local map around it (see the method notes).

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
grid, pool by adding their statistics (one amplitude pair for all), or keep
one amplitude pair per window:

```python
# windows: (TodcorWindow, noise) pairs, each window from prepare_window
stats = [todcor.window_stats(w.data_g, w.g1, w.g2, w.dv, v1, v2, sigma=s) for w, s in windows]
res = todcor.profile_from_stats(todcor.combine_stats(stats), v1, v2, nonneg=True)  # one (a1, a2)
res = todcor.profile_per_window(stats, None, v1, v2, nonneg=True)    # one (a1, a2) per window
```

## Per-window atoms

SPARTA's `CCF2d` is a chain of four functions on plain arrays. Chained by hand,
they reproduce its fit bit for bit:

- `prepare_window(wave, flux, t1_wave, t1_flux, t2_wave, t2_flux, v_max=, ramp=20)`
  → `TodcorWindow` (`wave`, `dv`, `data_g`, `g1`, `g2`, `extension`): the
  observed pixels the templates cover across ±`v_max`, and both templates
  resampled onto that grid extended by `extension` pixels on each side
  (`v_max` plus a `ramp` falling to zero);
- `joint_fit(stats, weights, prepared, v1, v2, nonneg=..., output=..., amplitudes='shared', fine_cache=None)`
  → `TodcorFit` (`pooled`, `profile`, `coarse`, `fine`, `crlb`, `at_peak`,
  `best`, `window_chi2`): pool the windows, find the peak on the coarse map,
  re-evaluate the statistics exactly on a fine local map (`refine_grid`),
  get the peak and the CRLB from that map, and evaluate the amplitudes at the
  refined velocities;
- `window_noise(chi2, nu, min_dof=50, amplitudes='shared')` →
  `(sigma, fallback)`: each window's noise from its share of the joint χ².

`amplitudes='per_order'` gives each window its own `(a1, a2)` — its own light
ratio, for two stars of different temperature over a wide range — at two
free parameters per window; `best['a1']`, `best['a2']` then hold one value per
window. Pass the same `amplitudes` to `window_noise`.

Continuing the quick start with a second, noisier order, at unknown noise:

```python
half = wave.size // 2
noisier = observed[half:] + rng.normal(0.0, 0.03, wave.size - half)  # a second, noisier order
orders = [(wave[:half], observed[:half]), (wave[half:], noisier)]
v1 = v2 = np.arange(-80.0, 80.5, 1.0)

prepared, stats = [], []
for w, f in orders:
    # v_max: the coarse grid plus the fine map's reach (2 coarse steps)
    win = todcor.prepare_window(w, f, t_wave, t1, t_wave, t2, v_max=82.0)
    kw = {}                                   # unit, white noise: estimated below
    prepared.append((win, kw))
    stats.append(todcor.window_stats(win.data_g, win.g1, win.g2, win.dv, v1, v2, **kw))

cache = {}                                    # fine-map statistics, shared by both passes
fit = todcor.joint_fit(stats, None, prepared, v1, v2, nonneg=True, output="ccf",
                       fine_cache=cache)
noise, fallback = todcor.window_noise(fit.window_chi2, [st["nu"] for st in fit.at_peak])
fit = todcor.joint_fit(stats, 1.0 / noise ** 2, prepared, v1, v2, nonneg=True, output="ccf",
                       fine_cache=cache)
(rv1, rv2), crlb = fit.crlb["peak_refined"], fit.crlb
print(f"noise per window: {noise.round(4)}")
print(f"v1 = {rv1:.3f} +- {crlb['sigma_v1']:.3f}, v2 = {rv2:.3f} +- {crlb['sigma_v2']:.3f} km/s, "
      f"a1 = {fit.best['a1']:.3f}, a2 = {fit.best['a2']:.3f}")
```

```text
noise per window: [0.01   0.0317]
v1 = 25.299 +- 0.016, v2 = -41.621 +- 0.041 km/s, a1 = 0.697, a2 = 0.298
```

The second pass weights each window by `1/σ_i²`. The statistics are linear in
the weights, so it shifts no template again, and it reuses the cached fine
map if the coarse peak has not moved. The true noise is 0.01 and
√(0.01² + 0.03²) ≈ 0.032. For a spectrum resampled from coarser native
pixels, pass `oversampling` and a `noise_psd` that models the resampled noise
(`todcor.resampled_noise_psd`); SPARTA's
`sparta.preprocess.resampling_noise_psd` builds it for SPARTA's spline
resampler.

## Learn the method

- [`docs/todcor_method.md`](https://github.com/SPARTA-dev/SPARTA/blob/master/packages/todcor/docs/todcor_method.md):
  the derivation (the window and its extended templates, the sufficient
  statistics, the χ² quadratic form, the two-dimensional correlation `R`,
  profile vs marginal amplitudes, shared and per-window amplitudes, the peak
  refinement, the noise budget and its assumptions).
- SPARTA's [`examples/howto/todcor_detailed/`](https://github.com/SPARTA-dev/SPARTA/tree/master/examples/howto/todcor_detailed)
  notebooks 01–03 walk through it numerically with this package alone.
  SPARTA's `CCF2d` and `Observations.calc_rv_against_two_templates` run it on
  multi-order spectra.

## Test

```bash
# both in one call, so pip uses the local unicor instead of looking it up on PyPI
pip install "./packages/unicor" "./packages/todcor[test]"
pytest packages/todcor
```

## Reference

Zucker, S. & Mazeh, T. 1994, ApJ, 420, 806 — "Study of spectroscopic binaries
with TODCOR. I. A new two-dimensional correlation algorithm."
