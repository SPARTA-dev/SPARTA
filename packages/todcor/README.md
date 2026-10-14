# todcor

Standalone Python engine for **Fourier-domain two-component (TODCOR) radial
velocities** of double-lined spectroscopic binaries — the array-level core of
[SPARTA](https://github.com/SPARTA-dev/SPARTA)'s TODCOR. Its only dependencies are
numpy and [`unicor`](../unicor) (for the shared log-λ grid and broadening
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

```python
import numpy as np
import todcor

dv, R = 1.0, 60000.0
wave, _ = todcor.log_lambda_grid(5000.0, 5100.0, dv)

# Residual templates (flux - 1), instrumentally broadened and apodised.
def prep(flux):
    g = todcor.broaden_gaussian_residual(flux - 1.0, dv, todcor.C_KMS / R)
    return g * todcor.tukey_window(len(g))

g1, g2 = prep(template1_flux), prep(template2_flux)   # on `wave`
data = prep(observed_flux)                             # composite, on `wave`

v1 = np.arange(-60, 61, 1.0)
v2 = np.arange(-60, 61, 1.0)
res = todcor.todcor_map(data, g1, g2, dv, v1, v2, sigma=0.01)
best = res["best"]                                    # v1, v2, a1, a2, chi2
crlb = todcor.crlb_from_map(res["logL_profile"], v1, v2)
```

Several windows (orders) of the same epoch:

```python
stats = [todcor.window_stats(d, g1, g2, dv, v1, v2, sigma=s) for d, g1, g2, s in windows]
res = todcor.profile_from_stats(todcor.combine_stats(stats), v1, v2)
```

## Learn the method

- [`docs/todcor_method.md`](docs/todcor_method.md): the derivation (sufficient
  statistics, the χ² quadratic form, profile vs marginal amplitudes, Parseval,
  the even-N Nyquist drop, window combination).
- SPARTA's [`examples/todcor/`](../../examples/todcor/) notebooks 01–03 walk
  through it numerically using this package alone; notebook 04 runs it through
  SPARTA's `Spectrum`/`Template`/`Observations` objects.

## Test

```bash
pip install "./packages/todcor[test]"
pytest packages/todcor
```

## Reference

Zucker, S. & Mazeh, T. 1994, ApJ, 420, 806 — "Study of spectroscopic binaries
with TODCOR. I. A new two-dimensional correlation algorithm."
