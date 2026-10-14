# unicor

Standalone Python engine for **one-dimensional cross-correlation radial
velocities** — the array-level core of [SPARTA](https://github.com/SPARTA-dev/SPARTA)'s
UNICOR. It needs only numpy and scipy.

- Correlates an observation against a template on a uniform log-λ
  (velocity-even) grid, scanning a symmetric velocity range. The template is
  sampled on the observed grid extended by the scan half-width, so every lag is
  a Pearson correlation over all the observed pixels — no taper that pulls the
  RV toward zero.
- Combines spectral orders by maximum likelihood (Zucker 2003, MNRAS 342, 1291).
- Maps each lag to a radial velocity in the `'cz'` Doppler convention
  (`λ_obs/λ_rest = 1 + v/c`; IAU 2000, IRAF fxcor) or, on request, the
  `'relativistic'` one (`√((1 + v/c)/(1 − v/c))`), using the grid's actual
  ln λ step.
- Reads the RV off a sub-pixel parabola fit, with the ML uncertainty from the
  CCF curvature, and computes the bisector span.
- Ships the shared log-λ DSP layer: grid construction, resampling, apodisation,
  Doppler shifts, rotational (Gray 2005) and Gaussian broadening.

All inputs and outputs are plain floats and arrays: **velocities in km/s,
wavelengths in Å**.

## Install

The PyPI name is `sparta-unicor` (plain `unicor` on PyPI is an unrelated
project). The import name is `unicor`.

```bash
# from a SPARTA checkout
pip install ./packages/unicor

# straight from GitHub
pip install "sparta-unicor @ git+https://github.com/SPARTA-dev/SPARTA#subdirectory=packages/unicor"
```

## Quick start

```python
import numpy as np
import unicor

# One order on a log-lambda grid with step dv = 0.5 km/s, red-shifted by 12.3 km/s.
dv, vmax = 0.5, 50.0
wave, _ = unicor.log_lambda_grid(5000.0, 5100.0, dv)
t_wave, _ = unicor.log_lambda_grid(4990.0, 5110.0, 0.1)        # wider template
t_flux = 1 - 0.5 * np.exp(-0.5 * ((t_wave - 5050.0) / 0.1) ** 2)
observed = unicor.resample(t_wave, t_flux, wave / unicor.doppler_factor(12.3))

# The template goes on the observed grid continued by K = vmax/dv steps on
# each side: lag k then correlates all observed pixels with the template
# shifted by k. It must cover that extended grid (resample clamps outside it).
K = round(vmax / dv)
template = unicor.resample(t_wave, t_flux, unicor.extend_log_grid(wave, K))

# Lag k shifts ln(lambda) by k * Delta (Delta measured from `wave`); its
# velocity is c * expm1(k * Delta) -- the 'cz' convention, the default -- or
# c * tanh(k * Delta) with velocity_convention='relativistic'. The axis is
# therefore very slightly non-uniform; nothing downstream assumes otherwise.
v, c = unicor.ccf_vs_velocity(template, observed, wave)
rv, peak = unicor.subpixel_peak(v, c)                 # ~12.3 km/s
# n_eff: independent spectral bins -- native (detector) pixels, not the
# pixels of a finer resampled grid. Here the grid is the native sampling.
erv = unicor.rv_uncertainty(v, c, rv, n_eff=wave.size)  # km/s

# Several orders on the same lag axis -> one ML-combined CCF, each order
# weighted by its bins; its error uses the total.
n_eff = np.array([wave.size, wave.size])
combined = unicor.combine_ccfs(np.vstack([c, c]), n_eff=n_eff)
rv_all, _ = unicor.subpixel_peak(v, combined)
erv_all = unicor.rv_uncertainty(v, combined, rv_all, n_eff=n_eff.sum())
```

## Velocity conventions

| `velocity_convention` | `λ_obs/λ_rest` | lag `k` → `v` |
| --- | --- | --- |
| `'cz'` (default) | `1 + v/c` | `c·expm1(k·Δ)` |
| `'relativistic'` | `√((1 + v/c)/(1 − v/c))` | `c·tanh(k·Δ)` |

`expm1(x) = eˣ − 1`, computed accurately for tiny `x`. For one spectrum the two
differ by ≈ v²/2c (37.5 m/s at 150 km/s). `unicor.doppler_factor`,
`unicor.shift_kms` and `unicor.velocity_from_log_shift` expose the same
mapping; pass the matching convention when you simulate a shifted spectrum.

## Test

```bash
pip install "./packages/unicor[test]"
pytest packages/unicor
```

## References

- Zucker, S. 2003, MNRAS, 342, 1291 — cross-correlation and maximum-likelihood
  analysis: a new approach to combining cross-correlation functions.
- Engel, M., Shahaf, S. & Mazeh, T. 2017, PASP, 129, 065002 — UNICOR.
- Gray, D. F. 2005, *The Observation and Analysis of Stellar Photospheres*.
