# unicor

Standalone Python engine for **one-dimensional cross-correlation radial
velocities** — the array-level core of [SPARTA](https://github.com/SPARTA-dev/SPARTA)'s
UNICOR. It needs only numpy and scipy.

- Correlates an observation against a template on a uniform log-λ
  (velocity-even) grid, scanning a symmetric velocity range.
- Combines spectral orders by maximum likelihood (Zucker 2003, MNRAS 342, 1291).
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

# Two orders, already on a common log-lambda grid with step dv = 0.5 km/s.
wave, _ = unicor.log_lambda_grid(5000.0, 5100.0, dv=0.5)
template = 1 - 0.5 * np.exp(-0.5 * ((wave - 5050.0) / 0.1) ** 2)
observed = unicor.resample(unicor.doppler_shift_wave(wave, 12.3), template, wave)

v, c = unicor.ccf_vs_velocity(template, observed, dv=0.5, vmax=50)
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
