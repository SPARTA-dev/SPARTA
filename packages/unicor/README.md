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
- Reads the RV off the continuous CCF (a local spline through its lags, so the
  RV does not depend on the velocity step while the peak spans ≳ 5 lags), with
  the ML uncertainty from its curvature. (`bisector_span` is also provided, but its formula is known
  to be incorrect and is under review; do not use it scientifically yet.)
- Ships the shared log-λ DSP layer: grid construction, resampling, apodisation,
  Doppler shifts, rotational (Gray 2005) and Gaussian broadening.

All inputs and outputs are plain floats and arrays: **velocities in km/s,
wavelengths in Å**.

## Install

UNICOR is installed from a clone of the SPARTA repository (it is not on
PyPI). The package name is `sparta-unicor` (`unicor` on PyPI is an unrelated
project); the import name is `unicor`.

```bash
git clone https://github.com/SPARTA-dev/SPARTA.git
cd SPARTA
pip install ./packages/unicor
```

For all of SPARTA, see the [main README](../../README.md#installation).

## Quick start

```python
import numpy as np
import unicor

rng = np.random.default_rng(0)

# A template with a few absorption lines (continuum 1) on a fine log-lambda grid
# wider than the observation, and one observed order with step dv = 0.5 km/s,
# red-shifted by 12.3 km/s, with white noise of 0.01 per pixel.
dv, vmax = 0.5, 50.0
t_wave, _ = unicor.log_lambda_grid(4990.0, 5110.0, 0.1)
centres = np.linspace(5010.0, 5090.0, 9)
t_flux = 1 - np.sum(0.5 * np.exp(-0.5 * ((t_wave - centres[:, None]) / 0.1) ** 2), axis=0)
wave, _ = unicor.log_lambda_grid(5000.0, 5100.0, dv)
observed = unicor.resample(t_wave, t_flux, wave / unicor.doppler_factor(12.3))
observed += rng.normal(0.0, 0.01, wave.size)

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
rv, peak = unicor.subpixel_peak(v, c)
# n_eff: independent spectral bins -- native (detector) pixels, not the
# pixels of a finer resampled grid. Here the grid is the native sampling.
erv = unicor.rv_uncertainty(v, c, rv, n_eff=wave.size)
print(f"one order:  RV = {rv:.3f} +- {erv:.3f} km/s (CCF peak {peak:.3f})")

# Several orders on the same lag axis -> one ML-combined CCF, each order
# weighted by its bins; its error uses the total. Here the second "order" is
# the same lines with independent noise.
observed2 = unicor.resample(t_wave, t_flux, wave / unicor.doppler_factor(12.3))
observed2 += rng.normal(0.0, 0.01, wave.size)
_, c2 = unicor.ccf_vs_velocity(template, observed2, wave)
n_eff = np.array([wave.size, wave.size])
combined = unicor.combine_ccfs(np.vstack([c, c2]), n_eff=n_eff)
rv_all, _ = unicor.subpixel_peak(v, combined)
erv_all = unicor.rv_uncertainty(v, combined, rv_all, n_eff=n_eff.sum())
print(f"combined:   RV = {rv_all:.3f} +- {erv_all:.3f} km/s")
```

```text
one order:  RV = 12.291 +- 0.012 km/s (CCF peak 0.987)
combined:   RV = 12.295 +- 0.009 km/s
```

## Two outputs: correlation or log-likelihood

`ccf_vs_velocity` and `combine_ccfs` take `output='ccf'` (the default: the
correlation) or `output='loglike'`: the Zucker (2003) log-likelihood
`lnL(v) = −(N/2)·ln(1 − C(v)²)`, with `N` the independent bins (`n_eff`,
required in this mode), taken as 0 where `C = 0`. For several orders it is the
sum `Σ_i −(N_i/2)·ln(1 − C_i²)`, which equals `−(N_tot/2)·ln(1 − ML²)` of the
combined CCF. `loglike_from_ccf(ccf, n_eff)` converts a CCF you already have.
The log-likelihood is monotonic in `C²`, so both curves peak at the same lag;
read the RV and its error off the CCF (`subpixel_peak`, `rv_uncertainty`) in
either case.

```python
v, lnl = unicor.ccf_vs_velocity(template, observed, wave, output="loglike", n_eff=wave.size)
total = unicor.combine_ccfs(np.vstack([c, c2]), n_eff=n_eff, output="loglike")
```

The uncertainty is the maximum-likelihood error of Zucker (2003): it assumes
white noise on the `n_eff` independent bins. Filtering or normalising the
spectrum first correlates the noise, and the error is then no longer
calibrated; check it by simulation.

## Per-order atoms

The steps above, packaged one call per order — the loop body of SPARTA's
`CCF1d`, which is built from exactly these calls:

- `lag_axis(waves, dv_kms, n_lags)` → `(dln, v)`: checks that every order is
  uniform in ln λ with the declared step and that the orders share it, and
  returns the common velocity of each lag (km/s).
- `ccf_order(wave, flux, t_wave, t_flux, n_lags, dv_kms=..., native_step_kms=...)`
  → `CCFOrder`: resamples the template onto the extended grid, correlates the
  pixels it covers at every lag (`template_coverage`), and returns
  `velocity`, `ccf`, `loglike`, `rv`, `erv` (with `errors=True`), `peak`,
  `n_eff` and the correlated pixels `lo:hi`. It raises `ValueError` if the
  template covers no pixel; `rv`, `erv` and `peak` are NaN if the peak lies
  beyond the scan.
- `effective_bins(n_pix, dv_kms, native_step_kms)`: `n_pix·min(1, dv/native)`,
  the independent bins behind a resampled order.
- `measure_rv(v, ccf, n_eff)` → `RVMeasurement(rv, erv, peak)`: the sub-pixel
  peak and its ML uncertainty.

Continuing the quick start, with the order split in two:

```python
waves, fluxes = np.array_split(wave, 2), np.array_split(observed, 2)
_, v = unicor.lag_axis(waves, dv, K)
orders = [unicor.ccf_order(w, f, t_wave, t_flux, K, dv_kms=dv, native_step_kms=dv)
          for w, f in zip(waves, fluxes)]
n_eff = np.array([o.n_eff for o in orders])
combined = unicor.combine_ccfs(np.vstack([o.ccf for o in orders]), n_eff=n_eff)
rv = unicor.measure_rv(v, combined, n_eff.sum())
print("per order: ", ", ".join(f"{o.rv:.3f}" for o in orders), "km/s")
print(f"combined:   RV = {rv.rv:.3f} +- {rv.erv:.3f} km/s (CCF peak {rv.peak:.3f})")
```

```text
per order:  12.268, 12.309 km/s
combined:   RV = 12.290 +- 0.012 km/s (CCF peak 0.987)
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
