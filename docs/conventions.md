# Conventions

Units, grids, frames and error bars: what every SPARTA number means. The
engines that rely on these conventions are described in
[correlation_engines.md](correlation_engines.md) and
[periodograms.md](periodograms.md); the per-product frames and sampling are
in [instruments.md](instruments.md).

## Units

| Quantity | Unit | Notes |
| --- | --- | --- |
| Velocities, RVs, velocity steps (`dv`, `dvel`, `vel_bound`, `vsini`, `bcv`) | km/s | `unicor.C_KMS` = 299792.458 km/s |
| Wavelengths | Å | air or vacuum depending on the product, see below |
| Times | days | in the time system the product records, see below |
| Periodogram frequencies | 1/day | must be > 0 |
| Template-broadening widths | km/s (FWHM) | instrumental FWHM = c/R |
| TODCOR Fourier conjugate ζ | s/km | |

Plain numbers are read in these units. Where a parameter accepts an astropy
`Quantity` (e.g. `dv`, `vel_bound`, `bcv`, the `Simulation` fields) it is
converted once at the API boundary; the engines (`unicor`, `todcor`,
`sparta.pdc`, `sparta.tiravel_ops`) take plain floats only.

## The uniform ln λ grid

A Doppler shift multiplies wavelengths by a constant factor, so on a grid
uniform in ln λ it is a translation by a whole number of pixels (or a phase
factor in the Fourier domain). UNICOR, TODCOR and TIRAVEL therefore work on
such a grid. Two constructions are in use:

| Builder | ln λ step Δ | velocity step c·Δ |
| --- | --- | --- |
| `Spectrum.interpolate(delta=dv, method='log')` | ln(1 + dv/c) | dv (1 − dv/2c) |
| `unicor.log_lambda_grid(lo, hi, dv)` | dv/c | dv |

They differ by dv/2c (relative), so no engine trusts a nominal `dv`. Each one
measures the step of the grid it is given:
`unicor.log_grid_dln(wave)` returns Δ = ln(λ_last/λ_first)/(N − 1) after
checking that the grid is strictly increasing and that the spread of its
ln λ steps is at most 10⁻³ of the median step; otherwise it raises
`ValueError` ("Resample it onto a log-lambda grid first").
`unicor.log_grid_step(wave)` is c·Δ in km/s. On top of that:

- `CCF1d` and `CCF2d` resample a spectrum whose `Info['GridType']` is not
  `'log'` with `Spectrum.interpolate(delta=dv, method='log')` (cubic spline,
  specutils' `SplineInterpolatedResampler`; the new grid starts one median
  input step inside each order);
- both check every order's measured step against `Info['GridDelta']`
  (relative tolerance 10⁻³) and raise on a mismatch, which catches a grid
  that only *claims* to be logarithmic (e.g. after pixels were dropped);
- `CCF1d` also requires all orders to share one Δ (to 10⁻³ of a lag at the
  scan edge), since their CCFs share one velocity axis; `CCF2d` passes each
  order's own c·Δ to `todcor.window_stats`.

## Velocity convention

A lag of k pixels is the wavelength ratio λ_obs/λ_rest = exp(kΔ). How that
ratio becomes a velocity is the `velocity_convention`:

| `velocity_convention` | λ_obs/λ_rest | lag k → v | log shift s(v) |
| --- | --- | --- | --- |
| `'cz'` (default) | 1 + v/c | c·expm1(kΔ) | c·ln(1 + v/c) |
| `'relativistic'` | √((1 + v/c)/(1 − v/c)) | c·tanh(kΔ) | c·artanh(v/c) |

`'cz'` is the IAU (2000) radial-velocity measure and the velocity IRAF
`fxcor` reports. For the same spectrum the `'cz'` velocity exceeds the
`'relativistic'` one by ≈ v²/2c: 16.7 m/s at 100 km/s, 37.5 m/s at 150 km/s.
Reading a lag as k·dv instead (what UNICOR did before 0.2.0) is off from
`'cz'` by −v²/2c. Because the lag → velocity map is non-linear, the CCF velocity axis
(`CCF1d.Corr['vel']`) is very slightly non-uniform; nothing downstream
assumes it is uniform.

The same keyword exists in `CCF1d.cross_correlate`, `CCF2d.cross_correlate`,
both `Observations` RV drivers, `Template.doppler`, `Simulation`, and the
engine functions (`unicor.doppler_factor`, `shift_kms`,
`velocity_from_log_shift`, `ccf_vs_velocity`; `todcor.window_stats`,
`todcor_map`, `composite_flux`, `model_flux`). Use one convention throughout a
project, including when you simulate.

**The barycentric correction is always `'cz'`:** λ → λ·(1 + bcv/c)
(`Spectrum.barycentric_correction`), whatever convention the RVs are read
in, because pipeline BERVs are defined to compose that way,
(1 + z_obs) = (1 + z_star)(1 + bcv/c). `Observations` applies it on ingest
and sets `bcv` to 0, so it is never applied twice.

## Air and vacuum wavelengths

The frame is a property of the archive product, listed per product in
[instruments.md](instruments.md) ("λ frame"; "—" means the archive does not
document it). The rules:

- PHOENIX models are in vacuum. `Template.from_phoenix(..., air=True)`
  converts them to air with Morton (2000, ApJS 130, 403):
  λ_air = λ_vac / n with
  n = 1 + 8.34254·10⁻⁵ + 2.406147·10⁻² / (130 − σ²) + 1.5998·10⁻⁴ / (38.9 − σ²),
  σ = 10⁴/λ_vac (λ in Å), before `wave_range` is applied.
- `Template.from_mask` places the mask windows on the PHOENIX (vacuum) grid
  as given; no conversion is made.
- `Simulation` produces air wavelengths when the emulated product is in air,
  vacuum otherwise (`Simulation.wavelength_frame`).
- Match the template to the data. A frame mismatch shifts every RV by
  roughly 80 km/s in the optical (83.58 km/s for the bundled HARPS TOI-193
  spectra, [tutorial 05](../examples/tutorials/05_tiravel.ipynb)).
  Relative RVs are unaffected, absolute ones are wrong.
- `Observations` refuses a directory whose visits disagree on
  `(time_system, wavelength_frame)`.

## Times

Each loader returns the time stored in the file, in the product's time
system (`LoadedSpectrum.time_system`, e.g. `'BJD-TDB'`, `'MJD-UTC'`; listed
per product in [instruments.md](instruments.md)). SPARTA records the system
(`Observations.time_system`) but **does not convert between systems**; the
attribute `Spectrum.bjd` holds the loader's time whatever its system.
`Observations` makes the times relative: `time_list` is days after
`first_time = int(min(times))`. The periodograms use time differences only.

## Native sampling and the number of independent pixels

Resampling onto a finer grid adds samples, not information: neighbouring
resampled pixels share one native pixel's noise. Both error estimators
therefore count *native* pixels.

- `Spectrum.native_steps()` returns each order's native velocity step (km/s):
  `Info['NativeStep']` if recorded, else c·median(Δλ/λ) of the current grid
  (a spectrum that was never resampled is taken as honestly sampled).
- `Spectrum.interpolate` records `Info['NativeStep']` as the **coarsest** step
  seen so far, max(previous native step, new grid step): resampling finer
  adds no independent pixels, resampling coarser leaves fewer. For a
  linear-λ grid the step is its mid-order value (the median Δλ/λ).
- Archive-resampled products (`Product.resampled`, e.g. the HARPS ESO ADP
  spectra on a 0.01 Å grid) are seeded with the detector step when the
  instrument documents one (`Instrument.native_step_kms`; HARPS: 0.609 km/s,
  from R = 120,000 and 4.1 pixels per FWHM). `Observations` stores
  max(detector step, archive step) per order, by the same coarsest-step rule.
  Products without a documented native step count their archive pixels (see
  [limitations.md](limitations.md#instruments-and-data)).
- UNICOR counts, per order, n_eff = n_pix · min(1, dv/native), where n_pix is
  the number of pixels actually correlated and dv the analysis step
  (`CCF1d.Corr['n_eff']`).
- TODCOR divides each order's Parseval weights by the oversampling
  M = max(1, native/dv) (`CCF2d.windows[i]['oversampling']`, `['n_eff']` =
  n_pix/M), and models the colour of the resampled noise (see
  [correlation_engines.md](correlation_engines.md#noise-oversampling-and-n_eff)).

## Preprocessing order

`Spectrum.preprocess` (and `Observations.preprocess`, epoch by epoch) runs, in
place and in this order:

1. **`trim`** — drop NaN pixels, then the padded order margins (leading and
   trailing runs whose 15-pixel running median is below 10% of the order's
   median flux: exact zeros and noisy unexposed detector alike), then
   `n_trim` pixels from each end.
2. **`interpolate`** — resample onto an even linear-λ grid (`delta` times
   each order's median step).
3. **`remove_cosmics`** — replace upward outliers (> `cosmic_sigma` ×
   mad_std above the median) by linear interpolation between unclipped
   neighbours; the grid is kept.
4. **`filter_spectrum`** — first-order Butterworth band-pass, run forward and
   backward (zero phase) on the flux minus its 80th percentile; each order is
   then standardised to zero mean and unit variance. The continuum goes to 0.
5. **`apply_cosine_bell`** — Tukey taper (`alpha`), bringing each order's
   ends to 0.

Defaults: `n_trim=10, clean_margins=True, remove_nans=True, delta=0.5,
cosmic_sigma=3, filter_lowcut=3, filter_highcut=0.15, alpha=0.3`; the
per-product recommendations are in [instruments.md](instruments.md).

Why this order:

- **Margins before the filter.** Band-pass filtering a jump from ~0 (padding)
  to the continuum turns it into a spurious step at the order edge. Removing
  the noisy NRES margins first lowered the RV scatter of the bundled TOI-677
  spectra from 82 to 75 m/s (CHANGELOG 0.3.0).
- **Resample before the cosmics and the filter.** The filter takes its
  Nyquist frequency from the first wavelength step, so it needs an even grid;
  cosmic replacement interpolates instead of deleting pixels so that the grid
  stays even.
- **Cosmics before the filter**, so a spike is clipped before the filter
  spreads it over its neighbours.
- **Taper last**, on a spectrum whose continuum is already 0: tapering a
  continuum-1 spectrum would itself create features at the order ends.

Pass the template through the same `preprocess` call as the data
([tutorial 03](../examples/tutorials/03_unicor.ipynb)). A preprocessed
spectrum is on a linear-λ grid, so the RV engines resample it once more, onto
their ln λ grid. If it is already tapered, `CCF2d.cross_correlate(...,
alpha=0)` skips TODCOR's own taper; its default `continuum='median'` is ≈ 0
for a filtered spectrum.

## When are the error bars calibrated?

UNICOR's maximum-likelihood error and TODCOR's Cramér–Rao bound are exact
under these assumptions:

1. white noise on the native pixels, resampled once by an interpolator
   linear in the flux;
2. a template that matches the spectrum up to an amplitude (UNICOR) or a pair
   of amplitudes (TODCOR);
3. a well-resolved, isolated peak.

Measured calibration (scatter of RV − truth over the reported error, so 1 is
honest and < 1 conservative):

| Setup | UNICOR | TODCOR | Source |
| --- | --- | --- | --- |
| White noise on the native grid | 1.03 | 0.94 (primary), 1.02 (secondary) | `tests/instruments/test_simulated.py` |
| HARPS archive grid, native step seeded | 1.07 (1.25 without seeding) | — | same |
| Photon noise (σ ∝ √flux) | 0.89 | — | same |
| After `preprocess` (band-pass, normalisation, linear-λ resampling) | 0.85 at `filter_highcut=0.15`, 0.72 at 0.9 (eRV/scatter 1.17, 1.38) | 0.72–0.74 | `CCF1d` / `CCF2d` docstrings |
| TODCOR analysis grid 1.5× coarser than native | — | 0.82–0.91 | [todcor_method.md §9](../packages/todcor/docs/todcor_method.md) |

So:

- **Calibrated:** white noise on native pixels, on the raw (continuum-
  normalised) spectrum, for products whose native step is known.
- **Conservative:** after filtering (17–38% too large for UNICOR, ~1.4× for
  TODCOR); under photon noise (UNICOR ~10%, because line cores are quieter
  than the white-noise formula assumes); with template mismatch, which both
  estimators count as noise (in
  [tutorial 07](../examples/tutorials/07_usurper.ipynb) the reported errors are
  about three times the scatter for a star whose temperature changes); on
  TODCOR grids whose nodes fall between native pixels.
- **Optimistic:** an archive-resampled product without a documented native
  step, since its archive pixels are counted as independent.

TODCOR with an explicit `sigma` takes it at face value; `chi2_red` is then a
consistency check, and the errors are not rescaled by it. With
`sigma=None` (default) the noise is estimated per order from the fit (see
[correlation_engines.md](correlation_engines.md#todcor-ccf2d)). When the
error bars matter, calibrate them for your setup: simulate it with
`sparta.instruments.Simulation` and compare the RV scatter with the reported
errors, or use the scatter of a constant star.
