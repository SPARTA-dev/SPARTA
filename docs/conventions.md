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
- TODCOR divides each order's pixel weights by the oversampling
  M = max(1, native/dv) (`CCF2d.windows[i]['oversampling']`, `['n_eff']` =
  n_pix/M), and models the colour of the resampled noise (see
  [correlation_engines.md](correlation_engines.md#noise-oversampling-and-n_eff)).

## Normalised flux

One normalisation, `unicor.dsp.standardize`: zero mean and unit variance per
order. The band-pass filter ends with it, so a preprocessed order is
dimensionless and its continuum is 0. USuRPER and the partial periodograms
use it too: their points on the unit sphere are `standardize(f)/√n` on the
spectra's common grid of n pixels, so the chord distance is the RMS difference
of the standardized spectra. The correlation (UNICOR) is blind to the scale.
TODCOR's amplitudes keep their meaning between data and templates of
different pixel counts, which a unit-length scaling would not give.

One deliberate exception: TIRAVEL (`normalize_rows`) subtracts each order's
mean but scales each epoch to unit length over all orders together, so that
orders count by their line content in the one correlation matrix.

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
   backward (zero phase) after removing the order's straight-line trend, with
   Gustafsson's initial conditions (`filtfilt(method='gust')`: each pass
   starts from the state the data themselves imply, so the order's ends are
   filtered as cleanly as its middle); each order is then standardised to
   zero mean and unit variance. The continuum goes to 0. It records
   `Info['NoiseFraction']`, the share of white native noise power the filter
   kept, which UNICOR's error bars count (see below).
   Both cuts are **velocity scales** (km/s), the periods of the band edges:
   variations broader than `continuum_scale_kms` and structure narrower than
   `noise_scale_kms` are removed (each to half its amplitude at the cut), so a
   template and the data get the same physical filter whatever their pixels.
No step tapers the ends. A Tukey taper used to bring them to 0, to hide the
filter's start-up, which spoiled ~continuum_scale/2π (~300 km/s) at each end
when the filter was padded with 30 guessed pixels. But a taper is applied to
the template before it is shifted, so its faded zone moves with the trial
velocity while the data's stays put: it biased UNICOR's RVs by up to ~1 m/s
and doubled their quoted errors. Gustafsson's start-up removes the damage
instead of hiding it (on the bundled NRES spectra the RV scatter fell from 75
to 70 m/s).

Defaults: `n_trim=10, clean_margins=True, remove_nans=True, delta=0.5,
cosmic_sigma=3, continuum_scale_kms=2000, noise_scale_kms=4`; the
per-product recommendations are in [instruments.md](instruments.md) (HARPS
2000 / 4 km/s, NRES 1400 / 9.6 km/s: 1.6–1.7 times the line-spread FWHM c/R
for the noise cut, the values the former pixel-unit settings gave on the
bundled data).

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

Pass the template through the same `preprocess` call as the data
([tutorial 03](../examples/tutorials/03_unicor.ipynb)). A preprocessed
spectrum is on a linear-λ grid, so the RV engines resample it once more, onto
their ln λ grid. TODCOR fits a free continuum offset per order, so the
filtered spectrum's zero level needs no special handling.

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
| White noise on the native grid | 1.03 | 0.97 (primary), 1.06 (secondary) | `tests/instruments/test_simulated.py` |
| HARPS archive grid, native step seeded | 1.07 (1.25 without seeding) | — | same |
| Photon noise (σ ∝ √flux) | 0.89 | — | same |
| After `preprocess` (band-pass, normalisation, linear-λ resampling), both engines counting the kept noise fraction | 0.70–0.80 (eRV/scatter 1.25 HARPS-like, 1.42 NRES-like, simulated); ~0.3 on the real HARPS TOI-193 RVs, where template mismatch adds to it | 0.86 (primary), 0.90 (secondary); 1.8 without the correction | `tests/test_ccf1d.py`, `tests/instruments/test_simulated.py`; docs/roadmap.md |
| TODCOR analysis grid 1.5× coarser than native | — | 0.79 (primary), 0.87 (secondary) | [todcor_method.md §9](../packages/todcor/docs/todcor_method.md) |

So:

- **Calibrated:** white noise on native pixels, on the raw (continuum-
  normalised) spectrum, for products whose native step is known.
- **Conservative, approximately:** after filtering. The band-pass removes
  most of the high-frequency noise, so a filtered spectrum looks cleaner than
  it is and the formula, counting every native pixel, would quote errors
  ~0.6–0.7× the real scatter. UNICOR therefore counts only the share of
  native noise the filter kept (`Info['NoiseFraction']`,
  `sparta.preprocess.filtered_noise_fraction`); that correction ignores the
  share of the lines' own signal the filter removes, so the errors err high,
  ~1.25–1.4× on simulated HARPS- and NRES-like spectra. An exact model is an
  open item ([roadmap](roadmap.md)). TODCOR applies the same correction
  and errs high by 1.1–1.2×. Also conservative: under photon noise (UNICOR ~10%, because line cores are quieter
  than the white-noise formula assumes); with template mismatch, which both
  estimators count as noise (in
  [tutorial 07](../examples/tutorials/07_usurper.ipynb) the reported errors are
  about three times the scatter for a star whose temperature changes); on
  TODCOR grids whose nodes fall between native pixels.
- **Optimistic:** an archive-resampled product without a documented native
  step, since its archive pixels are counted as independent.

TODCOR with an explicit `sigma` takes it at face value (so pass none after
`preprocess`, which normalises and filters the flux); `chi2_red` is then a
consistency check, and the errors are not rescaled by it. With
`sigma=None` (default) the noise is estimated per order from the fit (see
[correlation_engines.md](correlation_engines.md#todcor-ccf2d)). When the
error bars matter, calibrate them for your setup: simulate it with
`sparta.instruments.Simulation` and compare the RV scatter with the reported
errors, or use the scatter of a constant star.
