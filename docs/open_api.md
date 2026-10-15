# The open surface — what you can build an analysis from

**Contract.** A name is on the open surface when it imports from the path
shown, its docstring states units and assumptions, and a test covers it
(`tests/test_open_api.py` checks every row below against the code). The
rule behind it: a user should be able to build their own RV or periodicity
analysis from these pieces, call each step on its own, and print or plot
what it returns. The objects (`Observations`, `CCF1d`, ...) are recipes
assembled from exactly these atoms — the `tests/test_atoms_*.py` modules
rebuild their results from the atoms bit for bit.

**Where they live.** SPARTA's own atoms import from `sparta` itself
(resolved lazily: `import sparta` loads nothing, and neither matplotlib nor
specutils arrives until a function that needs it is called). The RV engines
are the standalone packages `unicor` and `todcor`; import their atoms from
them, not from `sparta`. The full signatures and docstrings are in
[api.md](api.md).

**Conventions every row relies on** (in full in
[conventions.md](conventions.md)): wavelengths in Å, velocities in km/s,
times in days, periodogram frequencies in 1/day. Plain floats and arrays
are read in these units. Correlation needs a velocity-even grid, uniform in
ln λ: SPARTA resamples with step ln(1 + dv/c), and every engine measures the
*actual* step of the grid it is given (`unicor.log_grid_step`) rather than
trusting a nominal `dv`. Velocities follow the `'cz'` Doppler convention,
λ_obs/λ_rest = 1 + v/c, unless `velocity_convention='relativistic'` is
asked for. Error bars count *native* pixels (`n_eff`, `oversampling`) and
assume white noise on them.

**Not on the surface, on purpose:** the objects `Spectrum`, `Template`,
`Observations`, `TimeSeries`, `CCF1d`, `CCF2d`, `TIRAVEL` and
`PeriodicityDetector`. They import from `sparta` too, but they are
conveniences built from the atoms below, for the common recipes; the
how-tos ([`examples/howto/`](../examples/howto/)) use them. Also off it:
`unicor.bisector_span` (known to be incorrect, under review) and
`unicor.check_output` (argument validation).

## Preprocessing — one order's arrays in, arrays out

The steps of `Spectrum.trim`, `interpolate`, `remove_cosmics`,
`filter_spectrum` and `preprocess`, which are loops over these.

| name | import from | what it does | units | assumptions |
|---|---|---|---|---|
| `trim_margins` | `sparta` | drop NaN pixels, then padded order margins, then fixed pixel counts from each end | Å; pixels | a margin is a leading/trailing run whose running median is below `margin_fraction` of the order's median; trim before filtering (a filtered edge jump becomes a step) |
| `resample_to_log_grid` | `sparta` | resample one order onto a ln λ grid | Å; `dv_kms` km/s | step ln(1 + dv/c), c from astropy; the grid is inset one median input step at each end, so the spline never extrapolates |
| `resample_to_linear_grid` | `sparta` | resample one order onto an evenly spaced λ grid | Å; `delta` in units of the median input step | inset one new step at each end; not velocity-even (no correlation on it) |
| `resample_flux` | `sparta` | the cubic spline behind both resamplings, at arbitrary wavelengths | Å (or `unit`) | increasing grids; NaN outside the input span; specutils' `SplineInterpolatedResampler`, loaded on the first call |
| `coarsest_native_step` | `sparta` | the native step after a regrid: max(previous, c·median(Δλ/λ) of the new grid) | km/s; Å | a finer grid adds samples, not independent pixels |
| `remove_cosmics` | `sparta` | replace upward outliers by linear interpolation in λ between unclipped neighbours | Å; threshold in mad_std | upward only, iterated about the median; NaN pixels stay NaN |
| `bandpass` | `sparta` | zero-phase Butterworth band-pass, output standardized | Å; `lowcut` in units of 1/(λ span), `highcut` in units of the pixel Nyquist frequency | evenly sampled order (Nyquist from the first step); the cuts in 1/Å therefore depend on the order's length and pixel step ([limitations](limitations.md#spectra-and-templates)); filters the flux minus its 80th percentile; output zero mean, unit variance — the error bars are then no longer calibrated |
| `resampling_noise_psd` | `sparta` | the noise power per Fourier mode of white native noise after `resample_flux` onto a log grid | km/s; relative power (white = 1) | one resampling, no earlier filtering; native phases uniformly spread unless `phases` is given; the `noise_psd` of `todcor.window_stats` |
| `native_phases` | `sparta` | sub-pixel phase of each native pixel on a log analysis grid | Å; `dv` km/s; phases in analysis pixels | the native wavelengths are known (the order was not resampled before) |
| `phoenix_spectrum` | `sparta` | a PHOENIX-ACES-AGSS-COND-2011 HiRes model over a wavelength range, as arrays | Å; flux erg/s/cm²/cm ÷ 1e13; Teff K, log g cgs, [Fe/H] and [α/Fe] dex | vacuum unless `air=True` (Morton 2000); not normalized, not broadened; read from the bundled cache — any other model needs `allow_download=True` once |

## UNICOR — grids, Doppler shifts and broadening (`unicor.dsp`)

| name | import from | what it does | units | assumptions |
|---|---|---|---|---|
| `log_lambda_grid` | `unicor` | a uniform ln λ grid and its velocity coordinate u = c ln(λ/λ_lo) | Å; `dv` km/s | step dv/c in ln λ — not ln(1 + dv/c); read the step back with `log_grid_step` |
| `log_grid_dln`, `log_grid_step` | `unicor` | verify a grid is uniform in ln λ and return its step | dimensionless / km/s | relative spread of the step within `rtol` (1e-3), else `ValueError` |
| `dv_from_wave` | `unicor` | velocity step c·median(Δλ/λ) of any grid | km/s | the mid-grid value on a linear grid |
| `extend_log_grid` | `unicor` | a ln λ grid continued by K steps on each side (the template grid of a CCF) | Å | the input is uniform in ln λ |
| `resample` | `unicor` | interpolate flux onto another grid, linear or cubic | Å | source strictly increasing; clamps to the edge values unless `fill` (pass 1.0 for a normalized continuum) |
| `doppler_factor`, `doppler_shift_wave` | `unicor` | λ_obs/λ_rest for a velocity, and shifted wavelengths | km/s; Å | `'cz'` (1 + v/c) by default, `'relativistic'` on request; the two differ by ≈ v²/2c |
| `shift_kms`, `velocity_from_log_shift` | `unicor` | velocity ↔ ln λ shift (as c ln(λ_obs/λ_rest)) | km/s; dimensionless | same conventions; computed with `log1p`/`expm1`, full precision for small v |
| `C_KMS` | `unicor` | the speed of light | km/s (299792.458) | — |
| `tukey_window` | `unicor` | Tukey (tapered-cosine) apodisation window | pixels | the taper makes the noise heteroscedastic, which the white-noise weighting does not model |
| `rotational_kernel`, `broaden_rotational`, `broaden_gaussian` | `unicor` | rotational (Gray 2005) and Gaussian broadening of normalized flux | km/s (`vsini`, FWHM = c/R) | uniform velocity (ln λ) grid; continuum 1 beyond the ends; linear limb darkening `epsilon`; a `vsini` below the grid step gives a one-tap kernel, i.e. no broadening |
| `broaden_rotational_residual`, `broaden_gaussian_residual` | `unicor` | the same on residual flux (continuum 0) | km/s | zero-padded ends; a width ≤ 0 returns the input |
| `rfft_positive`, `parseval_weights` | `unicor` | positive-frequency FFT (DC dropped) and the weights that make a Fourier χ² equal the real-space χ² | ζ in s/km; weights 1/flux² | white, homoscedastic per-pixel noise |
| `normalize`, `zero_lag_correlation` | `unicor` | mean-subtract and L2-normalize; the unshifted inner product | dimensionless | — |

## UNICOR — the per-order CCF (`unicor.ccf`)

The steps of `CCF1d.cross_correlate`, which calls `lag_axis` once,
`ccf_order` per order and `measure_rv` per combined RV.

| name | import from | what it does | units | assumptions |
|---|---|---|---|---|
| `lag_axis` | `unicor` | verify the orders share one ln λ step and return the velocity of each lag | km/s; dimensionless step | every order uniform in ln λ with its measured step equal to `dv_kms` within 1e-3; `'cz'` by default |
| `template_coverage` | `unicor` | the observed pixels whose template is known at every lag | pixel indices | the template is never extrapolated |
| `effective_bins` | `unicor` | independent spectral bins n_pix · min(1, dv/native) | dimensionless; km/s | the native step is the coarsest sampling the data have had |
| `ccf_order`, `CCFOrder` | `unicor` | one order against a template: CCF, log-likelihood, RV, error, peak, `n_eff` | Å; km/s | the order on a ln λ grid of step `dv_kms`; the template at rest, resampled linearly; RV is NaN when the peak cannot be located |
| `ccf_vs_velocity` | `unicor` | Pearson CCF of a signal against an extended template, lag by lag | km/s | template on `extend_log_grid`; every lag a full Pearson correlation over the same N pixels |
| `combine_ccfs` | `unicor` | maximum-likelihood combination of per-order CCFs (Zucker 2003) | dimensionless | a common lag axis; orders weighted by `n_eff` |
| `loglike_from_ccf` | `unicor` | the Zucker (2003) log-likelihood −(N/2) ln(1 − C²) | dimensionless (nats) | white noise on N independent bins; template matched up to an amplitude |
| `measure_rv`, `RVMeasurement` | `unicor` | sub-pixel peak and its ML uncertainty | km/s | for a combined CCF pass the total `n_eff`; raises when the peak cannot be located |
| `subpixel_peak`, `rv_uncertainty`, `ccf_at` | `unicor` | parabola peak, the curvature error (Zucker 2003 §2.3), 3-point interpolation | km/s | the error counts `n_eff` native bins and is calibrated for white native noise only |

## TODCOR — two templates, one window at a time (`todcor`)

The steps of `CCF2d.cross_correlate`: `prepare_window` and `window_stats`
per window, then `joint_fit` (and `window_noise` for per-window noise).

| name | import from | what it does | units | assumptions |
|---|---|---|---|---|
| `prepare_window`, `TodcorWindow` | `todcor` | data and both templates as tapered residuals on one ln λ window | Å; `dv` km/s | templates at rest, resampled linearly, uncovered pixels filled with continuum 1; continuum (median by default) removed before the taper |
| `window_stats` | `todcor` | the six weighted sufficient statistics of one window on a (v1, v2) grid | km/s; ζ s/km; `sigma` flux units | residuals on a uniform ln λ grid of actual step `dv`; white homoscedastic native noise; `oversampling` = dv_native/dv ≥ 1; `noise_psd` enters the noise budget ν only |
| `combine_stats` | `todcor` | pool windows by summing (optionally weighted) statistics | weights dimensionless, e.g. 1/σ² | the same velocity grid and the same epoch for every window; one (a1, a2) shared by all |
| `profile_from_stats` | `todcor` | profile out the amplitudes: correlation R and log-likelihood maps, best node | km/s; R in [0, 1]; ln L in nats | R and ln L share their argmax; `nonneg=True` sets the marginal equal to the profile |
| `chi2_from_stats` | `todcor` | χ² at given amplitudes on every (v1, v2) node | dimensionless | at the noise the statistics carry |
| `crlb_from_map` | `todcor` | Cramér–Rao errors from a local quadratic fit to ln L, and the sub-grid peak | km/s; covariance km²/s² | an isolated, well-resolved peak; a patch small next to the peak width (use a fine map); NaN on a grid edge or for non-positive-definite curvature; scale by √χ²_red when it is not 1 |
| `refine_grid` | `todcor` | the fine local (v1, v2) grid around a coarse node | km/s | an interior node |
| `joint_fit`, `TodcorFit` | `todcor` | coarse map, fine map, CRLB and every value at the refined peak of pooled windows | km/s | as `combine_stats`; fine statistics cached by coarse node for re-weighting |
| `window_noise` | `todcor` | each window's noise from its share of a unit-noise joint χ² | flux units | the four shared parameters charged to windows ∝ ν_i; windows with fewer than `min_dof` dof take the common estimate |
| `todcor_map` | `todcor` | one window's maps: `profile_from_stats(window_stats(...))` | as `window_stats` | one window of one epoch |
| `resampled_noise_psd` | `todcor` | relative noise power per mode from a resampler's impulse responses | dimensionless (white = 1) | any resampler linear in the flux (e.g. `resample_flux`'s cubic spline); responses at representative sub-pixel phases, weighted by their frequency |
| `shift_template_flux`, `composite_flux` | `todcor` | a template Doppler-shifted onto a grid, and a noiseless composite of shifted templates | Å; km/s; amplitudes dimensionless | rest-frame templates with continuum 1; `'cz'` by default; `kind='cubic'` for coarsely sampled templates (PHOENIX) |
| `model_flux` | `todcor` | the best-fit composite by the fit's own Fourier phase-shift model | km/s; amplitudes dimensionless | the circular shift the estimator assumes, not a resampling |

## TIRAVEL — a template from the observations (`sparta.tiravel_ops`)

The steps of `TIRAVEL.build_template` at fixed trial velocities (Zucker &
Mazeh 2006); the optimisation of ρ over the velocities is not implemented.

| name | import from | what it does | units | assumptions |
|---|---|---|---|---|
| `rest_frame_wavelengths` | `sparta` | observed → rest wavelengths, λ / (1 + v/c) | Å; km/s | the exact division, not λ(1 − v/c); exact under the 'cz' convention, which TIRAVEL uses throughout (no relativistic option) |
| `common_grid` | `sparta` | the reference epoch's rest-frame grid, clipped to the range every epoch covers | Å; km/s; `margin` pixels | no epoch is extrapolated; the reference epoch is not resampled |
| `align_spectra` | `sparta` | shift each epoch to rest and resample it onto the common grid | Å; km/s | cubic by default (linear would depress the correlations) |
| `normalize_rows` | `sparta` | mean-subtract each order per epoch, L2-normalize each epoch across orders | dimensionless | orders count by line content, not flux scale; a flat epoch stays zero |
| `correlation_matrix` | `sparta` | the K × K epoch correlation matrix, one Gram product | dimensionless, in [−1, 1] | rows from `normalize_rows` |
| `alignment` | `sparta` | ρ = (λ_max − 1)/(K − 1), the co-addition weights and diagnostics (`n_eff`, ...) | dimensionless | symmetric matrix (`eigh`); weights unit norm, bulk positive |
| `coadd` | `sparta` | Σ_i w_i · row_i, the maximum-likelihood template | arbitrary scale | the epochs are noisy, shifted copies of one template; maximum likelihood strictly for a rank-1 model with equal i.i.d. noise on the normalized rows, which unit-normalizing epochs of different S/N only approximates |
| `build_tiravel`, `TiravelResult` | `sparta` | the whole fixed-velocity step, orders pooled into one correlation matrix | Å; km/s | K ≥ 2 epochs with the same orders and finite flux; only velocity differences matter |

## PDC and USuRPER — periodicity in distances (`sparta.pdc`)

The steps of `PeriodicityDetector.calc_pdc`, `calc_usurper` and
`calc_partial_periodogram`: a distance matrix, its centring, the scan.

| name | import from | what it does | units | assumptions |
|---|---|---|---|---|
| `scalar_distance_matrix` | `sparta` | \|y_i − y_j\| of scalar values (e.g. RVs) | the unit of y | — |
| `error_aware_distance_matrix` | `sparta` | distances of values with Gaussian errors (Binnenfeld, Shahaf & Zucker 2024, Eq. 3) | √(unit of y) | the published form (no √π on the erf term); negative arguments clipped to 0 |
| `spectral_distance_matrix` | `sparta` | chord distances of spectra on the unit sphere (USuRPER, Binnenfeld et al. 2020) | Å in; distances in [0, 2] | spectra linearly resampled onto their common grid, mean-subtracted, L2-normalized |
| `get_wv_grid`, `resample_spec_to_common_grid` | `sparta` | the uniform λ grid of the spectra's overlap, and the spectra on it, normalized | Å | grid step = finest median pixel step × `factor`; linear interpolation |
| `phase_distance_matrix` | `sparta` | the phase kernel d(1/f − d), d = (t_i − t_j) mod 1/f | times days, f 1/day; days² | f > 0 |
| `double_centering` | `sparta` | double-centred matrix (the biased statistic) | as the input | — |
| `unbiased_u_centering`, `inner_prod`, `norm_z` | `sparta` | U-centring, the U-centred inner product, the component orthogonal to another matrix (Székely & Rizzo 2014) | as the input | n > 3 (`unbiased_u_centering` raises at n ≤ 3) |
| `partial_centering` | `sparta` | U-centre a matrix and a control, and remove the control's component (Binnenfeld et al. 2022) | as the input | the arguments of `pdc_power(..., control=C)` |
| `pdc_power` | `sparta` | the PDC periodogram at every trial frequency (Zucker 2018) | times days, frequencies 1/day; power dimensionless | `a_centered` centred as the phase matrices will be (unbiased by default); `control` only with the unbiased statistic; n > 3 for it |
| `pdc_pvalue`, `pdc_power_level` | `sparta` | single-frequency p-value of an unbiased power, χ²₁.sf(n·power + 1) (Shen et al. 2019), and its inverse | dimensionless | the unbiased statistic at one trial frequency — not a false-alarm probability over a frequency grid |

## Data in, figures out

| name | import from | what it does | units | assumptions |
|---|---|---|---|---|
| `instruments` | `sparta` | per-product archive loaders by survey key (`instruments.get(key).load(path, ...)`) and the `Simulation` of a product | Å; km/s; days | wavelength frame, time system and native sampling per product, in [instruments.md](instruments.md) |
| `plotting` | `sparta` | the figure style: data as black steps, models red, one extra series blue, maps grey-scale | — | importing it starts matplotlib; `apply_style` sets session rcParams, the plot methods never do |
