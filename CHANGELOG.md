# Changelog

Units throughout: velocities in km/s, wavelengths in Angstrom, times in days,
periodogram frequencies in 1/day.

## 0.3.0 (unreleased)

The public API of the object layer is now snake_case, the package layout is
flat, and `Observations` is the only time-series driver. **There are no
aliases or shims**: every old name below is gone and raises `ImportError`,
`AttributeError` or `TypeError`. Numerical results are unchanged except where
"Numerical changes" below says otherwise, with how far the golden master
moved; otherwise it (UNICOR, TODCOR, TIRAVEL, periodograms, instrument
readers) passes bit-for-bit or at 1e-9 relative. The `unicor` engine API
changed only additively: the `output` modes and the per-order atoms (see
"Other changes"). The `todcor` API changed with its windowing (see "Numerical
changes"): the taper and continuum options are gone.

### Modules

The `UNICOR`, `TODCOR`, `TIRAVEL`, `USURPER` and `Auxil` subpackages are
removed. `import sparta` exports `Spectrum`, `Template`, `Observations`,
`TimeSeries`, `CCF1d`, `CCF2d`, `TIRAVEL`, `PeriodicityDetector` and
`instruments`.

| 0.2.0 | 0.3.0 |
|---|---|
| `sparta.UNICOR.Spectrum` | `sparta.spectrum` |
| `sparta.UNICOR.Template` (+ `template_files/`) | `sparta.template` |
| `sparta.UNICOR.CCF1d` | `sparta.ccf1d` |
| `sparta.TODCOR.CCF2d` | `sparta.ccf2d` |
| `sparta.TIRAVEL.TIRAVEL` | `sparta.tiravel` |
| `sparta.TIRAVEL.tiravel_ops` | `sparta.tiravel_ops` |
| `sparta.Auxil.TimeSeries` | `sparta.timeseries` |
| `sparta.Auxil.PeriodicityDetector` | `sparta.periodogram` |
| `sparta.Auxil.PeriodicityDetector.is_numeric_scalar` | `sparta.timeseries.is_numeric_scalar` |
| `sparta.USURPER.USURPER_functions` | `sparta.pdc` |
| `sparta.Observations` (module) | `sparta.observations` |

### One driver: `Observations`

`TimeSeries` is now a plain container (times, values, errors, RVs).

| 0.2.0 | 0.3.0 |
|---|---|
| `TimeSeries.calc_rv_against_template(spec_list, template, ...)` | `Observations.calc_rv_against_template(template, ...)` |
| `TimeSeries.calc_rv_against_two_templates(spec_list, ...)` | `Observations.calc_rv_against_two_templates(...)` |
| `TimeSeries.build_tiravel_template(spec_list, velocities, ...)` | `Observations.build_tiravel_template(velocities, ...)` |
| `TimeSeries.getBIS(ccf_list, ...)` | `Observations.calc_bis(...)` |

- A combined single-lined RV run now fills `time_series.calculated_vrad_list`
  and `time_series.errors`, so the periodograms use the measured RVs; it used
  to be copied by hand. SB2 runs store `vels1/2`, `evels1/2`, `amps1/2` and
  leave the scalar series alone.
- `combine_ccfs=False` no longer reports the mean and standard deviation of
  the per-order RVs as an RV and its error: `vels`, `evels`, `ccf_peaks` are
  `None`, and the per-order results are `(epochs x orders)` arrays
  `vels_per_order`, `evels_per_order`, `ccf_peaks_per_order`.

### PDC: pure functions in `sparta.pdc`

| 0.2.0 | 0.3.0 |
|---|---|
| `calc_PDC(periodicity_detector, f)`, `calc_PDC_unbiased(periodicity_detector, f)` | `pdc_power(times, freqs, a_centered, unbiased=True, control=None)` |
| `calc_pdc_distance_matrix(periodicity_detector, ...)` | `scalar_distance_matrix`, `error_aware_distance_matrix`, `spectral_distance_matrix`, `phase_distance_matrix`, then `double_centering` / `unbiased_u_centering` / `partial_centering` |

`inner_prod`, `norm_z`, `unbiased_u_centering`, `get_wv_grid` and
`resample_spec_to_common_grid` keep their names. `PeriodicityDetector` no
longer mutates the caller's data, and:

- `calc_partial_periodogram(reverse_existing=...)`: the parameter is removed
  (both matrices are recomputed on every call).
- Attributes removed: `method`, `partial_type`, `pdc_a`, `pdc_c`,
  `pdc_mat_A_unbiased`, `pdc_mat_C_unbiased`, `pdc_res_freqs`,
  `pdc_res_power_biased`, `pdc_res_power_unbiased`, `GLS_frequency`,
  `GLS_power`. Results live in `results_frequency` / `results_power`.
- `plot_periodograms` no longer writes a sentinel into `period_truth`.

### Methods and parameters

`Spectrum`

| 0.2.0 | 0.3.0 |
|---|---|
| `Spectrum(wvUnits=...)` | `Spectrum(wave_unit=...)` |
| `InterpolateSpectrum(delta, InterpMethod)` | `interpolate(delta, method)` |
| `TrimSpec(Ntrim, NtrimLeft, NtrimRight, CleanMargins, RemoveNaNs)` | `trim(n_trim, n_trim_left, n_trim_right, clean_margins, remove_nans)` |
| `FilterSpectrum(lowcut, highcut, order)` | `filter_spectrum(continuum_scale_kms, noise_scale_kms, order)` (velocity scales, see Numerical changes) |
| `ApplyCosineBell(alpha)` | `apply_cosine_bell(alpha)` |
| `RemoveCosmics(**kwargs)` (`sigma_upper=`) | `remove_cosmics(sigma_upper=3)` |
| `BarycentricCorrection()` | `barycentric_correction()` |
| `SelectOrders(orders, remove)` | `select_orders(orders, remove)` |
| `SpecPreProccess(Ntrim, CleanMargins, RemoveNaNs, delta, RemCosmicNum, FilterLC, FilterHC, alpha)` | `preprocess(n_trim, clean_margins, remove_nans, delta, cosmic_sigma, continuum_scale_kms, noise_scale_kms, alpha)` |

`native_steps`, `orders` and `from_spectrum1d` are unchanged.

`Template`

| 0.2.0 | 0.3.0 |
|---|---|
| `Template(template=spectrum)` | `Template(spectrum)` |
| `Template(wavelengths=w, spectrum=f[, min_val, max_val])` | `Template.from_arrays(w, f)` |
| `Template(temp, log_g, metal, alpha, min_val, max_val[, air, allow_download])` | `Template.from_phoenix(teff, logg, feh=0.0, alpha=0.0, *, wave_range=(lo, hi), air=False, allow_download=False)` |
| `Template(mask=path, min_val, max_val)` | `Template.from_mask(path, wave_range, allow_download=False)` |
| `Template()` (empty) | removed: pass a `Spectrum` |
| `GaussianBroadening(resolution=[, wv, sp])` | `broaden_instrumental(resolution)` |
| `RotationalBroadening(vsini=, epsilon=)` | `broaden_rotational(vsini, epsilon)` |
| `cut_multiorder_like(obs, margins)` | `cut_multiorder_like(spectrum, margins)` |
| `doppler(vel, wv=[], ...)` | `doppler(vel, wv=None, ...)` |
| `create_PHOENIX_fname(...)` | private (`sparta.template._phoenix_filename`) |
| `download_PHOENIX_files(p, f_n, timeout)` | private; download with `from_phoenix(..., allow_download=True)` |
| `add_noise(snr, sp, rndseed)` | removed: simulate with `sparta.instruments.Simulation` |
| attributes `PHXMASS`, `PHXREFF`, `temp`, `log_g`, `metal`, `alpha`, `min_val`, `max_val`, `server_url`, `wl_f_n` | removed (never read) |

`integrate_spec`, `cut_multiorder` and `save_template_model` are unchanged.
`broaden_instrumental` now returns `self`, like `broaden_rotational`
(`GaussianBroadening` returned the flux array); its unused external `wv`/`sp`
mode is removed. `add_noise` broke on ragged multi-order models, and its only
caller used `add_noise(-1)`, which returned the model flux unchanged.

`CCF1d`

| 0.2.0 | 0.3.0 |
|---|---|
| `CrossCorrelateSpec(spec_in, template_in, dv, VelBound=100, err_per_ord, velocity_convention)` | `cross_correlate(spectrum, template, dv, vel_bound=150, errors_per_order, velocity_convention)` |
| `CombineCCFs()` | `combine()` |
| `extract_RV(x, y, vel, n_ord)` | `extract_rv(x, y, vel, n_ord)` |
| `calcBIS(x, y, bisect_val, n_ord)` | `calc_bis(x, y, bisect_val, n_ord)` |
| `subpixel_CCF(vels, ccf, v, Npts)` | `subpixel_peak(vels, ccf, v)` |
| `plotCCFs(PlotCombined, PlotSingleOrds, ords, alpha, **kwargs)` | `plot(plot_combined, plot_orders, orders, alpha)` |

`CCF2d`

| 0.2.0 | 0.3.0 |
|---|---|
| `CrossCorrelateSpec(spec_in, template1_in, template2_in, dv, VelBound=150, ...)` | `cross_correlate(spectrum, template1, template2, dv, vel_bound=150, ...)` |
| `BestFitSpectrum(order)` | `best_fit_spectrum(order)` |
| `plotTODCOR(levels, marginals, figsize)` | `plot(levels, marginals, figsize)` |

`TIRAVEL`

| 0.2.0 | 0.3.0 |
|---|---|
| `BuildTemplate(spec_list, velocities, ...)` | `build_template(spectra, velocities, ...)` |
| `plotWeights(times, figsize)` | `plot_weights(times, figsize)` |
| `plotTemplate(order, show_epochs, alpha, figsize)` | `plot_template(order, show_epochs, alpha, figsize)` |

`Observations`

| 0.2.0 | 0.3.0 |
|---|---|
| `PreProccessSpectra(Ntrim, CleanMargins, RemoveNaNs, delta, RemCosmicNum, FilterLC, FilterHC, alpha)` | `preprocess(n_trim, clean_margins, remove_nans, delta, cosmic_sigma, continuum_scale_kms, noise_scale_kms, alpha)` |
| `SelectOrders(orders, remove)` | `select_orders(orders, remove)` |
| `calc_rv_against_template(template, dv, VelBound, err_per_ord, combine_ccfs)` | `calc_rv_against_template(template, dv, vel_bound, errors_per_order, combine_ccfs)` |
| `calc_rv_against_two_templates(template1, template2, dv, VelBound, ...)` | `calc_rv_against_two_templates(template1, template2, dv, vel_bound, ...)` |
| `retrieve_BIS(bisect_val, use_combined)` | `calc_bis(bisect_val, use_combined)` |
| `initialize_periodicity_detector(freq_range, periodogram_grid_resolution)` | `init_periodicity_detector(freq_range, periodogram_grid_resolution)` |
| `convert_times_to_relative_float_values()` | private (`_relative_times`), run on ingest |

`build_tiravel_template` and `clean_time_series` are unchanged.
`select_orders` and `init_periodicity_detector` now return `self`.

`PeriodicityDetector`

| 0.2.0 | 0.3.0 |
|---|---|
| `calc_PDC(...)` | `calc_pdc(...)` |
| `calc_USURPER(...)` | `calc_usurper(...)` |
| `calc_GLS()` | `calc_gls()` |

`calc_partial_periodogram`, `calc_pdc_pval` and `plot_periodograms` keep
their names and parameters; `TimeSeries.plot_velocities` loses `sty` (see
the plot style under Other changes).

### Periodogram result keys

Keys of `results_frequency` / `results_power`:

| 0.2.0 | 0.3.0 |
|---|---|
| `'GLS'` | `'GLS'` |
| `'PDC_unbiased'`, `'PDC_biased'` | unchanged |
| `'PDC_unbiased'`, `'PDC_biased'` with `errors=True` (overwrote the plain PDC) | `'PDC_errors_unbiased'`, `'PDC_errors_biased'` |
| `'USURPER'` | `'USURPER_unbiased'` |
| `'USURPER_biased'` | `'USURPER_biased'` |
| `'shift_periodogram'`, `'shape_periodogram'` | `'partial_shift'`, `'partial_shape'` |

### Numerical changes (2026-10-15)

- **TODCOR fits each order on its observed pixels, with templates that
  reach past them.** The templates were shifted circularly within the order
  and both data and templates were tapered (`alpha`); a line shifted off one
  end re-entered at the other, and the taper, moving with the shifted
  template but not with the data, pulled |v| inward. Now `prepare_window`
  resamples each template onto the order's grid *extended* by the largest
  trial shift (`v_max`, fine map included) plus a 20-pixel ramp to zero
  (odd total length), the χ² runs over the observed pixels only, there is no
  taper, and each order has a free continuum offset, profiled out. Observed
  pixels the templates do not reach are dropped, never filled with 1.
  Removed: `alpha`, `continuum` and `zeta_max` (`CCF2d.cross_correlate`,
  `todcor.window_stats`, `todcor_map`, `prepare_window`) and
  `TodcorWindow.taper_power` (now `extension`); `window_stats` returns S11
  and S22 per trial velocity and a genuinely 2-D A12; `model_flux(g1, g2, n,
  dv, v1, v2, a1, a2)` works on the extended templates; `todcor` no longer
  re-exports `tukey_window`, `rfft_positive` or `parseval_weights` (they stay
  in `unicor`). A template's continuum level no longer matters, so TIRAVEL
  templates feed `CCF2d` as they are. `TodcorWindow.reach` (pixels, both
  signs and conventions of ±`v_max`) passed to `window_stats(reach=)`
  refuses a trial shift that would read the ramp.
- **TODCOR's peak is read by a degree-6 polynomial**, not a quadratic patch:
  `crlb_from_map` fits ±4 fine-map nodes (`CCF2d.crlb_half` 3 → 4) and takes
  the maximum by Newton steps and the covariance from the analytic Hessian.
  The quadratic patch misread the narrow ln L peak by up to 6.2 m/s.
  Noise-free bias, analytic line forests at R = 60,000: up to 11.8 m/s (50 Å
  orders) and 1.9 m/s (200 Å) with the old windowing, ≤ 0.023 m/s and
  ≤ 0.016 m/s now; amplitudes exact.
- **Per-order TODCOR amplitudes.** `CCF2d.cross_correlate(amplitudes=
  'per_order')` (and `todcor.profile_per_window`, `joint_fit(amplitudes=)`,
  `window_noise(amplitudes=)`) fits one (a1, a2) per order with shared
  velocities, for a light ratio that changes across the spectrum; `'shared'`
  stays the default. `a1_best` / `a2_best` (and `Observations.amps1/2`) then
  have one value per order, and the per-order amplitude maps are kept only
  with `keep_windows`; χ²_red divides by Σν − 2 − 2·n_orders. A template flat
  to 10⁻⁹ of its own scale in an order is set to zero there and its
  component gets amplitude 0 (it used to fit round-off with an absurd
  amplitude and a spurious velocity); the rule does not depend on the
  templates' units.
- **TODCOR's errors after the band-pass count only the noise it kept**, as
  UNICOR's do: `CCF2d` divides each order's `oversampling` by
  `Info['NoiseFraction']`. On simulated HARPS-like SB2s (7 × 50 epochs) the
  RV scatter over the quoted error was 1.8 after `preprocess` (errors too
  small) and is now 0.86 (primary) / 0.90 (secondary); unfiltered spectra
  are unaffected (0.97 / 1.06).
- Effect of the TODCOR changes: the SB2 golden RVs moved by up to 1.2 m/s
  (primary) and 5.7 m/s (secondary), and their errors fell by ~10% (the
  orders' ends are no longer faded out); against the truth the rms went
  8.1 → 7.5 m/s and 11.7 → 7.6 m/s. The atoms-baseline SB2 moved by +2.2 m/s
  and −28 m/s (truth offsets +8.5 → +10.6 m/s and +47.5 → +19.6 m/s), and
  with `sigma=None` its errors fell from 8.3 to 5.1 m/s, now equal to those
  at the true sigma: the old residual carried misfit. Tutorial 04's orbit
  residuals: primary 4.5 → 3.5 m/s, secondary 8.2 → 6.5 m/s.
- **No taper in preprocessing; the band-pass filters the order's ends
  cleanly.** `Spectrum.preprocess` no longer multiplies each order by a Tukey
  window, and `apply_cosine_bell` and `alpha` are removed (`preprocess`,
  `Observations.preprocess`, the instrument recommendations). The taper hid
  the filter's start-up, which with 30 guessed padding pixels spoiled
  ~continuum_scale/2π (~300 km/s) at each end; but, applied to a template
  before it is shifted, it biased UNICOR's RVs by up to ~1 m/s and doubled
  their quoted errors. `bandpass` now removes the order's straight-line trend
  and filters with Gustafsson's initial conditions (`filtfilt(method='gust')`).
  Effect: NRES TOI-677 RV scatter 75 → 70 m/s; HARPS TOI-193 14.1 → 14.2 m/s;
  simulated HARPS-like bias +0.03 m/s, scatter 1.75 → 1.70 m/s. Goldens: NRES
  recipe RVs +21 and +30 m/s, HARPS TIRAVEL library RVs +4.9 m/s.
- **UNICOR's errors after the band-pass count only the noise it kept.**
  New atom `sparta.preprocess.filtered_noise_fraction`; `filter_spectrum`
  records it as `Info['NoiseFraction']`; `unicor.ccf_order` takes
  `noise_fraction` and scales `n_eff` by it. Without it the quoted errors
  were 0.6–0.7× the scatter once the taper no longer depressed the CCF peak;
  with it they err high, 1.25–1.4× on simulated spectra (and ~3× on the real
  HARPS TOI-193 RVs, with template mismatch). Approximate: an exact error
  model for filtered spectra is an open item (docs/roadmap.md). NRES median
  eRV 24 → 58 m/s, HARPS 27.8 → 46.2 m/s.
- **The RV is the peak of the continuous CCF**, not the vertex of a 5-point
  parabola: `unicor.subpixel_peak` maximises a quintic spline through ±4 lags
  around the sampled maximum, and `rv_uncertainty` takes C and C″ from the
  same spline (it used a global quartic spline). The parabola misread the
  peak by up to 22 m/s at 0.75 km/s lags (R = 100,000, noise-free) depending
  on where it fell between lags; the spline is within 0.12 m/s there and
  0.01 m/s at 0.5 km/s, so the RV no longer depends on the step. With noise
  (S/N 80) the RV scatter at 0.75 km/s lags falls from 16.3 to 3.7 m/s,
  matching the quoted 3.4 m/s; at 0.1 km/s the two are identical. A peak
  spanning fewer than 5 lags above half maximum raises a `UserWarning`.
  `subpixel_peak` and `CCF1d.subpixel_peak` lose `npts`. Goldens: NRES RVs
  (0.05 km/s lags) −0.002 to −0.003 m/s; HARPS TIRAVEL ≤ 0.03 m/s; the
  simulated SB1 (0.1 km/s, S/N 80) up to 0.9 m/s per epoch, 0.36σ, as the two
  estimators read the same noise differently — against the truth both give
  rms 2.5 m/s and χ²/N 1.0.
- **The band-pass cuts are velocity scales.** `FilterLC` / `FilterHC` (a
  multiple of 1/span and a fraction of the pixel Nyquist frequency) are now
  `continuum_scale_kms` / `noise_scale_kms`, the periods of the band edges in
  km/s, converted at each order's median wavelength. The same setting is now
  the same physical filter on any pixels: at a 4 km/s scale the old HARPS
  setting passed 0.50 of the amplitude on a 0.005 Å grid at 5000 Å, 0.03 on a
  0.02 Å grid and 0.29 on a 0.01 Å grid at 6500 Å, so a template on finer
  pixels than the data was filtered differently. The recommendations keep the old values' meaning on the
  bundled data: HARPS 3 / 0.15 → 2000 / 4 km/s (they were 1978 / 3.97), NRES
  4 / 0.15 → 1400 / 9.6 km/s (they were 1129–1422 / 9.6, varying with each
  order's span); the `preprocess` defaults are HARPS's. Effect: the NRES
  TOI-677 RVs of how-to 01 moved by +6 to +16 m/s (mean +10), their rms (75
  m/s) and errors (median 24 m/s) unchanged, and tutorial 03's recipe, whose
  template sits on other pixels, now gives the same mean RV (38.182 km/s;
  the two disagreed by 7 m/s before); the NRES golden recipe, whose
  template was on finer pixels, moved by −19.5 m/s with 8% smaller errors;
  HARPS TOI-193 RVs: rms 14.1 m/s unchanged, median error 28.2 → 27.8 m/s;
  the TIRAVEL golden (HARPS, half the epochs): library RVs +1.7 m/s on
  average (at most 2.8 m/s), ρ 0.99683 → 0.99685.
- **The partial periodograms shift each spectrum exactly to rest**,
  λ / (1 + v/c), instead of λ·(1 − v/c), which left a residual shift of
  ≈ v²/c (33 m/s at 100 km/s) that the shape periodogram partly detected.
  `calc_partial_periodogram` takes `velocity_convention` ('cz' default) to
  invert the convention the RVs were measured in. The golden partial
  periodograms moved by up to 3.1e-4 (shape) and 4.2e-5 (shift) in power; a
  pure-Doppler test with K = 30 km/s, where the shape distance should vanish,
  moved by up to 0.023.
- **`Spectrum.interpolate(method='linear')` takes a length as a length.**
  `delta=0.25 * u.AA` is a 0.25 Å step (it was 0.25 × the median step); a
  plain number is still a multiple of the median step; a velocity or another
  unit raises `ValueError`. `Info['GridUnits']` is `'median step'` or
  `'wavelength'` accordingly.

### Other changes

- **One normalisation:** `unicor.dsp.standardize` (zero mean, unit variance)
  ends the band-pass filter and builds USuRPER's unit vectors
  (`standardize(f)/√n`); `unicor.normalize` (mean-subtract and unit length) is
  removed. Results change only by rounding (the golden master passes at 1e-9;
  the byte baseline is re-blessed). TIRAVEL keeps its joint unit-length
  scaling across orders on purpose (docs/conventions.md).
- **Downloaded PHOENIX models go to a user cache**, not into the package:
  `$SPARTA_CACHE_DIR/phoenix`, else `$XDG_CACHE_HOME/sparta/phoenix`, else
  `~/.cache/sparta/phoenix` (`SPARTA_CACHE_DIR` must be absolute). The four
  shipped files are read in place and never written (a corrupt one raises);
  a lookup creates no directory; downloaded files are 0644. Models an earlier
  version downloaded into `template_files/` are no longer read there: move
  them to the user cache, or download them again.
- `calc_partial_periodogram` raises `ValueError` naming the epochs whose RV is
  not finite (a NaN used to reach the distance matrices); a `method='linear'`
  step that is zero, negative, non-finite or an array raises `ValueError`.
- One `vel_bound` default, 150 km/s, in `CCF1d.cross_correlate` (was 100),
  `CCF2d.cross_correlate` and both `Observations` RV drivers. It equals the
  default margin of `Template.cut_multiorder_like`, so a template cut with the
  defaults covers every observed pixel at every lag.
- `Spectrum.interpolate` raises `ValueError` for an unknown `method` (was a
  `NameError`); `Template.from_phoenix` / `from_mask` raise `ValueError` for a
  `wave_range` off the PHOENIX grid (was an `IndexError`); `Template(x)` raises
  `TypeError` unless `x` is a `Spectrum`.
- `Template.from_mask` reads the mask with `numpy.loadtxt` (three columns:
  start, end, weight; `#` comments).
- The instruments' recommended preprocessing (`Recommended.preprocess`,
  `docs/instruments.md`) uses the new `preprocess` keywords.
- `Observations.calc_rv_against_template` and
  `calc_rv_against_two_templates` take `velocity_convention` ('cz' default,
  or 'relativistic') and pass it to `CCF1d` / `CCF2d`; the SB1 driver had no
  way to pass it, the SB2 one only through `**kwargs`.
- `Spectrum(wv, sp)` with different numbers of orders, and
  `Spectrum.select_orders` / `Observations.select_orders` with an index
  outside 0 .. n_ord - 1, raise `ValueError` (was `AssertionError`, skipped
  under `python -O`).
- Mutable default arguments are gone: `Spectrum(wv=None, sp=None)`,
  `Observations(time_series=None)`, `Template.integrate_spec(wv=None,
  sp=None)`, `Observations.calc_bis(bisect_val=(0.35, 0.95))`. Calls that
  pass these arguments behave as before.

- **Changes results on NRES:** `Spectrum.trim(clean_margins=True)`, the
  `preprocess` default, now removes padded order margins. These are the
  leading and trailing runs whose running median is below `margin_fraction`
  (default 0.1) of the order's median flux, so noisy unexposed detector is
  caught as well as exact zeros. Before, only exact zeros were removed, so
  the NRES margins survived. The band-pass filter then turned the jump from
  about 0 to the continuum into a step at the order edge. `preprocess` order:
  trim margins, resample, remove cosmics, remove the continuum by filtering,
  taper. Margins are now found after NaNs are dropped, and the last exposed
  pixel is kept (off-by-one fixed). `interpolate` raises a clear error for an
  empty order. Effect on the bundled NRES TOI-677 data: RV scatter 82 -> 75
  m/s, median eRV 26 -> 24 m/s.
- tqdm is no longer a required dependency and progress bars are off by
  default. `Observations.preprocess`, `calc_rv_against_template` and
  `calc_rv_against_two_templates` take `progress=False`; `progress=True`
  needs tqdm (`pip install tqdm`; the extra is `pip install ".[progress]"` from a
  checkout -- never `"SPARTA[progress]"`, which on PyPI is an unrelated package).
- `Observations` reads files in sorted order, so the epoch order no longer
  depends on the filesystem.
- **Two output modes, `output='ccf'` (default) or `output='loglike'`,** in
  `unicor.ccf_vs_velocity` / `combine_ccfs`, `todcor.profile_from_stats` /
  `todcor_map`, `CCF1d.cross_correlate`, `CCF2d.cross_correlate` and both
  `Observations` RV drivers; any other value raises `ValueError`. UNICOR's
  log-likelihood is Zucker's (2003) `-(N_i/2) ln(1 - C_i^2)` per order with
  the effective bins `n_eff`, summed over orders (`= -(N_tot/2) ln(1 - ML^2)`;
  new `unicor.loglike_from_ccf`); `CCF1d.Corr` / `CorrCombined` store it as
  `'loglike'` next to `'corr'`. TODCOR's correlation is Zucker & Mazeh's
  (1994) two-dimensional `R = sqrt(1 - chi2_min / SD)`, from the same
  sufficient statistics (`res['R']`, `best['R']`, `Corr2d['R']`,
  `Corr2d['R_best']`); its log-likelihood is the Gaussian `-chi2/2` as
  before. Both surfaces are always stored; the mode is recorded (`'output'`),
  selects TODCOR's primary `'map'` and what `plot()` draws. RVs, errors and
  amplitudes are identical in both modes (the errors still come from the
  log-likelihood). **Default change:** `CCF2d`'s primary map and plot are now
  `R`, not Delta lnL — pass `output='loglike'` for the old surface.
- One plot style for every figure, in the new `sparta.plotting` module
  (exported): data as thin black mid-centred steps (`plot_data`), scattered
  epochs as black points with error bars (`plot_points`), models red
  (`plot_model`), one extra series blue (`plot_extra`), derived curves such as
  CCFs and periodograms black lines (`plot_curve`), maps as black unfilled
  maps as grey-scale images with the peak black and a thin grey colour bar
  (`map_image`, `map_colorbar`; a correlation map shaded on its own scale,
  white at zero and black at the peak with a gamma = 2 stretch, a
  log-likelihood map as Delta lnL on a symmetric-log scale, linear to -4.5
  and logarithmic beyond, `dlnl_norm`), best fit a red `+`, truth a blue
  `x`, both white-edged so they read on black; `apply_style()` sets the
  matching rcParams. `CCF2d.plot` loses `levels` (no contours, no colour
  map). `TimeSeries.plot_velocities`
  loses `sty` and returns the figure instead of showing it;
  `plot_periodograms` returns the figure. All the example notebooks use the
  style.
- **New `sparta.preprocess`** (additive): the per-order steps of `Spectrum` as
  functions on plain arrays (wavelengths in Angstrom, velocities in km/s) --
  `trim_margins`, `resample_to_linear_grid`, `resample_to_log_grid`,
  `remove_cosmics`, `bandpass`, `coarsest_native_step` (the native-step rule of
  `interpolate`, `Observations` ingestion and `Template.integrate_spec`) and
  `resample_flux` (moved from `sparta.spectrum`, still importable there). The
  `Spectrum` methods are loops over them, bit-for-bit as before (pinned by
  `tests/test_atoms_baseline.py`). New `sparta.template.phoenix_spectrum`: the
  PHOENIX model as `(wave, flux)` arrays, which `Template.from_phoenix` wraps.
- **Per-order UNICOR atoms** (additive; `sparta-unicor` 0.2.0): `unicor.lag_axis`,
  `ccf_order` (returns `CCFOrder`), `template_coverage`, `effective_bins` and
  `measure_rv` (returns `RVMeasurement`) -- one order's CCF, log-likelihood, RV,
  uncertainty and `n_eff` on plain arrays (km/s, Angstrom). `CCF1d.cross_correlate`
  is a loop over `ccf_order`, and `combine` / `extract_rv` read the RV with
  `measure_rv`, bit-for-bit as before (pinned by `tests/test_atoms_baseline.py`).
  A malformed template (wavelengths not strictly increasing, or not one per flux
  value) raises `ValueError` naming the template order.
- **Per-window TODCOR atoms** (`sparta-todcor` 0.2.0): `prepare_window`
  (returns `TodcorWindow`: the data residual and the two templates on the
  window's grid extended by the largest shift), `refine_grid` (the fine local map), `joint_fit` (returns
  `TodcorFit`: pool, coarse map, fine map, CRLB and the values at the refined
  peak; `fine_cache` shares the fine statistics between passes) and
  `window_noise` (each window's noise from its share of the joint chi2).
  `CCF2d.cross_correlate` is a composition of them; the
  resampled-noise model is public as `sparta.preprocess.resampling_noise_psd` and
  `native_phases`. `CCF2d` prefixes `order {I}: ` to window-preparation errors.
- **TIRAVEL and PDC p-value atoms** (additive): `sparta.tiravel_ops.build_tiravel(waves,
  fluxes, velocities, orders=None, reference=0, margin=10, kind='cubic')` returns
  `TiravelResult` (`rho`, `weights`, `correlation`, `diagnostics`, `velocities`,
  `grids`, `template_flux`, `aligned`): the whole template build on per-epoch,
  per-order arrays (km/s, Angstrom). `TIRAVEL.build_template` wraps it in a
  `Template`, bit-for-bit as before. `sparta.pdc.pdc_pvalue(power, n)` and
  `pdc_power_level(pvalue, n)` give the Shen et al. (2019) single-frequency
  p-value of an unbiased PDC power and its inverse, vectorised.
  `PeriodicityDetector.calc_pdc_pval` calls them, bit-for-bit as before.
- **A lazy front door and the open surface** (additive): every public atom of
  `sparta.preprocess`, `sparta.pdc` and `sparta.tiravel_ops`, and
  `phoenix_spectrum`, imports from `sparta` itself
  (`from sparta import resample_to_log_grid, build_tiravel, pdc_power`), the
  same objects as in their modules; the `unicor`/`todcor` atoms stay in those
  packages. Every top-level name resolves on first access (PEP 562), so
  `dir(sparta)` is `sparta.__all__`; submodules still resolve after a bare
  `import sparta` (`sparta.pdc.pdc_power`), now on first access. matplotlib is imported by the plot methods, specutils by
  `resample_flux` and the `Spectrum` <-> specutils conversions, not at import:
  `import sparta` went from ~0.8 s (1.2 s cold) and ~1840 modules, matplotlib
  and specutils among them, to ~0.1 ms; importing the atoms takes ~0.4 s and
  the objects ~0.5 s, still without either. `docs/open_api.md` lists the
  atoms of SPARTA, UNICOR and TODCOR with their import path, units and
  assumptions, checked by `tests/test_open_api.py`.
- `TIRAVEL.plot_template` draws the template divided by the sum of the weights
  (the weighted mean of the aligned epochs), on the epochs' scale. The stored
  co-add is ~sqrt(K) larger, which made the epochs look like a shrunken copy.
  The y-axis reads "Flux (mean-subtracted, unit-norm epochs)". The stored
  template is unchanged.

- **Template broadening on a ln λ grid (changes results).**
  `Template.broaden_instrumental` / `broaden_rotational` used each order's
  median velocity step, so on a grid not uniform in ln λ the kernel had the
  wrong width away from the median, by the ratio of the local to the median
  step. PHOENIX HiRes is linear in λ with a step of 0.006 Å below 5000 Å and
  0.01 Å above (0.36 vs 0.60 km/s), so the error reaches ~1.7x (or ~0.6x)
  when the median falls on one side of 5000 Å (as in the new tests); over
  the shipped recipes' ranges it was 0.8–1.35x, and for the NRES golden
  template (a uniform 0.03 Å linear grid, 4500–7700 Å) it came from the 1/λ
  drift, 1.36x at the blue end to 0.79x at the red. Such an order is now resampled
  onto ln λ at its finest velocity step (cubic), broadened there, and
  resampled back onto its own wavelengths; the template keeps its grid. A
  ln λ order is broadened exactly as before. Re-pinned: the NRES UNICOR
  golden RVs move by −10.7 and −13.1 m/s (≈ 0.5σ; errors 2% smaller); the
  TOI-193 TIRAVEL golden by ≤ 0.06 m/s (ρ by 1.5e-9). The resampling round
  trip alone moves those RVs by ≤ 0.13 m/s. The 394 byte baselines do not
  broaden and are unchanged.
  An order that would take the new path is checked first: wavelengths
  strictly increasing, flux finite, and a finest step that does not blow the
  ln λ grid up past 4x the order's pixels (a near-duplicate wavelength in a
  stitched spectrum); each raises a clear `ValueError`.
- Tested on Python 3.10 (numpy 2.2, scipy 1.15, astropy 6.1, specutils 2.1)
  and on 3.12 and 3.13 with the then-latest releases (numpy 2.5, scipy 1.18,
  astropy 8.0, specutils 2.4, matplotlib 3.11): every suite and notebook
  passes. The byte-identity baselines skip, by design, outside the
  environment that recorded them.
- `SPARTA` requires `sparta-unicor>=0.2` and `sparta-todcor>=0.2` (it uses
  their per-order and per-window atoms); `>=0.1` let pip keep an older engine.
- **Security hardening of the PHOENIX download** (the former FTP connection;
  HTTPS with astropy's default certificate checks since July). Downloads are
  copied into a temporary file in the cache directory, fsynced, verified to be
  FITS with data, and only then moved into place with `os.replace`. An
  interrupted or failed download therefore never leaves a truncated model that
  later runs would trust. A corrupt cached file is reported (never silently
  deleted) when downloads are off, and replaced when they are on. Model
  parameters must be finite numbers (Teff > 0, log g >= 0) and are checked
  before any file access. The cache path must be a bare file name inside the
  cache directory.
- Install lines that would make pip fetch the unpublished engine names from
  PyPI (a dependency-confusion risk) now install the local engines in the same
  call, and the README warns never to run `pip install .` on its own while
  the engines are unpublished.

### Documentation

- The tutorials `examples/00`–`08` are rewritten the open way: every step is
  a visible atom call on plain arrays, and the objects appear only in a final
  cell that checks they give the same numbers (bit for bit wherever the code
  paths coincide) and points to the matching quickstart. Notebook 02 rebuilds
  the TOI-677 (NRES) RVs, and 03 a two-order TODCOR fit, from atoms alone.
  The quickstarts keep the objects. Gaps the rewrite met in the atom surface
  are listed in `docs/roadmap.md`.
- A new README: what SPARTA does (PDC, error-aware PDC, USuRPER and the
  partial periodograms are now described; the unimplemented "detection of
  systematic shifts" claim for UNICOR is gone), an offline quickstart on
  simulated HARPS data, the conventions (units, 'cz' vs 'relativistic',
  barycentric composition, log-λ grid, native-pixel error bars, uncalibrated
  errors after `preprocess`, air/vacuum), a module map and the papers to cite.
  Links are absolute, so they work on PyPI.
- `CONTRIBUTING.md`: layout, development install, tests, engine isolation,
  the golden-master policy, the add-an-instrument checklist and the code
  style.
- The engine READMEs' quickstarts are self-contained and noisy, with their
  output shown; the TODCOR one no longer broadens the observed composite.
- Every public function and method of the object layer has a numpydoc
  docstring with units; the docstrings that contradicted the code are fixed
  (e.g. `Spectrum.interpolate`'s step is a multiple of the *median*
  wavelength step).
- The bisector span (`unicor.bisector_span`, `CCF1d.calc_bis`,
  `Observations.calc_bis`) is documented as known to be incorrect and under
  review; it is unchanged.

## 0.2.0

Modernisation of the 2020 code base for the current scientific Python stack,
and correctness fixes that **change results**.

- **Packaging.** The correlation engines are standalone packages,
  `sparta-unicor` (`unicor`) and `sparta-todcor` (`todcor`); SPARTA sits on top
  in a `src` layout with `pyproject.toml`. Python >= 3.10, astropy >= 6,
  specutils >= 2. The bundled PHOENIX templates ship in the wheel.
- **Dependencies.** pandas, PyAstronomy, numba and easygui are dropped.
- **Velocity convention.** RVs are 'cz' by default
  (lambda_obs / lambda_rest = 1 + v/c, the IAU measure and IRAF fxcor's), with
  `velocity_convention='relativistic'` as an option, in UNICOR, TODCOR and the
  simulator. UNICOR used to report lag k as k * dv, off from cz by -v^2/2c
  (-37.5 m/s at 150 km/s). The barycentric correction stays
  lambda * (1 + bcv/c) whichever convention is used.
- **UNICOR v = 0 bias removed.** The template is correlated on the observed
  grid extended by the lag range, each lag a Pearson correlation over the same
  pixels. The old tapered CCF pulled every order's peak toward v = 0
  (18 m/s per 60 A order at vsini 10, up to ~1 km/s on short orders). An
  order the template does not cover is NaN, with a warning.
- **UNICOR eRV with native bins.** The ML uncertainty (Zucker 2003) counts
  each order's independent spectral bins, `n_pix * min(1, dv / native_step)`,
  instead of (lags x orders); orders are combined with those weights. Errors
  no longer depend on the scan width (they were up to 3.4x too large).
  `Spectrum.native_steps()` / `Info['NativeStep']` record the native sampling
  through every resampling, and archive-resampled products (e.g. HARPS ADP)
  are seeded with the detector step.
- **TODCOR.** The CRLB counts native pixels (it was ~sqrt(10) too small on the
  default grid); the peak is refined on a fine local map (the quadratic fit
  was biased by 2-5 sigma); `sigma=None` is the default and estimates each
  order's noise from its own residual in a second, re-weighted pass.
- **Readers.** `ReadSpec` is replaced by `sparta.instruments`: one module per
  instrument, an explicit registry keyed by survey, typed `LoadedSpectrum`
  records and a generated `docs/instruments.md`. `Observations` requires
  `survey`. Wavelength solutions and frames are fixed per instrument (LAMOST,
  ELODIE, TRES, UVES, APOGEE, ...); unknown snr / air_mass are NaN and an
  unknown vrad is None; `HARPS_15` / `FEROS_15` are now `HARPS-CERES` /
  `FEROS-CERES`; a directory whose visits disagree on time system or
  wavelength frame is refused. A `SIMULATED` survey builds synthetic SB1/SB2
  data with known truth through the same path.
- **Downloads are opt-in.** PHOENIX files come from the local cache;
  fetching one (over HTTPS, no longer FTP) needs `allow_download=True`.
- **Guards instead of silent garbage.** Unbiased U-centering raises for
  n <= 3; a frequency grid must start above 0; USuRPER refuses multi-order or
  non-overlapping spectra; partial periodograms no longer shift the caller's
  spectra in place; a CCF peak at the scan edge is NaN instead of wrapping
  around; non-uniform "log" grids, NaN pixels and line-free templates are
  refused; `clean_time_series` filters every per-epoch array with one mask.
- **Smaller fixes.** `RemoveCosmics` interpolates clipped pixels instead of
  deleting them (which broke the uniform grid); both broadenings act on every
  order; PHOENIX names for [M/H] < 0 with alpha enhancement are correct;
  `save_template_model` writes a per-order `.npz` that round-trips (the CSV
  could not be read back); the no-op `fastccf` parameter is removed.
