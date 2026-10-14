# Changelog

Units throughout: velocities in km/s, wavelengths in Angstrom, times in days,
periodogram frequencies in 1/day.

## 0.3.0 (unreleased)

The public API of the object layer is now snake_case, the package layout is
flat, and `Observations` is the only time-series driver. **There are no
aliases or shims**: every old name below is gone and raises `ImportError`,
`AttributeError` or `TypeError`. Numerical results are unchanged; the golden
master (UNICOR, TODCOR, TIRAVEL, periodograms, instrument readers) passes
bit-for-bit or at 1e-9 relative. The `unicor` and `todcor` engine APIs did not
change.

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
| `FilterSpectrum(lowcut, highcut, order)` | `filter_spectrum(lowcut, highcut, order)` |
| `ApplyCosineBell(alpha)` | `apply_cosine_bell(alpha)` |
| `RemoveCosmics(**kwargs)` (`sigma_upper=`) | `remove_cosmics(sigma_upper=3)` |
| `BarycentricCorrection()` | `barycentric_correction()` |
| `SelectOrders(orders, remove)` | `select_orders(orders, remove)` |
| `SpecPreProccess(Ntrim, CleanMargins, RemoveNaNs, delta, RemCosmicNum, FilterLC, FilterHC, alpha)` | `preprocess(n_trim, clean_margins, remove_nans, delta, cosmic_sigma, filter_lowcut, filter_highcut, alpha)` |

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
| `subpixel_CCF(vels, ccf, v, Npts)` | `subpixel_peak(vels, ccf, v, npts)` |
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
| `PreProccessSpectra(Ntrim, CleanMargins, RemoveNaNs, delta, RemCosmicNum, FilterLC, FilterHC, alpha)` | `preprocess(n_trim, clean_margins, remove_nans, delta, cosmic_sigma, filter_lowcut, filter_highcut, alpha)` |
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

`calc_partial_periodogram`, `calc_pdc_pval`, `plot_periodograms` and
`TimeSeries.plot_velocities` are unchanged.

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

### Other changes

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
