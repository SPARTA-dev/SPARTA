# Roadmap (proposed)

> **This is a proposal for the maintainers to edit, not a list of
> commitments.** Nothing here is scheduled. Each item names the motivation;
> most answer an entry in [limitations.md](limitations.md).

## Radial velocities

- **TIRAVEL velocity optimiser** — maximise the alignment measure ρ over the
  epochs' velocities, the half of Zucker & Mazeh (2006) that is still missing;
  today ρ is only evaluated at supplied velocities. The plan and what it
  needs: [tiravel.md](tiravel.md#development).
- **TODCOR beyond the window edges** — extend the templates past each window
  (or zero-pad) so the circular Fourier shift stops biasing short,
  line-crowded windows (15.6 m/s at 50 Å in tutorial 04).
- **Per-window TODCOR amplitudes** — profile (a1, a2) per order and sum the
  log-likelihood maps, for light ratios that change with wavelength.
- **A better sub-pixel peak** — a spline or other non-parabolic peak estimate
  for UNICOR, removing the 5-point parabola's pixel-phase error on coarse
  grids.
- **Fix the bisector span** — compute it from the velocities of the level
  crossings, choose and document the convention, give it an honest error and
  a working per-order path, then pin it in the golden master.
- **Template-pair search** — scan template grids (temperature, gravity,
  rotation) for both components of an SB2 by the TODCOR correlation R: the
  "analysis of composite spectra" planned in the original README, beyond
  fitting a fixed pair.
- **Templates with any continuum in TODCOR** — fill uncovered pixels with the
  template's own continuum, so TIRAVEL templates can feed `CCF2d`.
- **A band-pass cut in physical units** — `filter_lowcut` / `filter_highcut`
  in 1/Å or s/km instead of the order's span and pixel Nyquist frequency, so
  template and data are filtered alike whatever their pixel scales.
- **A wavelength `delta` for `Spectrum.interpolate`** — read a length
  Quantity as the step itself, or raise on one (either is a behaviour
  change, so a deliberate one).
- **Both broadening kernels in one call** — `broaden_rotational` and
  `broaden_instrumental` each resample a non-ln λ template onto ln λ and back;
  one call applying both would halve the interpolations.

## Periodograms

- **Significance for the partial periodograms** — a built-in permutation test
  (as in tutorial 08), since the χ² p-value is not established there; a global
  false-alarm estimate for all PDC-type periodograms.
- **Record the velocity convention with the RVs**, so that
  `calc_partial_periodogram` inverts the one they were measured in by
  default; today it assumes 'cz' unless told (`velocity_convention=`), and
  relativistic RVs shifted as 'cz' keep ≈ v²/2c (37 m/s at 150 km/s).
- **Error-weighted GLS** — pass the RV errors to `LombScargle`.

## Instruments and data

- **Verify FEROS and UVES against real archive files**, then mark them
  verified and pin them in the golden master.
- **Native steps for the other archive-resampled products**, so their errors
  count detector pixels.
- **BJD conversion** — convert each product's time system to BJD-TDB (with
  the observatory and target coordinates) instead of only recording it.
- **Migrate `Spectrum` to specutils** — `Spectrum1D` / `SpectrumCollection`
  with units on the spectral axis as the working store, replacing the `Info`
  dict.
- **PHOENIX alpha with [M/H] > 0**: the file-name builder silently ignores
  `alpha` when `feh > 0` (the grid has no such models); raise instead.

## The open surface

Gaps met while writing the open tutorials (each notebook works around them
with a few lines of its own):

- **Velocity-even broadening as an atom** — `broaden_on_log_grid(wave, flux,
  kernel)` (today the private `sparta.template._broaden_velocity_even`), for
  a model on a linear-λ grid such as raw PHOENIX.
- **One RV per single-order epoch** matching `Observations` bit for bit
  (`lag_axis` → `ccf_order` → `combine_ccfs` → `measure_rv`); the tutorials
  repeat a short loop.
- **The partial periodograms' shift and shape distances** as an atom; the
  composition (shift to rest by λ / doppler_factor(v), then the chord distance) lives only
  in `PeriodicityDetector.calc_partial_periodogram`.
- **The cut of `Template.cut_multiorder_like`** (the template range around
  each order, with a velocity margin) as an atom.
- **TODCOR's per-window noise setup** (`sigma`, `zeta_max`, `oversampling`,
  the tapered `resampling_noise_psd`) and the two-pass fit with
  `window_noise` as one atom; today that knowledge lives inside `CCF2d`.
- **The whole per-order preprocessing chain** on arrays, with its
  native-step bookkeeping, which is easy to get wrong by hand.
- **The simulator's array functions** (`simulate`, `component_spectra`,
  `upper_envelope` in `sparta.instruments.simulated`) on the open surface.
- Small: `Product.load` could default `min_wv`/`max_wv` for products that
  ignore them; `unbiased_u_centering(a, n)` takes `n` although it is `len(a)`.

- **Input checks on the preprocessing atoms** — `remove_cosmics` on an
  all-NaN order or a descending axis, and `resample_to_*_grid` on 2–3 pixel
  orders (which return empty arrays); `ccf_order` / `template_coverage` with a
  non-integer `n_lags`.

## Packaging and infrastructure

- **Import `todcor` only where needed** — `sparta.preprocess` imports it at
  module level for `resampling_noise_psd` alone.
- **Release the engines on PyPI** as `sparta-unicor` and `sparta-todcor`, and
  register both names first: `SPARTA` depends on them by name, and plain
  `unicor` and `sparta` on PyPI are unrelated projects.
- **Continuous integration** — run the three test suites, the slow notebook
  tests and the generated-docs checks (`docs/instruments.md`, `docs/api.md`)
  on every push.
