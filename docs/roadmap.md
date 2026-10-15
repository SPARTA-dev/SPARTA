# Roadmap (proposed)

> **This is a proposal for the maintainers to edit, not a list of
> commitments.** Nothing here is scheduled. Each item names the motivation;
> most answer an entry in [limitations.md](limitations.md).

## Radial velocities

- **An exact error model for filtered spectra (open issue, 2026-10-15)** —
  after the band-pass, UNICOR and TODCOR count only the share of native noise
  the filter kept (`filtered_noise_fraction`), which turns errors that were
  0.6–0.7× the scatter into 1.25–1.4× on simulated HARPS- and NRES-like
  spectra (UNICOR), and ~0.55× into 1.1–1.2× on simulated SB2s (TODCOR). The remaining excess comes from using one average:
  f counts noise only, while the velocity information sits in the line slopes,
  which the filter mostly keeps (the noise cut sits at ~1.6× the line FWHM),
  so the correction discounts the information more than the filter does
  ([conventions](conventions.md#after-a-band-pass-the-kept-noise-fraction-f));
  on the real HARPS TOI-193 RVs template mismatch adds to
  it (quoted 46 m/s for a 14 m/s scatter). Derive the error of a CCF of
  filtered spectra from the filter's response to both signal and noise, test
  it by Monte Carlo across instruments and noise cuts, and extend it to
  TODCOR.

- **TIRAVEL velocity optimiser** — maximise the alignment measure ρ over the
  epochs' velocities, the half of Zucker & Mazeh (2006) that is still missing;
  today ρ is only evaluated at supplied velocities. The plan and what it
  needs: [tiravel.md](tiravel.md#development).
- **Fix the bisector span** — compute it from the velocities of the level
  crossings, choose and document the convention, give it an honest error and
  a working per-order path, then pin it in the golden master.
- **Template-pair search** — scan template grids (temperature, gravity,
  rotation) for both components of an SB2 by the TODCOR correlation R: the
  "analysis of composite spectra" planned in the original README, beyond
  fitting a fixed pair.
- **Filter on the ln λ grid** — apply the band-pass after resampling onto
  the CCF's ln λ grid, where a velocity scale is exact everywhere, and drop
  the intermediate linear resampling (one interpolation fewer).
- **The noise cut against the resolution** — the recommended noise scale is
  ~1.6 × FWHM (c/R), where a line keeps only ~0.25–0.3 of its amplitude per
  Fourier mode; measure the RV precision and the error calibration as a
  function of `noise_scale_kms` / FWHM.
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
- **TODCOR's per-window noise setup** (`sigma`, `oversampling`,
  `resampling_noise_psd`) and the two-pass fit with
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
