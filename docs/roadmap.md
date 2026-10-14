# Roadmap (proposed)

> **This is a proposal for the maintainers to edit, not a list of
> commitments.** Nothing here is scheduled. Each item names the motivation;
> most answer an entry in [limitations.md](limitations.md).

## Radial velocities

- **TIRAVEL velocity optimiser** — maximise the alignment measure ρ over the
  epochs' velocities, the half of Zucker & Mazeh (2006) that is still missing;
  today ρ is only evaluated at supplied velocities.
- **TODCOR beyond the window edges** — extend the templates past each window
  (or zero-pad) so the circular Fourier shift stops biasing short,
  line-crowded windows (15.6 m/s at 50 Å in notebook 03).
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
  Quantity as the step itself (a behaviour change, so a deliberate one).

## Periodograms

- **Significance for the partial periodograms** — a built-in permutation test
  (as in notebook 08), since the χ² p-value is not established there; a global
  false-alarm estimate for all PDC-type periodograms.
- **Exact rest-frame shift** in the shape distance, λ/(1 + v/c).
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
- **A user cache for PHOENIX downloads** outside the package directory.

## Packaging and infrastructure

- **Release the engines on PyPI** as `sparta-unicor` and `sparta-todcor`, and
  register both names first: `SPARTA` depends on them by name, and plain
  `unicor` and `sparta` on PyPI are unrelated projects.
- **Continuous integration** — run the three test suites, the slow notebook
  tests and the generated-docs checks (`docs/instruments.md`, `docs/api.md`)
  on every push.
