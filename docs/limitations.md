# Known limitations

What SPARTA does not do, or does only approximately, as of 0.3.0. Each item
gives its size where it has been measured and points to the page or notebook
with the details. Proposed fixes are collected in [roadmap.md](roadmap.md).

## UNICOR

- **The bisector span is wrong.** `unicor.bisector_span`, `CCF1d.calc_bis`
  and `Observations.calc_bis` build the bisector from CCF *values* at the
  level crossings instead of the crossing *velocities*, so the result is not
  a velocity span; `Observations.calc_bis(use_combined=False)` raises
  `AttributeError`. It is pinned, unchanged, for a later fix. Do not use it.
  ([correlation_engines.md](correlation_engines.md#bisector-span-known-to-be-incorrect))
- **An undersampled CCF peak.** The RV is the maximum of a quintic spline
  through the CCF's lags, which stands for the continuous CCF as long as the
  peak is sampled by about 5 lags or more across its half width: on noiseless
  R = 100,000 line forests the error over sub-lag phases is ≤ 0.12 m/s at
  dv = 0.75 km/s (step / FWHM 0.18). Past that it degrades fast: 1.2 m/s at
  step / FWHM 0.24, 17 m/s at 0.37. `unicor.subpixel_peak` warns when fewer
  than 5 lags are above half maximum. (The 5-point parabola it replaced
  misread the peak by up to 22 m/s at 0.75 km/s.)
- **Shape changes leak into the RVs.** A CCF peak moves when the line
  shapes change, not only when the star moves. In
  [tutorial 07](../examples/tutorials/07_usurper.ipynb) a star whose temperature
  swings with no motion at all shows an 11.4 m/s RV signal at the
  temperature's period; in [tutorial 08](../examples/tutorials/08_partial_pdc.ipynb)
  the same leak (11.8 m/s) sits next to a 15 m/s planet. Use USuRPER and the
  partial periodograms to tell shifts from shapes
  ([periodograms.md](periodograms.md)).
- **One template per spectrum.** A blended double-lined spectrum biases a
  single-template CCF: −266 m/s on the primary of the tutorial 04 binary when
  the two stars are 11 km/s apart. Use TODCOR.

## TODCOR

- **Bias on short, line-crowded windows.** The Fourier shift is circular: a
  line pushed past one end of a window re-enters at the other, and the taper
  only damps the ends. On noiseless simulated SB2 spectra
  ([tutorial 04](../examples/tutorials/04_todcor.ipynb)) the largest
  velocity bias was 15.6 m/s on a 50 Å window (`alpha=0.1`,
  `continuum='median'`) and 3.1 m/s on a 200 Å window (`alpha=0.3`,
  `continuum=1.0`); the test suite quotes ~10 m/s at |v| ≈ 45 km/s over 50 Å
  and ~2 m/s over 200 Å. Use long windows and a firm taper; the
  template is not extended or zero-padded beyond the window yet.
- **One amplitude pair for all orders.** Pooled orders share (a1, a2). The
  light ratio of two stars of different temperature changes with wavelength,
  so over a wide range one pair is an approximation
  ([todcor_method.md §7](../packages/todcor/docs/todcor_method.md)).
- **`continuum='median'` biases the amplitudes.** The median of an
  absorption spectrum lies below its continuum. The test suite quotes the
  amplitudes ~0.5% low; in tutorial 04 (50 Å window) they came out 0.696 and
  0.297 for 0.7 and 0.3. For continuum-normalised spectra pass
  `continuum=1.0`; for filtered spectra the median (≈ 0) is right.
- **Twin templates have no component labels.** With the same template for
  both stars the fit is symmetric under exchanging (v1, a1) and (v2, a2), so
  which star is "1" is arbitrary and can change from epoch to epoch. Give the
  components disjoint trial grids (`v1_grid`, `v2_grid`) or sort the epochs'
  solutions afterwards.
- **A template whose continuum is not 1.** `CCF2d` fills observed pixels
  outside a template's coverage with 1.0, the normalised continuum. A
  TIRAVEL template is a normalised co-add whose continuum is near 0 (~0.01
  in a simulated test), so the fill puts a large step at the uncovered ends of
  each order. Cut the observed orders to the template's coverage, or use the
  TIRAVEL template with UNICOR, which drops uncovered pixels instead.
- **Local Cramér–Rao errors.** The errors come from a quadratic fit to the
  log-likelihood at an isolated, resolved peak. They degrade for blended or
  nearly degenerate components and are NaN when the peak is on the edge of the
  velocity grid ([correlation_engines.md](correlation_engines.md#peak-refinement-and-the-errors)).

## TIRAVEL

- **Fixed velocities only.** `TIRAVEL.build_template` evaluates the alignment
  measure ρ and builds the co-added template at the velocities you supply.
  The optimisation of ρ over the velocities, which is the rest of the method
  (Zucker & Mazeh 2006), is not implemented: bootstrap from RVs measured with
  any reasonable template ([tutorial 05](../examples/tutorials/05_tiravel.ipynb)).
- **Relative velocities.** Only velocity differences enter ρ; the template's
  rest frame inherits the zero point of the supplied velocities.

Status, diagnostics and the development plan: [tiravel.md](tiravel.md).

## Error bars

The calibration of UNICOR's and TODCOR's errors is summarised in
[conventions.md](conventions.md#when-are-the-error-bars-calibrated). In short:

- calibrated for white noise on the native pixels of an unfiltered spectrum;
- conservative after `preprocess`: UNICOR eRV/scatter 1.17–1.38 depending on
  the noise cut (`noise_scale_kms`), TODCOR ~1.4×;
- conservative under photon noise: UNICOR scatter/eRV ≈ 0.89 (~10%);
- conservative with template mismatch, which is counted as noise;
- TODCOR: conservative on analysis grids coarser than native whose nodes fall
  between native pixels (scatter/eRV 0.82–0.91 at 1.5× native); an explicit
  `sigma` is taken at face value and the errors are not rescaled by
  `chi2_red`.

## Periodograms

- **USuRPER and the partial periodograms take one order** of spectra that
  overlap in wavelength; multi-order or non-overlapping input raises.
- **p-values are local.** `calc_pdc_pval` gives a single-frequency p-value
  (Shen et al. 2019), not a false-alarm probability over the grid. For the
  partial periodograms the formula is borrowed from the distance correlation
  and not established; a permutation check in tutorial 08 held at one
  frequency ([periodograms.md](periodograms.md#p-values)).
- **A partial periodogram is not a conditional-independence test.** The
  control is removed only as far as it is linear in the U-centred distances,
  so part of an RV leak can survive (shift power 0.22 at the activity period
  of tutorial 08's planet-free active star, against 0.61 for the plain PDC); a
  pdCor of 0 does not mean conditional independence, nor a non-zero one a
  dependence beyond the control. Signals at the same period cannot be
  separated.
- **GLS is unweighted.** `calc_gls` does not pass the errors to astropy's
  `LombScargle`.
- **Small samples.** The unbiased statistics raise for n ≤ 3; the biased ones
  are not guarded.

## Instruments and data

- **Two instruments are verified.** Only the HARPS ESO ADP spectra (`HARPS`)
  and the NRES e91 spectra (`NRES`) have been checked against real archive
  files (bundled and pinned in the golden master). Every other loader is
  unverified and warns on each load ([instruments.md](instruments.md)).
- **Archive rebinning is modelled, not known.** For an archive-resampled
  product with a documented detector step (HARPS ADP: 0.609 km/s), the errors
  count detector pixels and TODCOR models the noise as resampled once, by
  SPARTA's own spline, from uniformly spread sub-pixel phases. The pipeline's
  real rebinning kernel is unknown, so the noise budget is approximate.
- **Resampled products without a documented native step** (`CORALIE`,
  `ELODIE`, `FEROS`, `LAMOST`, `PEPSI`, `SRG`, `UVES`) count their archive
  pixels as independent; if the archive grid oversamples the detector, the
  errors are too small.
- **Undocumented wavelength frames.** For `NRES`, `CORALIE`, `ELODIE`,
  `FEROS-CERES`, `SRG` and `TRES` the archive does not state air or vacuum;
  choose the template's frame yourself (a mismatch is a ~80 km/s offset,
  [conventions.md](conventions.md#air-and-vacuum-wavelengths)).
- **No time conversion.** Times are kept in each product's own system (BJD,
  MJD-UTC, JD-UTC, ...), which is recorded but not converted to BJD-TDB;
  `Spectrum.bjd` holds whatever the loader returned. `Observations` refuses
  to mix systems in one directory.
- **`Observations.file_list` lists every file read**, including those
  dropped by the `min_snr` / `min_avg_flux_val` cuts, so it is not aligned
  with the epochs after a cut.

## Spectra and templates

- **`Spectrum` is not yet a specutils object.** It is a multi-order container
  of per-order arrays plus an `Info` dict; specutils is the interchange format
  (`Spectrum.orders`, `Spectrum.from_spectrum1d`), not the working store.
- **The band-pass cuts are converted at each order's median wavelength**, so
  on a very wide single order (a whole 3800–6900 Å 1-D spectrum) they are off
  by up to ~30% at its ends; use a window (`window_A`) or echelle orders.
- **Trimming and the taper are set in pixels.** The band-pass cuts are
  velocity scales, but `n_trim` and the taper length (`alpha`, a fraction of
  the order) are not, so on a template with finer pixels or wider orders they
  cut a different stretch of spectrum.
- **`Template.integrate_spec`** bins only the first order, drops the trailing
  pixels and the last complete group, and sums (rather than averages) each
  group.
