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

- **Templates must reach past every order.** `CCF2d` reads each template up
  to v_max = the largest |trial velocity| plus the fine map's reach
  (`refine_span` = 2 coarse steps) beyond each end of each order, and drops
  the observed pixels it cannot (an order with fewer than 2 left raises). No
  line is lost to a wrap-around any more, but a short template costs pixels:
  a template cut by `Template.cut_multiorder_like(spectrum, margins=150)`
  leaves ~2 km/s plus a pixel at each end of each order uncovered at the
  default `vel_bound` and `dvel`.
- **A small velocity bias remains.** On noise-free simulated SB2 spectra
  (R = 60,000, templates sampled 10× finer than a 0.5 km/s grid) the
  pipeline's largest velocity error is ≤ 0.023 m/s on a 50 Å window and
  ≤ 0.016 m/s on 200 Å (the earlier circular, tapered shift: 11.8 and
  1.9 m/s). The exact χ² minimum is within 0.012 m/s of the truth; that
  remainder is the linear resampling of the templates onto the analysis grid
  ([todcor_method.md §2](../packages/todcor/docs/todcor_method.md#2-the-window-observed-pixels-extended-templates)).
- **Amplitudes are shared across orders by default.** The light ratio of two
  stars of different temperature changes with wavelength, so over a wide
  range one (a1, a2) is an approximation. `amplitudes='per_order'` fits one
  pair per order, at two free parameters per order; within an order the
  ratio is still one number
  ([correlation_engines.md](correlation_engines.md#pooling-orders)).
- **Twin templates have no component labels.** With the same template for
  both stars the fit is symmetric under exchanging (v1, a1) and (v2, a2), so
  which star is "1" is arbitrary and can change from epoch to epoch. Give the
  components disjoint trial grids (`v1_grid`, `v2_grid`) or sort the epochs'
  solutions afterwards.
- **A TIRAVEL template spans only the observed range.** Its continuum level no
  longer matters (each order's offset is free, and uncovered pixels are
  dropped, not filled), so it can feed `CCF2d`. But it covers only the
  epochs' common rest-frame range, so each order loses about v_max at each
  end (~2.5 Å at 5000 Å for the default 150 km/s scan); narrow the scan
  (`vel_bound` or the trial grids) to keep more. Its scale is arbitrary, so
  the amplitudes against it are not light fractions.
- **Local Cramér–Rao errors.** The errors come from a degree-6 polynomial
  fitted to the log-likelihood around an isolated, resolved peak. They degrade
  for blended or nearly degenerate components and are NaN when the peak is on
  the edge of the velocity grid or the polynomial's maximum lies more than one
  fine step from the grid maximum
  ([correlation_engines.md](correlation_engines.md#peak-refinement-and-the-errors)).

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
- after `preprocess`, UNICOR's errors count only the noise the band-pass
  kept, an approximate correction that errs high: eRV/scatter 1.25–1.4 on
  simulated spectra, and ~3 on the real HARPS TOI-193 RVs, where template
  mismatch adds to it; an exact error model for filtered spectra is open
  ([roadmap](roadmap.md)); TODCOR, with the same correction, 1.1–1.2× on
  simulated HARPS-like SB2s;
- conservative under photon noise: UNICOR scatter/eRV ≈ 0.89 (~10%);
- conservative with template mismatch, which is counted as noise;
- TODCOR: conservative on analysis grids coarser than native whose nodes fall
  between native pixels (scatter/eRV 0.8–0.9 at 1.5× native); an explicit
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
- **Trimming is set in pixels.** The band-pass cuts are velocity scales, but
  `n_trim` is not, so on a template with finer pixels it cuts a different
  stretch of spectrum.
- **`Template.integrate_spec`** bins only the first order, drops the trailing
  pixels and the last complete group, and sums (rather than averages) each
  group.
