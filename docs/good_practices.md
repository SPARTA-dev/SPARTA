# Good practices

A checklist for an analysis with SPARTA, in the order the work is done. Each
item is something to do; ⚠ says what goes wrong if it is skipped, with its size
where it has been measured; → points to where it is explained. The
explanations live in the reference pages and the tutorials demonstrate them;
every known limitation, with its measured size, is in
[limitations.md](limitations.md).

## Before you start

- [ ] **Install from a clone, never with `pip install sparta`.** ⚠ That name on
  PyPI is an unrelated package, and the engines are not on PyPI yet.
  → [README](../README.md#installation)
- [ ] **Pick one velocity convention and keep it** in the RV drivers,
  `Template.doppler` and `Simulation` ('cz' by default). ⚠ 'cz' and
  'relativistic' RVs of the same spectrum differ by ≈ v²/2c: 37.5 m/s at
  150 km/s. → [conventions](conventions.md#velocity-convention)

## Reading data

- [ ] **Read the product's entry in [instruments.md](instruments.md)**: wavelength
  frame, time system, SPECSYS, native step, recommended settings. Products
  marked *(unverified)* have not been checked against real files. ⚠ Several
  products keep times in their own system (MJD-UTC, JD-UTC, ...); nothing
  converts them to BJD-TDB yet. → [limitations](limitations.md#instruments-and-data)
- [ ] **Keep one product per directory.** `Observations` refuses a directory
  that mixes wavelength frames or time systems; do not work around it.
  → [conventions](conventions.md#air-and-vacuum-wavelengths)
- [ ] **Apply the barycentric correction once.** `Observations` applies the
  record's `bcv` on ingest; with the loaders alone, apply
  `unicor.doppler_shift_wave(wave, bcv)` yourself. It is composed as cz
  whatever convention the RVs use. → [tutorial 01](../examples/tutorials/01_reading_spectra.ipynb)
- [ ] **Know the native step.** Archive-resampled products (HARPS 1-D, 0.01 Å
  bins) carry the detector step (0.609 km/s for HARPS); blueward of ~4920 Å the
  0.01 Å bins are coarser than the detector, and then they set it. ⚠ The error
  bars count native pixels: a wrong step scales every error. →
  [conventions](conventions.md#native-sampling-and-the-number-of-independent-pixels),
  [tutorial 02](../examples/tutorials/02_log_lambda.ipynb)
- [ ] **Drop the orders the product's recommendation drops.** For NRES that is
  orders 0–9 and 50–66: NRES numbers its orders from red to blue, so these are
  the 10 reddest and the 17 bluest. → [instruments § NRES](instruments.md#nres)

## Templates

- [ ] **Put the template in the data's frame.** PHOENIX is in vacuum; for an
  air product (HARPS) use `Template.from_phoenix(..., air=True)` (Morton 2000).
  ⚠ A mismatch shifts every RV by ~83 km/s at 5000 Å, and by a slightly
  different amount in each order, so it does not cancel as one constant.
  → [conventions](conventions.md#air-and-vacuum-wavelengths),
  [tutorial 05](../examples/tutorials/05_tiravel.ipynb)
- [ ] **Broaden on a velocity-even grid.** Rotation and the line-spread
  function (FWHM c/R) have one width in velocity; use `Template.broaden_*`, or
  `unicor.broaden_*` on a ln λ grid. ⚠ PHOENIX's step jumps from 0.006 to
  0.01 Å at 5000 Å (0.36 to 0.60 km/s): a kernel sized in pixels on the linear
  grid is ~1.7× too wide or too narrow on one side, and the median-step kernel
  SPARTA used before 0.3.0 was off by 0.8–1.35×.
  → [tutorial 03](../examples/tutorials/03_unicor.ipynb)
- [ ] **Cut the template wider than the scan.** Every lag must find template:
  tutorial 03 cuts each order with a 150 km/s margin for a ±100 km/s scan.
  → [correlation engines § the CCF](correlation_engines.md#the-ccf)
- [ ] **Preprocess the template with the data's settings.** The band-pass
  cuts are velocity scales (km/s), so data and template get the same filter
  whatever their pixels; ⚠ `n_trim` is still in pixels.
  → [conventions](conventions.md#preprocessing-order),
  [limitations](limitations.md#spectra-and-templates)
- [ ] **Download templates only on purpose.** Three models ship with SPARTA;
  fetching another needs `allow_download=True` and puts it in the user cache
  (`$SPARTA_CACHE_DIR/phoenix`, else `$XDG_CACHE_HOME/sparta/phoenix` or
  `~/.cache/sparta/phoenix`). `SPARTA_CACHE_DIR` (an absolute path) can point
  several users at one cache; whoever downloads needs write access to it.

## Preprocessing

- [ ] **Start from `instruments.get(key).recommended`.** These are settings that
  worked on the bundled data, not instrument documentation.
  → [instruments](instruments.md)
- [ ] **Trim the margins before filtering** (`Spectrum.preprocess` does). ⚠ A
  band-pass over the jump from padding to continuum makes a spurious edge step;
  removing the NRES margins first lowered the TOI-677 RV scatter from 82 to
  75 m/s. → [conventions](conventions.md#preprocessing-order)
- [ ] **Know what `Spectrum.interpolate(method='linear')` reads.** A number is
  a multiple of each order's median step; a length (`0.25*u.AA`) is the step
  itself; a velocity raises (it belongs to `method='log'`).

## Radial velocities

- [ ] **Correlate on a uniform ln λ grid.** The engines resample when needed and
  raise on a grid that only claims to be one. → [conventions](conventions.md#the-uniform-ln-λ-grid)
- [ ] **Sample the CCF peak with at least ~5 lags across its half width.** The
  RV is read between lags from a spline of the CCF, so it does not depend on
  the step while the peak is well sampled (0.12 m/s at 0.75 km/s on R = 100,000
  lines); ⚠ fewer lags and it degrades fast (17 m/s at 1.5 km/s), and
  `subpixel_peak` warns. The recommended steps (0.05–0.1 km/s) are far inside.
  → [limitations](limitations.md#unicor)
- [ ] **Scan wide enough, and look for NaN.** A peak at the edge of the scan
  gives NaN for that order (UNICOR) or `on_edge` and a NaN error (TODCOR),
  rather than a wrong number. → [correlation engines](correlation_engines.md#rv-and-its-uncertainty)
- [ ] **Use the combined RV and its error.** The scatter of the per-order RVs
  is not an uncertainty. → [correlation engines § combining orders](correlation_engines.md#combining-orders-zucker-2003)
- [ ] **Do not use the bisector span.** ⚠ It is known to be incorrect and is
  pinned for a later fix. → [correlation engines](correlation_engines.md#bisector-span-known-to-be-incorrect)
- [ ] **Use TODCOR for a blended double-lined spectrum.** ⚠ A single-template
  CCF was −266 m/s off on the primary of tutorial 04's binary (stars 11 km/s
  apart). → [limitations](limitations.md#unicor)

## Error bars

- [ ] **Calibrate the errors after preprocessing.** The band-pass removes
  most of the noise and correlates the rest; UNICOR and TODCOR correct for the
  noise it removes, approximately. ⚠ The quoted errors err high: UNICOR
  1.25–1.4× the scatter on simulated spectra, ~3× on the real HARPS TOI-193
  RVs (template mismatch adds to it); TODCOR 1.1–1.2× on simulated SB2s. Calibrate by simulating your setup, or by the
  scatter of a constant star.
  → [conventions](conventions.md#when-are-the-error-bars-calibrated)
- [ ] **Expect a mismatched template to inflate the errors.** ⚠ In tutorial 07
  the reported errors were about three times the scatter when the template did
  not match the star's changing spectrum.
  → [tutorial 07](../examples/tutorials/07_usurper.ipynb)

## Double-lined binaries (TODCOR)

- [ ] **Cut the templates wider than the scan plus the fine map.** `CCF2d`
  needs both templates over each order ± (`vel_bound` + 2 `dvel`) plus a
  pixel. ⚠ Observed pixels they do not reach are dropped, which costs
  information. → [limitations](limitations.md#todcor)
- [ ] **Choose shared or per-order amplitudes.** Keep `amplitudes='shared'`
  (the default) when the light ratio is constant over the fitted orders
  (similar stars, a narrow range); use `amplitudes='per_order'` for stars of
  different temperature over a wide range. ⚠ Shared amplitudes are an
  approximation when the ratio changes with wavelength; per-order ones cost
  two free parameters per order and give one (a1, a2) per order.
  → [correlation engines § pooling orders](correlation_engines.md#pooling-orders)
- [ ] **Label the components.** With twin templates, give the stars disjoint
  trial grids (`v1_grid`, `v2_grid`) or sort the solutions afterwards. ⚠ Which
  star is "1" can change from epoch to epoch. → [limitations](limitations.md#todcor)
- [ ] **Look at the R map; take the errors from ln L.** ⚠ ln L spans ~10⁶ over
  the map and shows only the peak node. → [correlation engines § two maps](correlation_engines.md#two-maps-one-peak)

## TIRAVEL

- [ ] **Bootstrap the velocities.** ⚠ SPARTA does not optimise ρ over them: the
  template is only as aligned as the RVs you supply. → [tiravel](tiravel.md#using-it)
- [ ] **Anchor the zero point.** ⚠ Only velocity differences enter; the template
  and the RVs against it inherit the zero point of the input velocities.
  → [tiravel](tiravel.md#what-ρ-says-and-what-it-does-not)
- [ ] **Check the weights and `n_eff` against K.** ⚠ A negative or near-zero
  weight flags a bad spectrum or a wrong velocity. → [tiravel](tiravel.md#the-weights)
- [ ] **Do not read ρ on its own.** ⚠ 1 − ρ ∝ (S/N)⁻²: compare ρ between trial
  velocities of the same data, not between data sets.
  → [tiravel](tiravel.md#what-ρ-says-and-what-it-does-not)
- [ ] **With TODCOR, narrow the scan.** The template spans only the epochs'
  common range; ⚠ `CCF2d` drops about `vel_bound` from each end of each
  order, where the template does not reach. Its scale is arbitrary, so the
  amplitudes are not light fractions. → [limitations](limitations.md#todcor)

## Periodograms

- [ ] **Use a fine enough frequency grid** (in 1/day), a step several times
  finer than 1/T, starting above 0. → [tutorial 06](../examples/tutorials/06_pdc.ipynb)
- [ ] **Correct the p-values for the search.** ⚠ They are local, for one
  frequency; a rough look-elsewhere factor is T·(f_max − f_min). For the
  partial periodograms the formula is borrowed and not established: confirm by
  permutation or by injection and recovery.
  → [periodograms § p-values](periodograms.md#p-values)
- [ ] **Check P against 2P, and against the 1-day aliases.** ⚠ PDC-type
  periodograms respond weakly at twice the true period (0.24 against 0.66 in
  tutorial 08), and nightly sampling makes 1-day aliases (tutorial 09).
  → [periodograms § PDC](periodograms.md#pdc-phase-distance-correlation-zucker-2018)
- [ ] **Use the error-aware PDC when the error bars differ a lot;** GLS here is
  unweighted. → [periodograms](periodograms.md#error-aware-pdc-binnenfeld-shahaf--zucker-2024),
  [limitations](limitations.md#periodograms)
- [ ] **Have at least 4 epochs.** ⚠ The unbiased statistics raise for n ≤ 3;
  the biased ones are not guarded. → [limitations](limitations.md#periodograms)
- [ ] **Tell shifts from shapes before claiming a planet.** ⚠ Line-shape
  changes leak into the RVs at their own period (11 m/s in tutorials 07 and
  08). Use USuRPER and the shift/shape periodograms, which need one order of
  overlapping spectra and cannot split signals at the same period.
  → [periodograms § partial](periodograms.md#partial-periodograms-shift-and-shape-binnenfeld-et-al-2022),
  [limitations](limitations.md#periodograms)

## Reporting

- [ ] **Record what the numbers depend on:** the SPARTA version
  (`sparta.__version__`), the product key, the velocity convention, the
  template and its frame, the preprocessing settings and the CCF step.
- [ ] **Cite the papers behind the methods you used.** → [README](../README.md#citing)
