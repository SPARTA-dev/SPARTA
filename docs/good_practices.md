# Good practices

A checklist for an analysis with SPARTA, in the order the work is done. Each
item is something to do; ⚠ says what goes wrong if it is skipped, with its size
where it has been measured; → points to where it is explained. The
explanations live in the reference pages, the tutorials demonstrate them, and
[KNOWN_TRAPS.md](../KNOWN_TRAPS.md) records how each trap was found and what
was fixed (entries are cited by their ID or title).

## Before you start

- [ ] **Install from a clone, never with `pip install sparta`.** ⚠ That name on
  PyPI is an unrelated package, and the engines are not on PyPI yet.
  → [README](../README.md#installation)
- [ ] **Pick one velocity convention and keep it** in the RV drivers,
  `Template.doppler` and `Simulation` ('cz' by default). ⚠ 'cz' and
  'relativistic' RVs of the same spectrum differ by ≈ v²/2c: 37.5 m/s at
  150 km/s. → [conventions](conventions.md#velocity-convention); KNOWN_TRAPS D5

## Reading data

- [ ] **Read the product's entry in [instruments.md](instruments.md)**: wavelength
  frame, time system, SPECSYS, native step, recommended settings. Products
  marked *(unverified)* have not been checked against real files. ⚠ Several
  products keep times in their own system (MJD-UTC, JD-UTC, ...); nothing
  converts them to BJD-TDB yet. → [limitations](limitations.md#instruments-and-data); KNOWN_TRAPS
  "Instrument loaders"
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
  [tutorial 02](../examples/tutorials/02_log_lambda.ipynb); KNOWN_TRAPS D1
- [ ] **Drop the orders the product's recommendation drops.** For NRES that is
  orders 0–9 and 50–66: NRES numbers its orders from red to blue, so these are
  the 10 reddest and the 17 bluest. → [instruments § NRES](instruments.md#nres)

## Templates

- [ ] **Put the template in the data's frame.** PHOENIX is in vacuum; for an
  air product (HARPS) use `Template.from_phoenix(..., air=True)` (Morton 2000).
  ⚠ A mismatch shifts every RV by ~83 km/s at 5000 Å, and by a slightly
  different amount in each order, so it does not cancel as one constant.
  → [conventions](conventions.md#air-and-vacuum-wavelengths),
  [tutorial 05](../examples/tutorials/05_tiravel.ipynb); KNOWN_TRAPS "TIRAVEL
  quickstart/golden used a vacuum template"
- [ ] **Broaden on a velocity-even grid.** Rotation and the line-spread
  function (FWHM c/R) have one width in velocity; use `Template.broaden_*`, or
  `unicor.broaden_*` on a ln λ grid. ⚠ PHOENIX's step jumps from 0.006 to
  0.01 Å at 5000 Å (0.36 to 0.60 km/s): a kernel sized in pixels on the linear
  grid is ~1.7× too wide or too narrow on one side, and the median-step kernel
  SPARTA used before 0.3.0 was off by 0.8–1.35×.
  → [tutorial 03](../examples/tutorials/03_unicor.ipynb); KNOWN_TRAPS
  "template broadening on a non-ln-λ grid"
- [ ] **Cut the template wider than the scan.** Every lag must find template:
  tutorial 03 cuts each order with a 150 km/s margin for a ±100 km/s scan.
  → [correlation engines § the CCF](correlation_engines.md#the-ccf)
- [ ] **Preprocess the template with the data's settings, and mind the pixel
  scale.** ⚠ The band-pass cuts are in pixel units, so on a template with finer
  pixels the same settings are a different filter; changing the template's
  pixels moved the NRES RVs by −12 and −18 m/s.
  → [limitations](limitations.md#spectra-and-templates); KNOWN_TRAPS
  "`bandpass` cut in pixel units"
- [ ] **Download templates only on purpose.** Models come from the local
  cache; fetching another needs `allow_download=True`, and the file lands in the
  package directory. → [limitations](limitations.md#spectra-and-templates)

## Preprocessing

- [ ] **Start from `instruments.get(key).recommended`.** These are settings that
  worked on the bundled data, not instrument documentation.
  → [instruments](instruments.md)
- [ ] **Trim the margins before filtering** (`Spectrum.preprocess` does). ⚠ A
  band-pass over the jump from padding to continuum makes a spurious edge step;
  removing the NRES margins first lowered the TOI-677 RV scatter from 82 to
  75 m/s. → [conventions](conventions.md#preprocessing-order)
- [ ] **Give `Spectrum.interpolate(method='linear')` a number, not a length.**
  ⚠ A Quantity `delta` is read as a multiple of the median step: `0.25*u.AA`
  is not 0.25 Å. → [limitations](limitations.md#spectra-and-templates)

## Radial velocities

- [ ] **Correlate on a uniform ln λ grid.** The engines resample when needed and
  raise on a grid that only claims to be one. → [conventions](conventions.md#the-uniform-ln-λ-grid)
- [ ] **Use the recommended CCF step** (0.05–0.1 km/s). ⚠ The 5-point parabola
  misreads the peak by up to 6.5–14.6 m/s at dv = 0.75 km/s on sharp-lined
  R = 100,000 spectra; it falls as dv³. → [limitations](limitations.md#unicor);
  KNOWN_TRAPS "5-point parabola pixel-phase error"
- [ ] **Scan wide enough, and look for NaN.** A peak at the edge of the scan
  gives NaN for that order (UNICOR) or `on_edge` and a NaN error (TODCOR),
  rather than a wrong number. → [correlation engines](correlation_engines.md#rv-and-its-uncertainty)
- [ ] **Use the combined RV and its error.** The scatter of the per-order RVs
  is not an uncertainty. → [correlation engines § combining orders](correlation_engines.md#combining-orders-zucker-2003)
- [ ] **Do not use the bisector span.** ⚠ It is known to be incorrect and is
  pinned for a later fix. → [correlation engines](correlation_engines.md#bisector-span-known-to-be-incorrect);
  KNOWN_TRAPS D2
- [ ] **Use TODCOR for a blended double-lined spectrum.** ⚠ A single-template
  CCF was −270 m/s off on the primary of tutorial 04's binary (stars 11 km/s
  apart). → [limitations](limitations.md#unicor)

## Error bars

- [ ] **Calibrate the errors after preprocessing.** The ML (UNICOR) and
  Cramér–Rao (TODCOR) errors assume white noise on the native pixels;
  band-pass filtering correlates it. ⚠ In Monte Carlo the errors came out
  17–38 % too large (UNICOR) and ~1.4× (TODCOR). Calibrate by simulating your
  setup, or by the scatter of a constant star.
  → [conventions](conventions.md#when-are-the-error-bars-calibrated)
- [ ] **Expect a mismatched template to inflate the errors.** ⚠ In tutorial 07
  the reported errors were about three times the scatter when the template did
  not match the star's changing spectrum.
  → [tutorial 07](../examples/tutorials/07_usurper.ipynb)

## Double-lined binaries (TODCOR)

- [ ] **Use long windows, a firm taper and the right continuum.** ⚠ On
  noiseless simulated SB2 spectra the velocity bias was 15.6 m/s on a 50 Å
  window (`alpha=0.1`, `continuum='median'`) and 3.1 m/s on 200 Å (`alpha=0.3`,
  `continuum=1.0`). → [limitations](limitations.md#todcor),
  [tutorial 04](../examples/tutorials/04_todcor.ipynb); KNOWN_TRAPS
  "TODCOR taper-before-shift"
- [ ] **Pass `continuum=1.0` for continuum-normalised spectra.** ⚠ The median of
  an absorption spectrum lies below its continuum and lowers the amplitudes
  (0.696 and 0.297 for 0.7 and 0.3 in tutorial 04). → [limitations](limitations.md#todcor)
- [ ] **Label the components.** With twin templates, give the stars disjoint
  trial grids (`v1_grid`, `v2_grid`) or sort the solutions afterwards. ⚠ Which
  star is "1" can change from epoch to epoch. → [limitations](limitations.md#todcor)
- [ ] **Remember that (a1, a2) are shared by all orders.** ⚠ The light ratio of
  stars of different temperature changes with wavelength.
  → [limitations](limitations.md#todcor)
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
- [ ] **With TODCOR, cut the orders to the template's coverage.** ⚠ Its
  continuum is near 0 and `CCF2d` fills uncovered pixels with 1.
  → [limitations](limitations.md#todcor)

## Periodograms

- [ ] **Use a fine enough frequency grid** (in 1/day), a step several times
  finer than 1/T, starting above 0. → [tutorial 06](../examples/tutorials/06_pdc.ipynb)
- [ ] **Correct the p-values for the search.** ⚠ They are local, for one
  frequency; a rough look-elsewhere factor is T·(f_max − f_min). For the
  partial periodograms the formula is borrowed and not established: confirm by
  permutation or by injection and recovery.
  → [periodograms § p-values](periodograms.md#p-values); KNOWN_TRAPS "PDC
  p-value levels are single-frequency"
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
