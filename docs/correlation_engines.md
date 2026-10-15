# Correlation engines: UNICOR and TODCOR

SPARTA measures radial velocities with two engines. **UNICOR**
(`unicor`, driven by `sparta.CCF1d`) correlates a spectrum with one template;
**TODCOR** (`todcor`, driven by `sparta.CCF2d`) fits two templates at two
velocities, for double-lined binaries. Both run on the uniform ln λ grid and
count native pixels, as described in [conventions.md](conventions.md); both
return either a correlation or a log-likelihood (`output='ccf'` or
`output='loglike'`). The RVs and errors never depend on that choice.

Notation: C is a Pearson correlation, N an effective number of independent
spectral bins (`n_eff`), Δ the grid's measured ln λ step, k a lag in pixels.

## UNICOR (`CCF1d`)

### The CCF

For each order, the observation (N pixels on its ln λ grid) is correlated
with the template sampled on the **same grid extended by K pixels on each
side** (`unicor.extend_log_grid`), K = round(`vel_bound`/dv). At lag k the
N observed pixels are correlated with the N-pixel template segment shifted by
k, each mean-subtracted and normalised by its own norm, so every lag is a
full Pearson correlation over the same pixels
(`unicor.ccf_vs_velocity`). Lag k is the velocity c·expm1(kΔ) (`'cz'`) or
c·tanh(kΔ) (`'relativistic'`).

Why the extended template: correlating against a template on the observed
span only overlaps N − |k| pixels at lag k, so the CCF is tapered by about
1 − |k|/N and its peak is pulled toward v = 0 — by the same sign in every
order, so it does not average out. That was SPARTA's CCF before 0.2.0
(18 m/s per 60 Å order at vsini = 10 km/s, up to ~1 km/s on short orders;
CHANGELOG 0.2.0).

`CCF1d.cross_correlate` correlates only the observed pixels for which the
template is known at every lag; it never extrapolates the template. A
template that covers no pixel across the whole scan makes that order NaN,
with a warning. `Template.cut_multiorder_like(spectrum, margins=150)` cuts a
template with exactly the margin the default `vel_bound` (150 km/s) needs.

### Combining orders (Zucker 2003)

Order i with N_i independent bins and correlation C_i(v) has the
log-likelihood −(N_i/2) ln(1 − C_i²). The combined CCF is defined so that
−(N_tot/2) ln(1 − ML²) is their sum:

$$\mathrm{ML}(v) = \sqrt{1 - \prod_i \left(1 - C_i^2(v)\right)^{N_i/N_\mathrm{tot}}},\qquad N_\mathrm{tot} = \sum_i N_i$$

(`unicor.combine_ccfs`; C_i is clipped to [−1, 1] first; with equal N_i this
is Zucker's equal-weight formula). Each order is weighted by its native-pixel
count, N_i = n_pix,i · min(1, dv/native_i) ([conventions.md](conventions.md#native-sampling-and-the-number-of-independent-pixels)).
Each order keeps its own normalisation, so the orders need not share a line
depth or flux scale.

### Two output modes

- `output='ccf'` (default): the primary result is the CCF, per order
  (`Corr['corr']`) and combined (`CorrCombined['corr']`, = ML).
- `output='loglike'`: the primary result is Zucker's log-likelihood
  relative to no correlation,

  $$\ln L(v) = -\sum_i \frac{N_i}{2}\ln\left(1 - C_i^2(v)\right) = -\frac{N_\mathrm{tot}}{2}\ln\left(1 - \mathrm{ML}^2(v)\right),$$

  0 where C = 0 and +∞ at |C| = 1 (`unicor.loglike_from_ccf`).

Both are always stored (`'corr'` and `'loglike'` in `Corr` and
`CorrCombined`); the mode is recorded in `'output'` and selects what
`CCF1d.plot` draws. ln L is monotonic in C², so both curves peak at the same
lag.

### RV and its uncertainty

- **RV:** the vertex of a least-squares parabola through the 5 lags centred
  on the maximum of the combined CCF (`unicor.subpixel_peak`, `npts=5`; the
  fit is done in velocities relative to the central lag). It raises — and
  `CCF1d` reports NaN — when the maximum is within 2 lags of the scan edge
  or the parabola is not concave. The RV comes from the CCF in both output
  modes.
- **eRV:** the maximum-likelihood error of Zucker (2003, §2.3),

  $$\sigma_v^2 = -\left[N\,\frac{C''(v)\,C(v)}{1 - C^2(v)}\right]^{-1}$$

  at the peak, with C'' from an interpolating quartic spline and N the total
  native-pixel count (`unicor.rv_uncertainty`; `CorrCombined['n_eff']`). It
  is NaN where the curvature is not negative. Per-order errors are computed
  only with `errors_per_order=True`. Because N counts native pixels, the error
  does not depend on the scan width or on `dv`. It is calibrated for white
  noise on the native pixels and conservative after filtering
  ([conventions.md](conventions.md#when-are-the-error-bars-calibrated)).

### Bisector span: known to be incorrect

`unicor.bisector_span`, `CCF1d.calc_bis` and `Observations.calc_bis` return a
number that is **not a bisector span**: the formula builds the bisector
midpoints from the CCF *values* at the level crossings instead of the
crossing *velocities*, so the result is not a velocity. It is pinned,
unchanged, for a later fix, and is not covered by the golden master; do not
use it. `Observations.calc_bis(use_combined=False)` raises `AttributeError`.

## TODCOR (`CCF2d`)

The full derivation is in
[packages/todcor/docs/todcor_method.md](../packages/todcor/docs/todcor_method.md);
this section states what the code computes and what it returns.

### Model and sufficient statistics

TODCOR (Zucker & Mazeh 1994) is recast as a Gaussian profile likelihood in the
Fourier domain. With residuals g = flux − continuum on the ln λ grid, the
model of the data residual is

$$g(u) = a_1\,g_1(u - s_1) + a_2\,g_2(u - s_2),\qquad s_i = c\ln\frac{\lambda_\mathrm{obs}}{\lambda_\mathrm{rest}}(v_i),$$

and a shift is a phase factor exp(−2πiζs), so the templates are transformed
once and reused at every trial (v1, v2). For one window (one order), the
weighted χ² at any (v1, v2, a1, a2) is a quadratic form in six numbers, the
sufficient statistics of `todcor.window_stats`: S11, S22, S_D (scalars), b1(v1),
b2(v2) (data–template cross terms) and A12(v1, v2) (template–template):

$$\chi^2 = S_D - 2(a_1 b_1 + a_2 b_2) + a_1^2 S_{11} + 2a_1a_2A_{12} + a_2^2 S_{22}.$$

The weights are Parseval weights, so χ² equals the real-space χ² at noise σ
per pixel; the DC mode (and, for an even number of pixels, the Nyquist mode)
is dropped.

### Amplitudes in closed form

At every (v1, v2) node the amplitudes solve the 2×2 normal equations
N a = b, N = [[S11, A12], [A12, S22]], b = (b1, b2):
a1 = (S22 b1 − A12 b2)/det, a2 = (S11 b2 − A12 b1)/det,
det = S11 S22 − A12² (`todcor.profile_from_stats`). With `nonneg=True` (the
`CCF2d` default) both amplitudes are constrained to be ≥ 0 by a per-node
two-variable NNLS: a node whose unconstrained solution has a negative
amplitude takes the better of the two one-template edge solutions.

### Two maps, one peak

- `output='ccf'` (default): the primary map is Zucker & Mazeh's
  two-dimensional correlation with the light ratio fitted at every node,

  $$R^2 = 1 - \frac{\chi^2_\mathrm{min}}{S_D} = \frac{b^\top N^{-1} b}{S_D} = \frac{c_1^2 - 2c_1c_2c_{12} + c_2^2}{1 - c_{12}^2},$$

  with c1 = b1/√(S11 S_D), c2 = b2/√(S22 S_D), c12 = A12/√(S11 S22). R is
  in [0, 1] and free of the noise scale (with orders pooled, only their
  relative weights enter); with one template orthogonal to everything it
  reduces to the one-dimensional |C|.
- `output='loglike'`: the primary map is the Gaussian profile log-likelihood
  ln L = −χ²/2 at noise σ.

At fixed data **ln L = −(S_D/2)(1 − R²)**, so the two maps share their argmax:
the velocities, amplitudes and errors are identical in both modes (they always
come from ln L). Both maps are always stored (`Corr2d['R']`,
`Corr2d['logL_profile']`); `Corr2d['map']` is the one `output` selects and
`CCF2d.plot` draws it. R is the default because on a high-S/N spectrum ln L
spans ~10⁵ across the map and hides everything but the peak, while R keeps
the ridges where one template matches one star
([tutorial 04](../examples/tutorials/04_todcor.ipynb)). `logL_marginal`
(amplitudes marginalised under a flat prior; equal to `logL_profile` when
`nonneg=True`) is stored as well.

### Pooling orders

χ² is additive over windows with independent noise and linear in the six
statistics, so orders combine by **adding their statistics**
(`todcor.combine_stats`) into one joint fit with shared (v1, v2, a1, a2).
Two constraints: every window must be evaluated on the same velocity grid
(checked), and all pooled windows must belong to **one epoch** (not
checkable: orbital motion changes the velocities between epochs).
`Observations.calc_rv_against_two_templates` fits each epoch separately. One
amplitude pair is shared by all orders, which is an approximation when the
light ratio changes with wavelength ([limitations.md](limitations.md#todcor)).

### Noise, oversampling and n_eff

- **Oversampling.** Each order's Parseval weights are divided by
  M = max(1, native/dv), so a native pixel's information is counted once
  (`windows[i]['oversampling']`, `windows[i]['n_eff']` = n_pix/M).
- **Noise budget.** Each window also reports ν, the expected χ² of its pure
  noise. Resampled noise is smoother than native noise, so `CCF2d` builds
  the noise power spectrum from the resampler's impulse responses at the
  sub-pixel phases of the native pixels (`todcor.resampled_noise_psd`), times
  the taper's mean square. Then χ²_red = χ²_best/(Σν − 4) (four fitted
  parameters) is ≈ 1 for honest noise on any analysis step, finer or coarser
  than native.
- **`sigma=None` (default): per-order noise, two passes.** Pass 1 fits all
  orders at unit noise and locates the joint peak. Each order's share of the
  joint χ² there gives its own noise,
  σ_i² = χ²_i / (ν_i (1 − 4/Σν)). Pass 2 re-weights each order's statistics
  by 1/σ_i² (no new Fourier transforms) and re-fits; the RVs, amplitudes and
  errors come from pass 2. An order with fewer than `min_noise_dof` = 50
  noise degrees of freedom, or no residual, gets the common estimate instead
  (warning; `windows[i]['sigma_fallback']`). `Corr2d['sigma']` holds the
  estimates, and χ²_red ≈ 1 by construction.
- **Explicit `sigma`** (scalar or per order, per native pixel) is taken at
  face value, in one pass; its ratios weight the orders and its scale sets the
  errors. χ²_red is then a consistency check; the errors are not rescaled.

### Peak refinement and the errors

1. The coarse map (default `dvel` = 1 km/s over ±`vel_bound` = 150 km/s, i.e.
   301 × 301 nodes) only locates the peak.
2. The statistics are re-evaluated exactly on a fine local map: ±2 coarse
   steps around the coarse argmax at a tenth of the step (41 × 41 nodes;
   `refine_span`, `refine_factor`). A quadratic fit to the coarse map would
   be biased, because ln L follows the squared correlation and its peak is
   only ~σ_ccf/√2 wide.
3. `todcor.crlb_from_map` fits a 2-D quadratic to ln L over ±3 fine nodes
   (`crlb_half`). The Fisher matrix is −Hessian(ln L); its inverse is the
   (v1, v2) covariance (`Corr2d['cov']`, `'eRV1'`, `'eRV2'`, `'rho12'`).
   Because ln L is profiled over the linear amplitudes this is the full,
   local Cramér–Rao bound. The refined peak is the quadratic's stationary
   point, clipped to one fine step (`'RV1'`, `'RV2'`).
4. The amplitudes, χ² and R are evaluated exactly at the refined velocities
   (`'a1_best'`, `'a2_best'`, `'chi2_best'`, `'chi2_red'`, `'R_best'`).

If the coarse argmax is on the edge of the map, no refinement is done: a
warning is issued and the errors are NaN. If the fine argmax is on the edge of
the fine map (a ridge or a tie), a warning is issued and the coarse-map
estimate is reported.

### Preparing the windows

For each order `CCF2d` removes the continuum from data and templates alike
(`continuum='median'`, the default, or a fixed level such as 1.0), resamples
each template linearly onto the observed grid (filling uncovered pixels with
1.0, the normalised continuum), and applies a Tukey taper (`alpha`, default
0.1). The Fourier shift is circular, so lines shifted across a window edge
re-enter at the other end; keep windows long and the taper firm
([limitations.md](limitations.md#todcor)).

## Side by side

| | UNICOR — `CCF1d` | TODCOR — `CCF2d` |
| --- | --- | --- |
| Model | one template, scaled (Pearson C per order) | two templates at (v1, v2), amplitudes a1, a2 fitted |
| Orders | per-order CCFs combined by Zucker's ML rule; each order has its own scale | statistics summed; one (v1, v2, a1, a2) for all orders |
| Velocity grid | lags of the ln λ grid, ±`vel_bound` (default 150 km/s, dv 0.1 km/s) | coarse (v1, v2) map, `dvel` 1 km/s, then a fine 41 × 41 map |
| `output='ccf'` | `Corr['corr']` per order, `CorrCombined['corr']` = ML(v) | `Corr2d['map']` = `Corr2d['R']` (R ∈ [0, 1]) |
| `output='loglike'` | Zucker ln L, `Corr['loglike']`, `CorrCombined['loglike']` | Gaussian ln L = −χ²/2, `Corr2d['map']` = `Corr2d['logL_profile']` |
| Always stored | both `'corr'` and `'loglike'` | `'R'`, `'logL_profile'`, `'logL_marginal'`, `'a1'`, `'a2'`, `'chi2'` maps; the same under `'fine'` |
| RV | 5-point parabola on the combined CCF: `CorrCombined['RV']` | stationary point of a quadratic on the fine ln L map: `Corr2d['RV1']`, `['RV2']` |
| Error | Zucker ML error from the CCF curvature, N = native pixels: `CorrCombined['eRV']` | local Cramér–Rao bound from the ln L Hessian: `'eRV1'`, `'eRV2'`, `'rho12'`, `'cov'` |
| Noise level | not needed (implicit in C) | `sigma` per order, estimated by default; `'sigma'`, `'chi2_red'` |
| Peak value | `CorrCombined['peakCorr']` (CCF at the vertex) | `'R_best'` (R at the refined RVs) |
| Amplitudes | — | `'a1_best'`, `'a2_best'` |
| Bookkeeping | `'n_eff'`, `'vel'`, `'units'`, `'velocity_convention'`, `'output'` | `windows[i]` (`'n_eff'`, `'oversampling'`, `'nu'`, `'sigma'`, ...), `stats`, `crlb`, `'velocity_convention'`, `'output'` |
| Time-series driver | `Observations.calc_rv_against_template` → `vels`, `evels`, `ccf_peaks` (and `time_series.calculated_vrad_list`, `errors`) | `Observations.calc_rv_against_two_templates` → `vels1/2`, `evels1/2`, `amps1/2` |

Every key is documented in [api.md](api.md) under `CCF1d.cross_correlate`,
`CCF1d.combine` and `CCF2d.cross_correlate`.

## References

- Zucker, S. 2003, MNRAS, 342, 1291 — the ML combination of orders and the RV
  uncertainty.
- Engel, M., Shahaf, S. & Mazeh, T. 2017, PASP, 129, 065002 — UNICOR.
- Zucker, S. & Mazeh, T. 1994, ApJ, 420, 806 — TODCOR.
