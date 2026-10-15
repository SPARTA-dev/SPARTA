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

- **RV:** the maximum of the continuous combined CCF. The CCF of adequately
  sampled spectra is band-limited, so its lag values determine it between
  them: an interpolating quintic spline through ±4 lags around the sampled
  maximum stands for it, and the RV is the root of its derivative within one
  lag (`unicor.subpixel_peak`). It does not depend on where the peak falls
  between lags, so not on `dv`, while the peak spans ≳ 5 lags above half
  maximum (it warns otherwise). It raises — and `CCF1d` reports NaN — when
  the maximum is within 2 lags of the scan edge, the spline has no maximum
  within a lag, or the peak is not concave. Combining orders first is exact:
  Zucker's combination is monotone in the summed log-likelihood, so the
  combined spline peak is the maximum of the orders' summed continuous
  log-likelihoods (tested to 0.3 m/s). The RV comes from the CCF in both
  output modes.
- **eRV:** the maximum-likelihood error of Zucker (2003, §2.3),

  $$\sigma_v^2 = -\left[N\,\frac{C''(v)\,C(v)}{1 - C^2(v)}\right]^{-1}$$

  at the peak, with C and C'' from the same spline and N the total
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

TODCOR (Zucker & Mazeh 1994) is recast as a Gaussian profile likelihood over
the observed pixels of each window (one order). On the ln λ grid the data are
modelled as

$$d(u) = a_1\,g_1(u - s_1) + a_2\,g_2(u - s_2) + c_0,\qquad s_i = c\ln\frac{\lambda_\mathrm{obs}}{\lambda_\mathrm{rest}}(v_i),$$

with a free continuum offset c₀ per window. The templates g_i are given on the
window's grid extended past both ends by the largest trial shift (see
[Preparing the windows](#preparing-the-windows)) and shifted by a Fourier
phase factor exp(−2πi f s/dv) (f in cycles per pixel), then read on the
window's pixels: every shift reads the template's real neighbouring
spectrum. For one window, the χ² at any (v1, v2, a1, a2), with the offset
profiled out, is a quadratic form in six numbers, the sufficient statistics of
`todcor.window_stats`: S_D (scalar), S11(v1), S22(v2) (template powers over the
window), b1(v1), b2(v2) (data–template cross terms) and A12(v1, v2)
(template–template):

$$\chi^2 = S_D - 2(a_1 b_1 + a_2 b_2) + a_1^2 S_{11} + 2a_1a_2A_{12} + a_2^2 S_{22}.$$

Each is a weighted inner product over the window's n pixels of mean-subtracted
arrays (the offset projected out), with the pixel weight w = 1/(σ² M), σ the
native per-pixel noise and M the oversampling; e.g.
b1 = w Σ d G1 − (w Σ d)(w Σ G1)/(w n).

### Amplitudes in closed form

At every (v1, v2) node the amplitudes solve the 2×2 normal equations
N a = b, N = [[S11, A12], [A12, S22]], b = (b1, b2):
a1 = (S22 b1 − A12 b2)/det, a2 = (S11 b2 − A12 b1)/det,
det = S11 S22 − A12² (`todcor.profile_from_stats`). A template with no lines
in an order (flat there; `prepare_window` sets it to zero) gets amplitude 0, and
a node where the pair is degenerate (det < 10⁻¹⁰ S11 S22) takes the better
one-template solution. With `nonneg=True` (the `CCF2d`
default) both amplitudes are constrained to be ≥ 0 by a per-node two-variable
NNLS: a node whose unconstrained solution has a negative amplitude takes the
better of the two one-template edge solutions.

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
spans ~10⁶ across the map and hides everything but the peak, while R keeps
the ridges where one template matches one star
([tutorial 04](../examples/tutorials/04_todcor.ipynb)). `logL_marginal`
(amplitudes marginalised under a flat prior; equal to `logL_profile` under
`nonneg=True` or `amplitudes='per_order'`) is stored as well.

### Pooling orders

χ² is additive over windows with independent noise, so the orders of one
epoch combine into one joint fit with shared (v1, v2), in one of two ways
(`amplitudes=`):

- `'shared'` (default): one (a1, a2) for all orders. χ² is linear in the six
  statistics, so the orders combine by **adding their statistics**
  (`todcor.combine_stats`).
- `'per_order'`: each order keeps its own (a1, a2), i.e. its own light ratio,
  and the orders' profiled χ² maps are summed (`todcor.profile_per_window`);
  `'a1_best'`, `'a2_best'` then hold one value per order. Use it when the
  light ratio changes across the spectrum (stars of different temperature);
  it costs two parameters per order, so with a constant ratio `'shared'` is
  tighter.

Two constraints: every window must be evaluated on the same velocity grid
(checked), and all pooled windows must belong to **one epoch** (not
checkable: orbital motion changes the velocities between epochs).
`Observations.calc_rv_against_two_templates` fits each epoch separately
(`amplitudes` and the other `CCF2d.cross_correlate` keywords pass through).

### Noise, oversampling and n_eff

- **Oversampling.** Each order's pixel weight is divided by
  M = max(1, native/dv), so a native pixel's information is counted once
  (`windows[i]['oversampling']`, `windows[i]['n_eff']` = n_pix/M).
- **Noise budget.** Each window also reports ν = n_pix κ/M − 1, the expected
  χ² of its pure noise (one dof for its offset). Resampled noise is smoother
  than native noise (κ < 1), so `CCF2d` builds the noise power spectrum from
  the resampler's impulse responses at the sub-pixel phases of the native
  pixels (`todcor.resampled_noise_psd`; κ is its mean). Then
  χ²_red = χ²_best/(Σν − 4) (two velocities and two shared amplitudes; Σν − 2
  − 2 n_orders under `'per_order'`) is ≈ 1 for honest noise on any analysis
  step, finer or coarser than native.
- **`sigma=None` (default): per-order noise, two passes.** Pass 1 fits all
  orders at unit noise and locates the joint peak. Each order's share of the
  joint χ² there gives its own noise,
  σ_i² = χ²_i / dof_i, with dof_i = ν_i (1 − 4/Σν) (`'shared'`) or
  ν_i (1 − 2/Σν) − 2 (`'per_order'`) (`todcor.window_noise`). Pass 2
  re-weights each order's statistics by 1/σ_i² (no template is shifted
  again) and re-fits; the RVs, amplitudes and
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
   `refine_span`, `refine_factor`).
3. `todcor.crlb_from_map` fits a 2-D polynomial of total degree 6 to ln L
   over ±4 fine nodes (`crlb_half`) and takes its maximum (`'RV1'`, `'RV2'`;
   NaN errors if it lies more than one fine step from the argmax). ln L
   follows the squared correlation, so its peak is narrow (~σ_ccf/√2) and not
   quadratic over the patch: a quadratic misread it by up to 6.2 m/s; with
   degree 6 the pipeline is within 0.023 m/s of the truth (noise-free simulations;
   [method §2, §7](../packages/todcor/docs/todcor_method.md#7-peak-refinement-and-the-cramérrao-errors)).
   The Fisher matrix is −Hessian(ln L) at the maximum; its inverse is the
   (v1, v2) covariance (`Corr2d['cov']`, `'eRV1'`, `'eRV2'`, `'rho12'`).
   Because ln L is profiled over the linear amplitudes this is the full,
   local Cramér–Rao bound.
4. The amplitudes, χ² and R are evaluated exactly at the refined velocities
   (`'a1_best'`, `'a2_best'`, `'chi2_best'`, `'chi2_red'`, `'R_best'`).

If the coarse argmax is on the edge of the map, no refinement is done: a
warning is issued and the errors are NaN. If the fine argmax is on the edge of
the fine map (a ridge or a tie), a warning is issued and the coarse-map
estimate is reported.

### Preparing the windows

For each order `CCF2d` calls `todcor.prepare_window` with v_max = the largest
|trial velocity| plus the fine map's reach (`refine_span` coarse steps). Both
templates are resampled linearly onto the order's ln λ grid extended past each
end by v_max plus a 20-pixel ramp to zero, so a shifted template is read from
its real spectrum: nothing wraps around, no line goes missing at the order's
edges, and no taper is applied. Observed pixels whose templates do not reach
v_max beyond them are dropped (as in UNICOR), never filled; an order with
fewer than 2 covered pixels raises. The continuum level of the data and of
the templates does not matter: each order's offset is a free parameter.
`CCF2d.best_fit_spectrum` returns the model with that fitted offset.

## Side by side

| | UNICOR — `CCF1d` | TODCOR — `CCF2d` |
| --- | --- | --- |
| Model | one template, scaled (Pearson C per order) | two templates at (v1, v2), amplitudes a1, a2 fitted |
| Orders | per-order CCFs combined by Zucker's ML rule; each order has its own scale | one (v1, v2) for all orders; (a1, a2) shared (statistics summed) or per order (`amplitudes='per_order'`); a free offset per order |
| Velocity grid | lags of the ln λ grid, ±`vel_bound` (default 150 km/s, dv 0.1 km/s) | coarse (v1, v2) map, `dvel` 1 km/s, then a fine 41 × 41 map |
| `output='ccf'` | `Corr['corr']` per order, `CorrCombined['corr']` = ML(v) | `Corr2d['map']` = `Corr2d['R']` (R ∈ [0, 1]) |
| `output='loglike'` | Zucker ln L, `Corr['loglike']`, `CorrCombined['loglike']` | Gaussian ln L = −χ²/2, `Corr2d['map']` = `Corr2d['logL_profile']` |
| Always stored | both `'corr'` and `'loglike'` | `'R'`, `'logL_profile'`, `'logL_marginal'`, `'a1'`, `'a2'`, `'chi2'` maps; the same under `'fine'` |
| RV | maximum of a quintic spline through the combined CCF's lags: `CorrCombined['RV']` | maximum of a degree-6 polynomial fitted to the fine ln L map: `Corr2d['RV1']`, `['RV2']` |
| Error | Zucker ML error from the CCF curvature, N = native pixels: `CorrCombined['eRV']` | local Cramér–Rao bound from the ln L Hessian: `'eRV1'`, `'eRV2'`, `'rho12'`, `'cov'` |
| Noise level | not needed (implicit in C) | `sigma` per order, estimated by default; `'sigma'`, `'chi2_red'` |
| Peak value | `CorrCombined['peakCorr']` (CCF at the vertex) | `'R_best'` (R at the refined RVs) |
| Amplitudes | — | `'a1_best'`, `'a2_best'` (one per order under `'per_order'`) |
| Bookkeeping | `'n_eff'`, `'vel'`, `'units'`, `'velocity_convention'`, `'output'` | `windows[i]` (`'n_eff'`, `'oversampling'`, `'nu'`, `'sigma'`, ...), `stats`, `crlb`, `'velocity_convention'`, `'output'` |
| Time-series driver | `Observations.calc_rv_against_template` → `vels`, `evels`, `ccf_peaks` (and `time_series.calculated_vrad_list`, `errors`) | `Observations.calc_rv_against_two_templates` → `vels1/2`, `evels1/2`, `amps1/2` |

Every key is documented in [api.md](api.md) under `CCF1d.cross_correlate`,
`CCF1d.combine` and `CCF2d.cross_correlate`.

## References

- Zucker, S. 2003, MNRAS, 342, 1291 — the ML combination of orders and the RV
  uncertainty.
- Engel, M., Shahaf, S. & Mazeh, T. 2017, PASP, 129, 065002 — UNICOR.
- Zucker, S. & Mazeh, T. 1994, ApJ, 420, 806 — TODCOR.
