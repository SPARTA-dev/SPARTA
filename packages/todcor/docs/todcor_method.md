# TODCOR — method & conventions

Single source of truth for the two-component (double-lined) velocity fit
implemented in the `todcor` package. It fits the radial velocities — and relative
amplitudes — of *both* components of an SB2 against the actual template
spectra: TODCOR (Zucker & Mazeh 1994) recast as a Gaussian profile likelihood.
The same sufficient statistics give Zucker & Mazeh's two-dimensional
correlation `R(v1, v2)` (§4, *The two-dimensional correlation*), the default
primary map.

The implementation is split into `unicor.dsp` and `unicor.ccf` (grid,
resampling, coverage and broadening primitives, shared with UNICOR),
`todcor.simulate` (forward model) and `todcor.estimator` (the derivation).
SPARTA's `sparta.ccf2d.CCF2d` adds the `Spectrum`/`Template` orchestration on
top — the double-lined counterpart of UNICOR's `CCF1d`. The `todcor` namespace
re-exports the full public API, which is what the example notebooks use. Note
that `rotational_kernel` is a *velocity-space* Gray broadening kernel — a
real-space convolution kernel, not a Fourier-space amplitude.

Conventions: velocities and the log-λ shift `s` in km/s; spectra on a uniform
log-λ (constant-velocity-step `dv`) grid, with `dv = c·Δ` for the grid's
*actual* ln λ step `Δ`; fluxes in the data's own units (any constant level);
one **window** = the observed pixels of one echelle order (or of one
hand-picked line region).

## 1. Model

On a window of `n` observed pixels the data are modelled as two shifted,
scaled templates plus a free continuum offset `c0`:

    d_j = a1·G1_j(s1) + a2·G2_j(s2) + c0 + noise_j        (j = 1 … n)

where `G_i(s)` is template *i* shifted by `s_i = c·ln(λ_obs/λ_rest)` and read on
the window's pixels (§2). On a uniform log-λ grid a Doppler shift is a
translation. The amplitudes `(a1, a2)` are the linear light/dilution fractions
of each component's lines (for templates normalised like the data); the offset
absorbs any constant level, so neither the data nor the templates need a
particular continuum.

**Velocity convention.** What velocity a shift `s` is depends on the Doppler
convention (`velocity_convention`, via `unicor.dsp.shift_kms`):

| convention | `λ_obs/λ_rest` | `s(v)` | `v(s)` |
| --- | --- | --- | --- |
| `'cz'` (default; IAU 2000, IRAF fxcor) | `1 + v/c` | `c·ln(1 + v/c)` | `c·expm1(s/c)` |
| `'relativistic'` | `√((1 + v/c)/(1 − v/c))` | `c·artanh(v/c)` | `c·tanh(s/c)` |

`expm1(x) = eˣ − 1`, computed without cancellation for small `x`. For the same
spectrum the `'cz'` velocity exceeds the `'relativistic'` one by `≈ v²/2c`
(37.5 m/s at 150 km/s). `s` is measured in units of the grid's actual step:
a grid built with `Δ = ln(1 + dv/c)` (SPARTA's `Spectrum.interpolate`) has
`c·Δ = dv·(1 − dv/2c)`, so `CCF2d` passes each order's measured `c·Δ`
(`unicor.dsp.log_grid_step`), not the nominal `dv`.

## 2. The window: observed pixels, extended templates

`prepare_window(wave, flux, t1_wave, t1_flux, t2_wave, t2_flux, v_max=, ramp=20)`
builds one window (`TodcorWindow`):

- **Observed pixels only.** The χ² is a sum over the window's observed
  pixels; nothing is padded, filled or tapered.
- **Templates extended past the window.** Both templates are resampled
  (linearly, `unicor.dsp.resample`) onto the window's grid extended by
  `e = k + ramp` pixels on each side (`unicor.ccf.extend_log_grid`), where
  `k = ⌈max|s(±v_max)|/dv⌉ + 1` is the largest trial shift in pixels (over
  both signs and both conventions: under 'cz', −v_max shifts furthest) and
  `v_max` the largest |trial velocity|, refinement included. An even window
  gets one more pixel on the right, so the extended length is always odd.
  `TodcorWindow.reach = k`; `window_stats(..., reach=k)` refuses any trial
  shift beyond it, which would read the ramp.
- **Coverage.** Observed pixels whose templates do not reach `k` pixels beyond
  them are dropped (`unicor.ccf.template_coverage`, as UNICOR does), never
  filled with a continuum; fewer than 2 covered pixels raise `ValueError`.
- **The ramp.** The outermost `ramp` pixels of each extended template fall to
  zero along a raised cosine. They are never read (below); they only make the
  extended template's two ends meet at zero.
- Medians are subtracted from the data and from the templates only to keep the
  numbers small: the offset `c0` is profiled out (§3). A template flat to 10⁻⁹
  of its own scale over the pixels a shift can read has no lines in the window
  and is set to exactly zero.

**The shift** (`_shifted_rows`). Each extended template is transformed once
(`rfft`); its shift by `p = s/dv` pixels is the phase factor
`exp(−2πi f p)` (`f` in cycles per pixel, the shift theorem) and an inverse
transform, read on the window's `n` pixels. This is band-limited interpolation,
exact at whole-pixel shifts. The shift is circular on the *extended* array, but
for `|p| ≤ k` every pixel read comes from the template's real neighbouring
spectrum; the wrap-around junction stays at least `ramp` pixels outside the
window. The odd extended length leaves no Nyquist mode, whose phase factor
would not keep the shifted template real. With `e = 0` (templates given on the
window itself) `window_stats` falls back to a circular shift on the window.

**Why.** The earlier estimator shifted window-length templates circularly, so a
line pushed past one end re-entered at the other, and tapered every window
(Tukey) to damp that. Measured on noise-free analytic line forests (R = 60,000,
templates sampled 10× finer than a 0.5 km/s analysis grid; largest velocity
error over the tested velocities):

| window | earlier: taper 0.3 + circular shift, exact optimum | earlier pipeline (quadratic peak patch, §7) | current pipeline |
| --- | --- | --- | --- |
| 50 Å | 11.8 m/s | 6.2 m/s | ≤ 0.023 m/s |
| 200 Å | 1.9 m/s | 3.9 m/s | ≤ 0.016 m/s |

The exact minimum of the current χ² (continuous optimiser) is within
0.012 m/s of the truth; that remainder is the linear resampling of the
templates. The amplitudes are recovered exactly (0.7000 / 0.3000). Ringing
from the Fourier shift of the extended templates is < 10⁻⁵ m/s, even with
strong lines at the array ends and no ramp.

In `CCF2d`, `v_max = max(|v1_grid|, |v2_grid|) + refine_span · (largest coarse
step)`: the coarse map plus the fine map's reach around an edge node (§7).

## 3. The six sufficient statistics

For one window, with the pixel weight `w = 1/(σ²·M)` (σ the native per-pixel
noise, `M` the oversampling, §8), the shifted templates `G_i(s)` on the window,
`m_X = w·Σ_j X_j` and `W0 = w·n`, `window_stats` returns

    SD         = w Σ d²     − m_d² / W0                    (scalar)
    S11(v1)    = w Σ G1²    − m_G1² / W0                   (n1,)
    S22(v2)    = w Σ G2²    − m_G2² / W0                   (n2,)
    b1(v1)     = w Σ d G1   − m_G1 m_d / W0                (n1,)
    b2(v2)     = w Σ d G2   − m_G2 m_d / W0                (n2,)
    A12(v1,v2) = w Σ G1 G2  − m_G1 m_G2 / W0               (n1, n2)

(sums over the window's `n` pixels). Each is a weighted inner product of
mean-subtracted arrays — the offset `c0` projected out, as in a Pearson
correlation. `b1, b2` are the data–template cross terms, `A12` the
template–template one. The template powers `S11, S22` depend on the velocity,
because the part of the template inside the window moves with the shift, and
`A12` depends on both velocities, not only on their difference. **Given these
six the raw spectrum is never needed again** — the sufficiency claim and the
point of the design. They are indexed by *velocity*, not by pixel — the key to
§6. Cost: one shift per trial velocity and template, then `A12` is one
`(n1 × n)·(n × n2)` matrix product.

## 4. The χ² quadratic form and profiled amplitudes

Minimising `w Σ (d − a1 G1 − a2 G2 − c0)²` over `c0` leaves

    χ²(v1,v2,a1,a2) = SD − 2(a1 b1 + a2 b2)
                          + a1² S11 + 2 a1 a2 A12 + a2² S22

(`chi2_from_stats`). Linear in `(a1, a2)`, so at each `(v1, v2)` the amplitudes
solve in closed form. Normal equations `N a = b` with
`N = [[S11, A12], [A12, S22]]`, `b = (b1, b2)`:

    det = S11·S22 − A12²          (≥ 0 by Cauchy–Schwarz)
    a1  = (S22 b1 − A12 b2) / det
    a2  = (S11 b2 − A12 b1) / det

A component with no lines in the window (its template zeroed by
`prepare_window`, so `S_ii = 0`) gets amplitude 0, and where the pair is
degenerate on a node (`det < 10⁻¹⁰·S11·S22`) the better single-component fit is
used; both rules are invariant to the templates' scale.
Substituting back gives `logL_profile = −χ²/2` over the `(v1, v2)` grid, and its
normalised form, the two-dimensional correlation `R` (below).
`profile_from_stats` does this; `todcor_map` is the single-window convenience
wrapper `profile_from_stats(window_stats(...))`.

### The two-dimensional correlation `R` (`output='ccf'`)

At the profile amplitudes `â = N⁻¹b` the misfit is `χ²_min = SD − bᵀN⁻¹b`, so the
fraction of the data's weighted power that the best-fit pair of templates explains
is

    R² ≡ 1 − χ²_min / SD = bᵀN⁻¹b / SD
       = (S22 b1² − 2 A12 b1 b2 + S11 b2²) / (det · SD)
       = (c1² − 2 c1 c2 c12 + c2²) / (1 − c12²)

with the normalised correlations `c1 = b1/√(S11 SD)` (data vs template 1 at `v1`),
`c2 = b2/√(S22 SD)` and `c12 = A12/√(S11 S22)` (template 1 at `v1` vs template 2
at `v2`); the last line divides numerator and denominator by `S11 S22 SD`. It
is Zucker & Mazeh's (1994) `R(s1, s2)` with the light ratio at its optimum at
every node — their analytic maximisation over α is the 2×2 least-squares fit
here. Properties:

- **Pearson-normalised.** Every inner product is of mean-subtracted arrays
  (§3); `σ` and the oversampling factor cancel. `R` is free of the noise scale;
  with windows pooled (§6) it is the correlation of the weighted ensemble, so
  only the windows' *relative* weights enter.
- **Bounded.** `0 ≤ χ²_min ≤ SD` (`a = 0` is feasible, χ² is a sum of squares), so
  `R ∈ [0, 1]`, and `R = 1` exactly for a noiseless composite of the two
  templates at a node. Under `nonneg=True`, `χ²_min` is the constrained minimum
  and `R² = 1 − χ²/SD` still holds, with the same bounds; on the edge `a2 = 0`,
  `R = max(c1, 0)`.
- **One template.** If template 2 is orthogonal to the data and to template 1 at
  every shift (`c2 = c12 = 0`), `R = |c1(v1)|`: the one-dimensional normalised
  CCF. `c1` at a whole-pixel shift is the Pearson correlation, over the
  window's observed pixels, of the data with the shifted extended template —
  UNICOR's per-lag CCF (`unicor.ccf_vs_velocity`) of the same extended
  template.
- **Same peak as the likelihood.** At fixed `SD`, `logL_profile = −(SD/2)(1 − R²)`,
  monotonic in `R`: the two maps share their argmax, so the velocities, the
  amplitudes and the CRLB (§7, which needs the log-likelihood's curvature) do not
  depend on which map is the primary one.

`profile_from_stats` and `todcor_map` always return both maps (`'R'`,
`'logL_profile'`, plus `'logL_marginal'`) and `best['R']` at the argmax;
`output='ccf'` (default) or `'loglike'` selects which is `'map'`, the primary
surface. `R` costs one element-wise operation on `χ²`. `CCF2d` stores both maps
on its coarse and fine grids, `R_best` at the refined peak, and draws the
primary map. Why `R` is the default: on a high-S/N spectrum `ΔlnL` across the
map is enormous and only the node at the peak is within a few units of the
maximum, while `R` keeps the ridges where one template matches one star and
the crossing where both do.

## 5. Profile vs marginal likelihood

Since `χ²(a) = χ²_min + (a − â)ᵀ N (a − â)`, the likelihood in the amplitudes is
Gaussian with **precision matrix N** (the ½ in `exp(−χ²/2)` cancels the 2 in the
Hessian). Marginalising under a **flat, improper, uniform prior on
`(a1, a2) ∈ ℝ²`** is the Gaussian integral `∫d²a exp(−½(a−â)ᵀN(a−â)) = 2π/√det N`,
so

    logL_marginal = logL_profile − ½ ln|det|

(the amplitude-posterior width; the `ln 2π`, and the offset's own Gaussian
integral, are grid constants). `det` depends on `(v1, v2)` through `S11`,
`S22` and `A12`. The reported peak (`best`) is always the **profile** argmax —
the marginal peak can shift by the `det`-gradient.

**Non-negative amplitudes (`nonneg=True`).** Light fractions cannot be negative,
so the amplitudes are constrained to the quadrant `a ≥ 0` (per-node 2-variable
NNLS: a node whose unconstrained solution leaves the quadrant takes the better
single-component fit), removing the sign-flip degeneracy that lets an
under-constrained component chase noise. The *exact* marginal would then be an
**orthant-truncated** Gaussian, `logL_profile − ½ ln|det| + ln Φ_N(a ≥ 0)`. The
code does not evaluate that: in constrained mode `logL_marginal` is set equal
to `logL_profile`, dropping **both** the `−½ ln|det|` and the truncation term
`ln Φ_N`. With per-window amplitudes (§6) `logL_marginal` is likewise set equal
to `logL_profile`. Practical impact is small because velocities come from the
profile peak regardless.

## 6. Combining windows

Because χ² is additive over independent-noise windows and the statistics are
indexed by *velocity*, the windows of one epoch evaluated on one `(v1, v2)` grid
combine into one joint fit with **shared `(v1, v2)`**, in one of two ways
(`joint_fit(..., amplitudes=)`, `CCF2d.cross_correlate(..., amplitudes=)`):

- **`amplitudes='shared'`** (default): one `(a1, a2)` for all windows. Each
  window's χ² is linear in its six statistics, so the windows combine by
  **summation** of the statistics (`combine_stats`), then `profile_from_stats`.
- **`amplitudes='per_order'`**: each window *i* gets its own `(a1_i, a2_i)` —
  its own light ratio — at every `(v1, v2)` (`profile_per_window`):

      logL = −½ Σ_i min_{a_i} χ²_i,        R² = 1 − Σ_i χ²_i,min / Σ_i SD_i

  It costs two free parameters per window. Use it when the light ratio of the
  two stars changes across the spectrum (different temperatures over a wide
  wavelength range); with a constant ratio the shared fit is tighter. `CCF2d`
  then returns `Corr2d['a1_best']`, `['a2_best']` with one value per window.

Windows may differ in `dv`, resolution, or instrument. In SPARTA the natural
windows are the **spectral orders** of one observation, which is exactly what
`CCF2d.cross_correlate` pools; hand-picked line windows (He II 4686, N IV
4058, …) work the same way. The statistics are linear in the weight `1/σ²`,
so `combine_stats(stats, weights=w)` — each window's six statistics times
`w_i`, `ν` unchanged — equals recomputing window *i* at noise `σ_i/√w_i`:
windows are re-weighted without new shifts (§8). `profile_per_window` takes
the same `weights`.

Two usage constraints — the first is checked at runtime (shape guard in
`combine_stats`), the second cannot be:

1. **Same velocity grid.** `b1/b2/A12` (and `S11/S22`) are summed element-wise,
   so every window must be evaluated on the *same* `v1_grid`/`v2_grid`
   (identical velocity nodes; `dv`/`R`/instrument may still differ).
2. **Same epoch.** All pooled windows must share one true `(v1, v2)`. **Do not
   combine across epochs** — orbital motion changes the velocities, so a
   shared-`v` fit across epochs is physically meaningless.

**Measured** (`CCF2d` end to end on a simulated two-order HARPS-like SB2, 12
epochs, S/N 100): primary RV rms 1.11 m/s against a median eRV of 1.17 m/s;
`'shared'` and `'per_order'` gave identical results (the simulated light ratio
is constant).

**Within a window** the amplitudes are constant: for chemically distinct lines
(e.g. an absorption primary and an emission secondary) the light/line-strength
ratio is line-dependent, so one amplitude pair per window is a modelling
approximation, not an exact description.

## 7. Peak refinement and the Cramér–Rao errors

`crlb_from_map(logL, v1_grid, v2_grid, half=4)` fits a 2-D polynomial of total
degree 6 (least squares, in coordinates scaled to the patch) to `logL_profile`
over the `(2·half + 1)²` nodes around its argmax, and takes its maximum (Newton
steps from the argmax node, analytic gradient and Hessian). For a Gaussian
likelihood `−Hessian(logL)` is the Fisher information and its inverse is the
`(v1, v2)` covariance; because `logL` is the *profile* over the (linear)
amplitudes, this is already the full CRLB (profile = marginal for
linear-Gaussian nuisances). The amplitudes are linear only at fixed velocity,
so it is a **local** CRLB — valid near an isolated, well-resolved peak; it
degrades for blended / near-degenerate peaks (`A12² → S11·S22`, `det → 0`).
`peak_refined` is the polynomial's maximum. The errors are `nan`, and
`peak_refined` the argmax node, when the Hessian is not negative definite or
the maximum lies more than one grid step from the argmax; near the map's edge
the patch is clipped and the degree drops to fit it (≥ 2); on the edge itself
(`on_edge`) nothing is refined.

**Why degree 6, and why a fine map.** `logL` follows the *squared*
correlation, so its peak is only ~`σ_ccf/√2` wide, and it is not quadratic over
a patch of useful size: the quadratic patch of the earlier pipeline misread it
by up to 6.2 m/s (§2 table); with degree 6 the whole pipeline is within
0.023 m/s of the truth.
`CCF2d` runs two passes (one call of `joint_fit`, its fine grid from
`refine_grid`):

1. the coarse map (default `dvel` = 1 km/s) only *locates* the peak;
2. the six statistics are re-evaluated **exactly** (`window_stats` on the
   prepared windows) on a fine local grid — ±2 coarse steps around the coarse
   argmax, step `dvel/10` (41 × 41 nodes; `refine_span`, `refine_factor`) —
   and `crlb_from_map` (±4 fine nodes, `crlb_half`, i.e. ±0.4 `dvel`) is
   applied to the fine map.

The coarse step must already resolve the peak (`dvel ≲ σ_ccf`) for the coarse
argmax to be trustworthy. The amplitudes and χ² are then evaluated exactly at
the refined `(v1, v2)`, so the reported RVs and amplitudes belong to one point.
If the coarse argmax is on the grid edge no refinement is done (warning, `nan`
errors); if the fine argmax lands on the fine-map edge (a ridge or a tie), a
warning is issued and the coarse-map estimate is reported.

## 8. The noise budget

The Monte Carlo figures in this section were measured with the earlier
estimator (circular shift, Tukey taper α = 0.1, quadratic peak patch). The
noise model they test — oversampling, `ν`, per-order σ — is unchanged, but
they have not been re-run.

**Oversampled grids.** Resampling a spectrum onto a grid `M = dv_native/dv`
times finer (e.g. ~1 km/s pixels onto the 0.1 km/s CCF grid) adds samples but
no information: ~M analysis pixels share each native pixel's noise. White
weights `1/σ²` would count the Fisher information M times and shrink the CRLB
by `√M` — Monte Carlo scatter / CRLB was ~2.5. `window_stats(...,
oversampling=M)` divides the weight by M (`w = 1/(σ²M)`, equivalently
`σ → σ√M`), which restores the native-pixel information for any line resolved
by the native sampling; `CCF2d` takes `M_i = max(1, dv_native,i/dv)` per order
from `Spectrum.native_steps()` (the coarsest sampling the spectrum has had).
Monte Carlo scatter / CRLB was then 0.92 / 0.97 for the two components.

**Noise degrees of freedom `ν`.** Each window also returns `nu`, the expected
χ² of its pure-noise residual:

    ν = n · mean(noise_psd) / M − 1          (one for the offset)

For white noise on the analysis grid (`noise_psd = 1`, `M = 1`) that is
`n − 1`. Resampled noise is not just correlated but *smoother*: its variance per
analysis pixel is `κσ²` with `κ = mean(noise_psd)` = 2/3 for linear
interpolation and ≈ 0.87 for the cubic spline of `Spectrum.interpolate`.
Dividing χ² by the bare independent-pixel count `n_eff = n/M` would therefore
leave χ²_red ≈ 0.87 (2/3 for linear interpolation) for honest noise.
`noise_psd` — from `resampled_noise_psd`, the averaged power spectrum of the
resampler's impulse responses — gives the noise power the fit actually sees,
and

    χ²_red = χ²_best / (Σ_windows ν − 4)              (amplitudes='shared')
    χ²_red = χ²_best / (Σ_windows ν − 2 − 2·n_win)    (amplitudes='per_order')

(two shared velocities, plus two amplitudes shared or per window) is ≈ 1 for
honest noise (1.004 in the Monte Carlo above).

The impulse response of a native pixel depends on its sub-pixel phase between
two analysis nodes, so `CCF2d` probes the resampler at 32 phases (cached per
`(dv_native, dv)`) and averages them weighted by how many of the order's native
pixels sit at each phase. This matters most on a grid *coarser* than native:
nodes that fall on native pixels pass their noise unchanged (κ = 1, white),
nodes between them average it down and correlate neighbours (κ ≈ 0.87 over
random phases). A grid commensurate with a log-λ native grid (e.g. exactly 2×
coarser) has only a few phases, a linear-λ native grid sweeps them all. When
`CCF2d` resamples the spectrum itself it measures the phases; for a spectrum
resampled beforehand it assumes them uniform (which can be off by ~15 % for a
commensurate log grid). Monte Carlo (100 realisations, 1 km/s native, log-λ and
linear-λ native grids): χ²_red = 1.006–1.014 for dv = 0.25, 0.5, 1, 1.5, 2,
3 × native.

**Per-order noise (`sigma=None`).** χ² and the map curvature scale as `1/σ²`,
so one common σ would not move the argmax and would only rescale the errors by
`√χ²_red`. But echelle orders differ in S/N by 2–5×, and equal weights then
waste the clean orders: with two orders of noise 0.003 and 0.03 the RV scatter
was 4.6–4.9× that of the fit with the true per-order σ (Monte Carlo, 200
realisations; the errors were honest, the weighting was not). `CCF2d`'s
default `sigma=None` therefore fits twice:

1. **Pass 1** fits every order at unit σ and locates the joint peak (coarse +
   fine maps), with the amplitudes and every window's statistics evaluated
   exactly at the refined `(v1, v2)`.
2. Each window's share of the joint χ² there, `χ²_i = chi2_from_stats(stats_i,
   a1, a2)` (with its own amplitudes under `'per_order'`; the shares sum to the
   joint χ²), gives its own noise `σ_i² = χ²_i / dof_i` (`window_noise`), with

       dof_i = ν_i (1 − 4/Σν)              (amplitudes='shared')
       dof_i = ν_i (1 − 2/Σν) − 2          (amplitudes='per_order')

   The shared parameters' dof are charged to the windows in proportion to
   `ν_i`; under `'per_order'` each window also pays its own two amplitudes. The
   exact charge — the trace of the window's block of the hat matrix — follows
   its line content instead, but the two differ by at most `4/ν_i` in `σ_i²`
   (≲ 0.2 % for an order with `ν ≈ 2000`). The charges sum to the fitted
   parameters, so with one window `σ² = χ²_red`.
3. **Pass 2** weights each window by `1/σ_i²` and re-profiles. No shift is
   repeated for this: the statistics are linear in the weight, so
   `combine_stats(stats, weights=1/σ_i²)` (or `profile_per_window` with those
   weights) re-pools the per-window coarse statistics (kept for the call
   anyway), and the per-window fine-map statistics are cached by coarse node
   and re-pooled when the coarse argmax has not moved. Only the evaluation at
   the new refined peak (one 1 × 1 `window_stats` per window) is new. The RVs,
   amplitudes and errors come from pass 2, at the noise `σ_i`; the maps,
   `self.stats` and `self.crlb` are on that noise scale, and
   `Corr2d['sigma']` / `windows[i]['sigma']` report the per-window estimates.

A window with fewer than `min_noise_dof` = 50 dof (its σ would scatter by
> 10 %), or with no residual to measure (`χ²_i ≤ 0`, which would otherwise get
an infinite weight), gets the common estimate `Σχ²_i/Σdof_i` over the windows
that did measure their own (all windows if none did), with a warning and
`windows[i]['sigma_fallback'] = True`. An order whose residual is dominated by
template mismatch rather than noise gets a large σ_i and is down-weighted
accordingly.

Monte Carlo (two 35 Å orders on 1 km/s native grids resampled to 0.1 km/s,
lines clear of the order edges, 200 realisations, shared amplitudes): with
noise 0.003 and 0.03 per order, per-order σ gave the true-σ fit's RV scatter
(0.00742 and 0.0106 km/s; std/eRV 1.07 and 0.98), against 0.0340 and
0.0521 km/s with one common σ; with equal noise (0.01, 0.01) the two agreed.
The σ_i estimates were within 1.5–2.3 % (rms) of the truth. One pass leaves a
small bias: the cleaner window's pass-1 residual also holds the misfit of a fit
pulled by the noisier window, so its σ runs high by 3 % at a 10× noise
contrast (0.5 % at 3×), its errors are that much conservative, and χ²_red at
the pass-2 peak is ≈ 0.97 instead of 1. A second re-estimation removes it but
moves the RVs by only ~0.01 eRV, so `CCF2d` stops after one.

An explicit `sigma` (scalar or per order) is taken at face value — one pass, no
re-estimation — and χ²_red is a consistency check.

## 9. Assumptions and caveats

Exact **only** under these; the first is the one to watch:

- **White native noise `σ` per native pixel, resampled once by an
  interpolator linear in the flux.** `oversampling`, `noise_psd` and the
  native-pixel count behind the CRLB model that resampling (in `CCF2d`: one
  `Spectrum.interpolate` call from a uniform native grid — the cubic spline of
  `sparta.spectrum.resample_flux` — probed with unit impulses). **After
  `filter_spectrum` / `preprocess` this calibration does not hold**:
  band-pass filtering, per-order normalisation and the double resampling
  (`preprocess`'s linear-λ resampling, then `CCF2d`'s log-λ one) remove and
  correlate noise power that ν and the CRLB do not know about. SPARTA's
  `CCF2d` divides `oversampling` by the share of native noise the band-pass
  kept (`Info['NoiseFraction']`); without that the errors were ~0.55× the RV
  scatter, with it they are 1.1–1.2× (RV scatter / eRV 0.86 primary, 0.90
  secondary, simulated HARPS-like SB2s, 7 × 50 epochs). χ²_red then drifts from 1 and `sigma=None` absorbs only each
  order's mean scale. Calibrate the errors by simulation (or a constant star)
  for preprocessed spectra.
- **White weights.** The Fisher weights ignore the colour of the resampled
  noise. An interpolator that attenuates in-band noise (linear interpolation's
  passband droop) leaves the CRLB mildly conservative; so does an analysis
  grid coarser than native whose nodes fall between native pixels (smoothed,
  correlated noise): Monte Carlo RV scatter / eRV = 0.8–0.9 at 1.5× native
  (log- and linear-λ native grids) and 2× (linear-λ), against 0.97–1.06 at 3×
  and at 2× on a log-λ native grid, whose nodes then all fall on native pixels
  (earlier estimator, §8).
- **Templates that reach past the window** by the largest trial shift. Pixels
  they do not reach are dropped (§2), which costs information but not
  accuracy.
- **Linear template resampling.** The templates are interpolated linearly onto
  the analysis grid; on the noise-free test of §2 that is the remaining
  0.012 m/s.
- **χ²_red counts the fitted parameters** (§8). When `nonneg=True` clamps an
  amplitude to zero one fewer is free; the difference is negligible against
  `Σν`.
- **Per-order noise under `sigma=None`** (§8) is one noise level per order:
  noise that varies *within* an order (e.g. with the blaze) is averaged. Each
  order is charged the shared dof in proportion to `ν_i`, not by its line
  content (≤ `4/ν_i` in `σ_i²`); one re-estimation leaves the cleaner of two
  very unequal orders with a few-per-cent conservative σ. Orders below
  `min_noise_dof` fall back to the common estimate. An order whose residual is
  template mismatch rather than noise is down-weighted as if it were noisy.
- **Local CRLB** (§7): valid for an isolated, well-resolved peak sampled by
  the fine map.
- **Flat ℝ² amplitude prior** for the marginal map (§5).
- **Shared velocity grid** for `combine_stats` and `profile_per_window` (§6).
- **One `(a1, a2)` per window** (or for all windows, `'shared'`) is an
  approximation where the line-strength ratio varies line by line (§6).
- **Non-negative and per-window marginals** drop terms (§5).

## 10. Function map

All of these except `CCF2d` are reachable from the `todcor` namespace; the
"lives in" column says where they are defined.

| Function | Lives in | Role |
| --- | --- | --- |
| `log_lambda_grid`, `resample` | `unicor.dsp` | grid + linear/cubic resampling (auxiliary) |
| `broaden_gaussian_residual`, `rotational_kernel`, `broaden_rotational_residual` | `unicor.dsp` | template broadening |
| `shift_kms`, `doppler_factor`, `velocity_from_log_shift` | `unicor.dsp` | the log-λ shift `s = c·ln(λ_obs/λ_rest)` and its inverse, per velocity convention (§1) |
| `log_grid_step` | `unicor.dsp` | a log grid's actual step `c·Δ` (km/s), verified uniform |
| `shift_template_flux`, `composite_flux` | `todcor.simulate` | forward model / simulate composites (shift by physical resampling) |
| `prepare_window` → `TodcorWindow` | `todcor.estimator` | one window's observed pixels and the two extended, ramped templates (§2) |
| `window_stats` | `todcor.estimator` | the six sufficient statistics (+ noise dof `nu`) for one window (§3, §8) |
| `resampled_noise_psd` | `todcor.estimator` | noise spectrum of resampled white native noise → `noise_psd` (§8) |
| `combine_stats` | `todcor.estimator` | (weighted) sum of statistics across windows → shared-amplitude joint fit (§6) |
| `chi2_from_stats` | `todcor.estimator` | χ² of statistics at given amplitudes — per-window shares of the joint χ² (§8) |
| `profile_from_stats` | `todcor.estimator` | profile/marginalise amplitudes → `R` and logL maps (§4–5) |
| `profile_per_window` | `todcor.estimator` | each window's own amplitudes, χ² maps summed (§6) |
| `todcor_map` | `todcor.estimator` | single-window convenience wrapper (§4) |
| `crlb_from_map` | `todcor.estimator` | peak and Cramér–Rao velocity errors from a degree-6 polynomial (§7) |
| `refine_grid` | `todcor.estimator` | the fine local `(v1, v2)` grid around a coarse node (§7) |
| `joint_fit` → `TodcorFit` | `todcor.estimator` | one pass of the multi-window fit: pool (`amplitudes=`), coarse map, fine map, CRLB, values at the refined peak (§6, §7) |
| `window_noise` | `todcor.estimator` | per-window noise from its share of the joint χ² (§8, `sigma=None`) |
| `model_flux` | `todcor.simulate` | the best-fit `a1·G1(v1) + a2·G2(v2)` on a window's pixels, by the fit's own shift (no offset), for residual plots |
| `CCF2d` | `sparta.ccf2d` | `Spectrum` + two `Template`s → coarse + fine maps, RVs, errors, χ²_red, per-order noise (orders as windows) |

## 11. Reference

Zucker, S. & Mazeh, T. 1994, "Study of Spectroscopic Binaries with TODCOR. I.",
ApJ 420, 806.
