# Fourier-domain TODCOR — method & conventions

Single source of truth for the two-component (double-lined) velocity fit
implemented in the `todcor` package. It fits the radial velocities — and relative
amplitudes — of *both* components of an SB2 by transforming the actual template
spectra, TODCOR (Zucker & Mazeh 1994) recast as a Gaussian profile likelihood.

The implementation is split into `unicor.dsp` (grid / FFT / Parseval /
broadening primitives, shared with UNICOR), `todcor.simulate` (forward model)
and `todcor.estimator` (the derivation). SPARTA's `sparta.TODCOR.CCF2d` adds the
`Spectrum`/`Template` orchestration on top — the double-lined counterpart of
UNICOR's `CCF1d`. The `todcor` namespace re-exports the full public API, which
is what the example notebooks use. Note that `rotational_kernel` is a
*velocity-space* Gray broadening kernel — a real-space convolution kernel, not a
Fourier-space amplitude.

Conventions: velocities and the log-λ shift `s` in km/s; ζ (Fourier conjugate)
in s/km; spectra on a uniform log-λ (constant-velocity-step `dv`) grid; residual
flux `g = flux − 1`; Fourier grid `np.fft.rfftfreq(N, d=dv)[1:]` (DC dropped,
positive frequencies only). The math below was independently re-derived and
verified against the code (2026-07-07).

## 1. Model

On a uniform log-λ grid a Doppler shift is a translation, so in the Fourier
domain it is a phase factor (shift theorem). Unlike the hot-star SB2 work — where
the modelled factor was the transform of a near-δ line — here the **template
spectra themselves** are transformed once and reused. With shifts
`s_i = c·ln(1 + v_i/c)`:

    g(u)   = a1·g1(u − s1) + a2·g2(u − s2)                       (velocity space)
    G̃(ζ)  = a1·G1(ζ)·exp(−2πiζ s1) + a2·G2(ζ)·exp(−2πiζ s2)     (Fourier space)

`G1, G2 = rfft` of the broadened, windowed residual templates; `D(ζ) = rfft` of
the data residual. The amplitudes `(a1, a2)` are the linear light/dilution
fractions of each component's lines.

## 2. Fourier grid and Parseval weights

Positive frequencies only, DC dropped (`rfft_positive`). Dropping ζ=0 also
removes the mean of the windowed residual — any residual continuum offset is
projected out for free.

Per-mode weights (`parseval_weights`) make the Fourier χ² equal the real-space
χ². For real, white, per-pixel noise `σ`, the positive interior modes carry
`w_k = 2/(N σ²)`; the DC and (even-N) Nyquist modes carry `1/(N σ²)`. Then

    Σ_k w_k |r_k|²  =  Σ_n r_n² / σ²        (Parseval, real residual r)

with the dropped DC corresponding to the removed mean. On a grid oversampled
relative to the native pixels the weights are further divided by the
oversampling factor (§8).

## 3. The six sufficient statistics

For one window the weighted χ² at **any** `(v1, v2, a1, a2)` is a fixed quadratic
form in six numbers — the window's sufficient statistics (`window_stats`):

    S11 = Σ_k w_k |G1_k|²                                        (scalar)
    S22 = Σ_k w_k |G2_k|²                                        (scalar)
    SD  = Σ_k w_k |D_k|²                                         (scalar)
    b1(v1) = Re Σ_k w_k conj(G1_k) D_k exp(+2πiζ_k s1)           (n1,)
    b2(v2) = Re Σ_k w_k conj(G2_k) D_k exp(+2πiζ_k s2)           (n2,)
    A12(v1,v2) = Re Σ_k w_k conj(G1_k) G2_k exp(−2πiζ_k (s2−s1)) (n1, n2)

`S11, S22, SD` are constants; `b1, b2` are the data–template cross-terms (one per
trial velocity); `A12` is the template–template cross-term. **Given these six the
raw spectrum is never needed again** — that is the sufficiency claim and the
whole point of the design. They are indexed by *velocity* (through `s_i`), not by
Fourier mode — the key to §7.

## 4. The χ² quadratic form and profiled amplitudes

Expanding `Σ_k w_k |D_k − a1 G1_k e^{−iφ1} − a2 G2_k e^{−iφ2}|²` (φ_i = 2πζ s_i):

    χ²(v1,v2,a1,a2) = SD − 2(a1 b1 + a2 b2)
                          + a1² S11 + 2 a1 a2 A12 + a2² S22

Linear in `(a1, a2)`, so at each `(v1, v2)` the amplitudes solve in closed form.
Normal equations `N a = b` with `N = [[S11, A12], [A12, S22]]`, `b = (b1, b2)`:

    det = S11·S22 − A12²          (≥ 0 by Cauchy–Schwarz; floored > 0 in code)
    a1  = (S22 b1 − A12 b2) / det
    a2  = (S11 b2 − A12 b1) / det

Substituting back gives `logL_profile = −χ²/2` over the `(v1, v2)` grid — the
TODCOR correlation map. `profile_from_stats` does this; `todcor_map` is the
single-window convenience wrapper `profile_from_stats(window_stats(...))`.

## 5. Profile vs marginal likelihood

Since `χ²(a) = χ²_min + (a − â)ᵀ N (a − â)`, the likelihood in the amplitudes is
Gaussian with **precision matrix N** (the ½ in `exp(−χ²/2)` cancels the 2 in the
Hessian). Marginalising under a **flat, improper, uniform prior on
`(a1, a2) ∈ ℝ²`** is the Gaussian integral `∫d²a exp(−½(a−â)ᵀN(a−â)) = 2π/√det N`,
so

    logL_marginal = logL_profile − ½ ln|det|

(the amplitude-posterior width; the `ln 2π` is a grid constant). The reported
peak (`best`) is always the **profile** argmax — the marginal peak can shift by
the `det`-gradient.

**Non-negative amplitudes (`nonneg=True`).** Light fractions cannot be negative,
so the amplitudes are constrained to the quadrant `a ≥ 0` (per-node 2-variable
NNLS), removing the sign-flip degeneracy that lets an under-constrained component
chase noise. The *exact* marginal would then be an **orthant-truncated** Gaussian,
`logL_profile − ½ ln|det| + ln Φ_N(a ≥ 0)`. The code does not evaluate that: in
constrained mode `logL_marginal` is set equal to `logL_profile`, dropping **both**
the `−½ ln|det|` and the truncation term `ln Φ_N`. Practical impact is small
because velocities come from the profile peak regardless.

## 6. Nyquist handling (even N)

For even `N` the Nyquist coefficient (ζ = 1/2dv) of a real signal must be real,
and the `irfft`-based model (`model_flux`) enforces that — but the complex phase
factors `exp(−2πiζ s)` used in `b1/b2/A12` do not. `window_stats` therefore
**drops the single Nyquist mode** from `D, G1, G2, w` alike. This keeps the fit
self-consistent and preserves the *exact* Parseval identity for the retained
subspace (the removed mode's power is negligible); as a secondary benefit it
removes a small shift-dependent (`∝ cos(πs/dv)`) velocity bias. Odd `N` has no
Nyquist mode. Regression-tested: exact-shift χ² → ~1e−12 for even and odd N.

## 7. Combining windows — the sufficient-statistic addition

Because χ² is additive over independent-noise windows and each window's χ² is
**linear** in its six statistics, and the statistics are indexed by *velocity*
(not Fourier mode), windows combine by **summation** of the six statistics
(`combine_stats`) into one joint fit with **shared `(v1, v2, a1, a2)`**. Windows
may differ in `dv`, resolution, or instrument. In SPARTA the natural windows are
the **spectral orders** of one observation, which is exactly what
`CCF2d.CrossCorrelateSpec` pools; hand-picked line windows (He II 4686, N IV
4058, …) work the same way — one window per line → `combine_stats` → a joint
velocity fit.

Two usage constraints — the first is checked at runtime (shape guard in
`combine_stats`), the second cannot be:

1. **Same velocity grid.** `b1/b2/A12` are summed element-wise, so every window
   must be evaluated on the *same* `v1_grid`/`v2_grid` (identical velocity nodes;
   `dv`/`R`/instrument may still differ).
2. **Same epoch.** All pooled windows must share one true `(v1, v2)`. **Do not
   combine across epochs** — orbital motion changes the velocities, so a
   shared-`v` fit across epochs is physically meaningless.

**Shared-amplitude approximation.** A *single* `(a1, a2)` is shared across all
pooled windows. For chemically-distinct lines (e.g. an absorption primary and an
emission secondary) the light/line-strength ratio is line-dependent, so one
global amplitude pair is a modelling approximation, not an exact description.

## 8. Uncertainties (CRLB), peak refinement and the noise budget

`crlb_from_map` fits a local 2-D quadratic to `logL_profile` at its peak. For a
Gaussian likelihood `−Hessian(logL)` is the Fisher information and its inverse is
the `(v1, v2)` covariance; because `logL` is the *profile* over the (linear)
amplitudes, this is already the full CRLB (profile = marginal for linear-Gaussian
nuisances). The amplitudes are linear only at fixed velocity, so it is a **local**
CRLB — valid near an isolated, well-resolved peak; it degrades for blended /
near-degenerate peaks (`A12² → S11·S22`, `det → 0`) and returns `nan` off a
positive-definite Hessian. `peak_refined` is the sub-grid stationary point of the
same quadratic (clipped to one grid step).

**Patch size vs peak width — the two-pass refinement.** `logL` follows the
*squared* correlation, so its peak is only ~`σ_ccf/√2` wide (~2.4 km/s for
8 km/s FWHM lines). A quadratic over ±3 nodes of a 1 km/s map picks up the
quartic fall-off: on noiseless 8 km/s lines the refined RVs were off by
0.03–0.1 km/s against an eRV of ~0.005–0.009 km/s (several σ), and by up to
0.06 km/s with twin templates and the truth exactly on a node. The quadratic's
validity is set by the line/template width, not by the S/N, so the fix is a
finer patch, not a ΔlogL cut. `CCF2d` therefore runs two passes:

1. the coarse map (default `dvel` = 1 km/s) only *locates* the peak;
2. the six statistics are re-evaluated **exactly** (`window_stats` on the
   prepared windows) on a fine local grid — ±2 coarse steps around the coarse
   argmax, step `dvel/10` (41 × 41 nodes) — and `crlb_from_map` (±3 fine nodes,
   i.e. ±0.3 `dvel`) is applied to the fine map.

The coarse step must already resolve the peak (`dvel ≲ σ_ccf`) for the coarse
argmax to be trustworthy; a tenth of it keeps the quadratic patch within
~0.3 `σ_ccf`, where the quartic term shifts the curvature by a few per cent and
the peak location negligibly (noiseless refined RVs then agree with the truth
to ≲ 0.001 km/s). The amplitudes and χ² are then evaluated exactly at the
refined `(v1, v2)`, so the reported RVs and amplitudes belong to one point. If
the coarse argmax is on the grid edge no refinement is done (warning, `nan`
errors); if the fine argmax lands on the fine-map edge (a ridge or a tie), a
warning is issued and the coarse-map estimate is reported. Cost: ~0.4 s on a
2-order, 2×2·10⁵-pixel spectrum with a 301 × 301 coarse map (~14 % of the call).

**Oversampled grids.** Resampling a spectrum onto a grid `M = dv_native/dv`
times finer (e.g. ~1 km/s pixels onto the 0.1 km/s CCF grid) adds samples but
no information: ~M analysis pixels share each native pixel's noise. White
Parseval weights then count the Fisher information M times and shrink the CRLB
by `√M` — Monte Carlo scatter / CRLB was ~2.5 (≈ √10 before the curvature bias
above partly compensated). `window_stats(..., oversampling=M)` divides the
weights by M (equivalently `σ → σ√M`), which restores the native-pixel
information for any line resolved by the native sampling; `CCF2d` takes
`M_i = max(1, dv_native,i/dv)` per order from `Spectrum.native_steps()`
(the coarsest sampling the spectrum has had). Monte Carlo scatter / CRLB is then 0.92 / 0.97
for the two components.

**Noise budget and scale.** Each window also returns `nu`, the expected χ² of
its pure-noise residual, `Σ_k w_k E|N_k|²` over the *retained* modes (DC,
Nyquist and anything above `zeta_max` excluded). For white noise on the
analysis grid that is the number of retained modes. Resampled noise is not just
correlated but *smoother*: its variance per analysis pixel is `κσ²` with
`κ = 2/3` for linear interpolation and ≈ 0.87 for the cubic spline of
`Spectrum.InterpolateSpectrum`, and the Tukey taper multiplies it by
`mean(w²)` (≈ 0.94 at α = 0.1). Dividing χ² by the bare independent-pixel
count `n_eff = n/M` would therefore leave χ²_red ≈ 0.8 (≈ 0.63 for linear
interpolation) for honest noise. Instead `noise_psd` — from
`resampled_noise_psd`, i.e. the averaged power spectrum of the resampler's
impulse responses, times `mean(w²)` — gives the noise power the fit actually
sees, and

    χ²_red = χ²_best / (Σ_windows ν − 4)          (four fitted parameters)

is ≈ 1 for honest noise (1.004 in the Monte Carlo above).

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
3 × native. (Probing a single phase had given ν ≈ 0 at 2× — χ²_red undefined —
and χ²_red 1.11–1.34 at 1.5× and 3×.)

Since χ² and the map curvature scale as `1/σ²`, a single common σ does not
move the argmax and rescales the errors by `√χ²_red`. `CCF2d`'s default
`sigma=None` uses exactly that: fit at unit σ, report `sigma = √χ²_red` (the
native per-pixel noise, χ²_red at unit σ) and errors scaled by it. The reported
`chi2_best` / `chi2_red` are at the noise behind the errors, so under
`sigma=None` χ²_red = 1 by construction; `chi2_map` is the same χ² on the
maps' noise scale (unit σ under `sigma=None`, where the logL/χ² maps, the fine
maps, `self.stats` and `self.crlb` all stay at unit σ). An explicit `sigma`
(scalar or per order) is taken at face value — no rescale — and χ²_red is a
consistency check.

## 9. Assumptions and caveats

Exact **only** under these; the first is the one to watch:

- **White native noise `σ` per native pixel, resampled once by a linear
  interpolator.** `oversampling`, `noise_psd` and the native-pixel count
  behind the CRLB model that resampling (in `CCF2d`: one `InterpolateSpectrum`
  call from a uniform native grid, probed with unit impulses). **After
  `FilterSpectrum` / `SpecPreProccess` this calibration does not hold**:
  band-pass filtering, per-order normalisation and the double resampling
  (`SpecPreProccess`'s linear resampling, then `CCF2d`'s log one) remove and
  correlate noise power that ν and the CRLB do not know about. Measured by
  Monte Carlo on preprocessed spectra: TODCOR RV scatter / eRV = 0.72–0.74,
  i.e. errors ~1.4× too large; UNICOR eRV / RV scatter = 1.17 at
  `SpecPreProccess`'s default `FilterHC = 0.15` and 1.38 at `FilterHC = 0.9`.
  χ²_red then drifts from 1 and `sigma=None` absorbs only its mean scale.
  Calibrate the errors by simulation (or a constant star) for preprocessed
  spectra. The taper makes the noise heteroscedastic (`σ_n = w_n σ`); only its
  mean square enters `ν`.
- **White weights.** The Fisher weights ignore the colour of the resampled
  noise. An interpolator that attenuates in-band noise (linear interpolation's
  passband droop) leaves the CRLB mildly conservative; so does an analysis
  grid coarser than native whose nodes fall between native pixels (smoothed,
  correlated noise): Monte Carlo RV scatter / eRV = 0.82–0.91 at 1.5× native
  (log- and linear-λ native grids) and 2× (linear-λ), against 0.97–1.06 at 3×
  and at 2× on a log-λ native grid, whose nodes then all fall on native pixels.
- **χ²_red counts four parameters.** When `nonneg=True` clamps an amplitude to
  zero only three are free; the difference is negligible against `Σν`.
- **Local quadratic CRLB** (§8): valid for an isolated, well-resolved peak
  sampled by the fine map.
- **Flat ℝ² amplitude prior** for the marginal map (§5).
- **Shared velocity grid** for `combine_stats` (§7).
- **Shared single `(a1, a2)`** across pooled lines is an approximation (§7).
- **Non-negative marginal** drops two terms (§5).

## 10. Function map

All of these except `CCF2d` are reachable from the `todcor` namespace; the
"lives in" column says where they are defined.

| Function | Lives in | Role |
| --- | --- | --- |
| `log_lambda_grid`, `resample`, `tukey_window` | `unicor.dsp` | grid + apodisation (auxiliary) |
| `rfft_positive`, `parseval_weights` | `unicor.dsp` | Fourier transform + Parseval weights (§2) |
| `broaden_gaussian_residual`, `rotational_kernel`, `broaden_rotational_residual` | `unicor.dsp` | template broadening |
| `shift_kms` | `unicor.dsp` | the log-λ shift `s = c·ln(1 + v/c)` |
| `shift_template_flux`, `composite_flux` | `todcor.simulate` | forward model / simulate composites |
| `window_stats` | `todcor.estimator` | the six sufficient statistics (+ noise dof `nu`) for one window (§3, §8) |
| `resampled_noise_psd` | `todcor.estimator` | noise spectrum of resampled white native noise → `noise_psd` (§8) |
| `combine_stats` | `todcor.estimator` | sum statistics across windows → joint fit (§7) |
| `profile_from_stats` | `todcor.estimator` | profile/marginalise amplitudes → logL maps (§4–5) |
| `todcor_map` | `todcor.estimator` | single-window convenience wrapper (§4) |
| `crlb_from_map` | `todcor.estimator` | Cramér–Rao velocity errors from curvature (§8) |
| `model_flux` | `todcor.simulate` | reconstruct the best-fit composite for residual plots |
| `CCF2d` | `sparta.TODCOR` | `Spectrum` + two `Template`s → coarse + fine maps, RVs, errors, χ²_red (orders as windows) |

## 11. Reference

Zucker, S. & Mazeh, T. 1994, "Study of Spectroscopic Binaries with TODCOR. I.",
ApJ 420, 806.
