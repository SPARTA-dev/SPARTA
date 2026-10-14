# Periodograms: GLS, PDC, USuRPER and the partial periodograms

`sparta.PeriodicityDetector` searches one time series on one frequency grid.
It holds the series and the results; the distance-correlation math is a set of
pure array functions in `sparta.pdc`. Times are in days and frequencies in
1/day. The tutorials are
[notebook 05](../examples/05_finding_periods.ipynb) (all the periodograms),
[07](../examples/07_usurper_concept.ipynb) (USuRPER) and
[08](../examples/08_partial_pdc_concept.ipynb) (the partial periodograms).

## Setting up

```python
obs.init_periodicity_detector(freq_range=(0.01, 1.0), periodogram_grid_resolution=2000)
pd = obs.periodicity_detector
```

- The grid is `np.arange(f_min, f_max, 1/periodogram_grid_resolution)`
  (1/day; `f_max` excluded). `f_min` must be > 0 (the phase distance takes
  Δt mod 1/f) and `f_max` > `f_min`; otherwise `ValueError`.
  `Observations.init_periodicity_detector` defaults to `(1/1000, 1)` and 1000
  points per 1/day. A peak is about 1/T wide (T the time span), so the step
  should be several times finer than 1/T.
- **What is searched.** For a scalar series, its values. For a spectrum
  series, GLS and PDC read the RVs in `time_series.calculated_vrad_list`
  (km/s), which `Observations.calc_rv_against_template` fills (with
  `combine_ccfs=True`, the default) together with their errors in
  `time_series.errors`. A TODCOR run leaves them alone; assign one component
  explicitly (`obs.time_series.calculated_vrad_list = obs.vels1`).
  USuRPER and the partial periodograms read the spectra themselves.
- Only time differences enter; `Observations` makes the times relative to
  `first_time` ([conventions.md](conventions.md#times)).

## Results

Every periodogram is stored under a key in `pd.results_power` (dimensionless
power) and `pd.results_frequency` (a copy of the grid, 1/day):

| Periodogram | Call | Keys | Input |
| --- | --- | --- | --- |
| GLS | `calc_gls()` (also returns `(frequency, power)`) | `'GLS'` | values, or `calculated_vrad_list` |
| PDC | `calc_pdc(calc_biased_flag=False, calc_unbiased_flag=True)` | `'PDC_unbiased'`, `'PDC_biased'` | values, or `calculated_vrad_list` |
| Error-aware PDC | `calc_pdc(errors=True)` | `'PDC_errors_unbiased'`, `'PDC_errors_biased'` | as PDC, plus `time_series.errors` |
| USuRPER | `calc_usurper(calc_biased_flag=False, calc_unbiased_flag=True)` | `'USURPER_unbiased'`, `'USURPER_biased'` | single-order spectra |
| Partial (shift) | `calc_partial_periodogram("shift")` | `'partial_shift'` | single-order spectra and `calculated_vrad_list` |
| Partial (shape) | `calc_partial_periodogram("shape")` | `'partial_shape'` | single-order spectra and `calculated_vrad_list` |

By default only the unbiased PDC and USuRPER are computed. `plot_periodograms`
draws every stored periodogram; nothing is written to the time series.

## GLS

The generalised Lomb–Scargle periodogram of `astropy.timeseries.LombScargle`
with its defaults (floating mean, `'standard'` normalisation), evaluated on
the grid. The errors are not passed to it: the fit is unweighted.

## PDC: phase distance correlation (Zucker 2018)

PDC asks whether epochs close in *phase* are also close in *value*, with no
assumed light-curve shape. It is a distance correlation between two n × n
matrices of pairwise distances:

- the values' distances, a_ij = |y_i − y_j| (`sparta.pdc.scalar_distance_matrix`);
- the phase distances at trial frequency f = 1/P,
  b_ij = φ_ij (P − φ_ij) with φ_ij = (t_i − t_j) mod P
  (`phase_distance_matrix`; in days², a scale that cancels in the
  correlation).

Both are centred the same way, and the power at f is their cosine,

$$\mathrm{PDC}(f) = \frac{\sum_{ij} A_{ij}B_{ij}}{\sqrt{\sum_{ij}A_{ij}^2\,\sum_{ij}B_{ij}^2}}$$

(`pdc_power`):

- **Biased** (`'PDC_biased'`): double centring, A = a − row means − column
  means + grand mean (`double_centering`). This is the squared sample distance
  correlation of Székely, Rizzo & Bakirov (2007), which is positive even for
  independent data.
- **Unbiased** (`'PDC_unbiased'`, the default): U-centring (Székely & Rizzo
  2014; `unbiased_u_centering`): row and column sums divided by n − 2, the
  grand sum by (n − 1)(n − 2), and a zero diagonal; the U-centred inner product
  carries 1/(n(n − 3)) (`inner_prod`). The result is the bias-corrected
  statistic, ≈ 0 for independent data and possibly negative.
  These factors are singular or negative for small samples, so **n ≤ 3 raises
  `ValueError`**.

A PDC-type periodogram also responds, weakly, at twice the true period: in
notebook 08 the noiseless RVs of a 7.3-day planet give PDC 0.66 at P and 0.24
at 2P. Check P against 2P before believing a long period.

## Error-aware PDC (Binnenfeld, Shahaf & Zucker 2024)

`calc_pdc(errors=True)` replaces |y_i − y_j| by the distance of two Gaussian
measurements, Eq. 3 of Binnenfeld, Shahaf & Zucker (2024, A&A 686, A192)
**exactly as published** (`error_aware_distance_matrix`):

$$d_{ij} = \sqrt{s\left(e^{-x^2} + x\,\mathrm{erf}(x) - \frac{e_i + e_j}{\sqrt2\,s}\right)},\qquad s = \sqrt{e_i^2 + e_j^2},\quad x = \frac{y_i - y_j}{\sqrt2\,s},$$

with e_i the 1σ errors (`time_series.errors`, in the values' unit; one per
value, else `ValueError`). A negative argument of the root is set to 0 and the
diagonal is 0; d is in √(unit of y). There is deliberately **no √π** on the
x·erf(x) term: an energy-distance derivation suggests one, but the authors
confirmed that the published form is the intended one, and a golden test pins
it. In the limit of vanishing errors d_ij → (|y_i − y_j|/√2)^{1/2}, a monotone
function of |y_i − y_j| but not |y_i − y_j| itself, so the error-aware PDC does
not reduce exactly to the plain PDC. In notebook 05, with exponentially
distributed error bars, it found the true period in 26 of 30 simulated data
sets against 23 for the plain PDC.

## USuRPER (Binnenfeld, Shahaf & Zucker 2020)

The *Unit Sphere Representation PERiodogram* runs the PDC on the spectra
themselves (`calc_usurper`, `spectral_distance_matrix`):

1. every spectrum is resampled (linear interpolation) onto the grid common to
   all of them: from the largest first wavelength to the smallest last one,
   evenly spaced at the finest median pixel step (`get_wv_grid`);
2. each is mean-subtracted and L2-normalised, a point x_i on a unit sphere;
3. the distance is the chord d_ij = |x_i − x_j| = √(2(1 − x_i·x_j)) ∈ [0, 2];
4. the PDC scan above, unbiased by default.

No template and no RV enter, and a change of flux scale or offset moves
nothing on the sphere, so USuRPER responds to any periodic change of the
spectrum: Doppler shifts and line-shape changes alike
([notebook 07](../examples/07_usurper_concept.ipynb): a moving star, a
temperature change with no motion, an unresolved twin binary found at half its
period). It needs **single-order** spectra (a multi-order spectrum raises;
select one order with `Observations.select_orders`) that **overlap** in
wavelength (no common range raises), and it says *that* the spectra vary, not
how.

## Partial periodograms: shift and shape (Binnenfeld et al. 2022)

`calc_partial_periodogram` separates Doppler shifts from shape changes with a
partial distance correlation (Székely & Rizzo 2014). It builds two distance
matrices from the RVs v_i (`time_series.calculated_vrad_list`, one per epoch,
else `ValueError`) and the single-order spectra:

- **shift distance**: |v_i − v_j|;
- **shape distance**: the USuRPER chord distance between the spectra after
  each is shifted back to rest by its own RV (on copies; the caller's spectra
  are not modified), so a pure Doppler shift leaves no shape distance.

| Call | Searched matrix a | Control matrix c | Key |
| --- | --- | --- | --- |
| `calc_partial_periodogram("shift")` | shift distance | shape distance | `'partial_shift'` |
| `calc_partial_periodogram("shape")` | shape distance | shift distance | `'partial_shape'` |

With Ã, C̃ the U-centred a and c, and (X·Y) = Σ X_ij Y_ij / (n(n − 3)),
`partial_centering` projects the control out of the searched matrix,
P(Ã) = Ã − (Ã·C̃)/(C̃·C̃) C̃, and `pdc_power(..., control=C̃)` does the same to
every U-centred phase matrix B̃ before taking the cosine:

$$\mathrm{pdCor}(f) = \frac{P(\tilde A)\cdot P(\tilde B)}{|P(\tilde A)|\,|P(\tilde B)|}.$$

Both matrices are residualised on the control, at every trial frequency. The
partial statistic is defined for the unbiased (U-centred) form only (a
control with the biased statistic raises). What it can and cannot tell:

- The projection removes only the part of the dependence that is *linear in
  the U-centred control*. A pdCor of 0 is not conditional independence
  (Székely & Rizzo 2014), and a non-zero pdCor does not by itself prove a
  dependence beyond the control: part of an RV leak that follows the shape
  can survive. In notebook 08's star with activity and no planet, the plain PDC
  of the RVs has 0.61 at the activity period and the shift periodogram still
  0.22 (local p = 2·10⁻³).
- When the RVs are nothing but a leak of the shape change, the control also
  removes part of the shape signal (notebook 08: 0.37 against 0.64 with a
  planet present).
- Signals at the same period cannot be separated, and a shape change outside
  the chosen order cannot be controlled for.

## p-values

`calc_pdc_pval(values)` converts unbiased PDC power to a p-value with the
χ² approximation of the distance-correlation test (Shen et al. 2019,
arXiv:1912.12150):

$$p = \chi^2_1.\mathrm{sf}(n \cdot \mathrm{power} + 1),$$

n the number of epochs; `inverse_sf=True` returns the power level of each
p-value instead. This is a **local, single-frequency** p-value, not a
false-alarm probability over the grid. Searching many frequencies needs a
look-elsewhere correction; a rough factor is the number of independent
frequencies, T·(f_max − f_min) (notebook 05).

The approximation is derived for the distance correlation. SPARTA applies the
same formula to USuRPER and to the partial periodograms (`plot_periodograms`
draws the levels on those panels too), but it is **not established for the
partial statistic**. Notebook 08 checks it by permuting the epochs' times at
one frequency: over 4000 permutations the p = 0.05 level was exceeded in 4.5%
(shift) and 4.7% (shape) of them, and the p = 0.01 level in 0.43% and 0.60%.
Treat partial p-values as a guide and confirm a candidate by a permutation
test or by injection and recovery.

## References

- PDC: Zucker, S. 2018, MNRAS, 474, L86
- USuRPER: Binnenfeld, A., Shahaf, S. & Zucker, S. 2020, A&A, 642, A146
- Partial PDC: Binnenfeld, A., et al. 2022, A&A, 659, A189
- Error-aware PDC: Binnenfeld, A., Shahaf, S. & Zucker, S. 2024, A&A, 686, A192
- Distance correlation: Székely, G. J., Rizzo, M. L. & Bakirov, N. K. 2007,
  Ann. Stat., 35, 2769; partial distance correlation and U-centring:
  Székely, G. J. & Rizzo, M. L. 2014, Ann. Stat., 42, 2382
- The χ² test of distance correlation: Shen, C., Panda, S. & Vogelstein, J. T.
  2019, arXiv:1912.12150
