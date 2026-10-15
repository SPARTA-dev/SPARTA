# SPARTA

<p>
<a href="https://github.com/SPARTA-dev/SPARTA">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-SPARTA--dev%2FSPARTA-blue?style=flat"></a>
<a href="https://github.com/SPARTA-dev/SPARTA/blob/master/LICENSE">
    <img alt="MIT license" src="https://img.shields.io/badge/license-MIT-blue?style=flat"></a>
<a href="https://ui.adsabs.harvard.edu/abs/2020A%26A...642A.146B/abstract">
    <img alt="USuRPER paper" src="https://img.shields.io/badge/read-USuRPER_paper-yellowgreen?style=flat"></a>
<a href="https://ui.adsabs.harvard.edu/abs/2022A%26A...659A.189B/abstract">
    <img alt="partial PDC paper" src="https://img.shields.io/badge/read-partial_PDC_paper-yellowgreen?style=flat"></a>
<a href="https://github.com/SPARTA-dev/SPARTA/tree/master/examples">
    <img alt="tutorial notebooks" src="https://img.shields.io/badge/tutorials-notebooks-green?style=flat"></a>
</p>

**SPARTA** — **SP**ectroscopic v**AR**iabili**T**y **A**nalysis — measures radial
velocities of single- and double-lined spectra and searches spectroscopic time
series for periodicity. It reads archive spectra from about a dozen
instruments (or simulates them), measures RVs with cross-correlation engines whose error bars
are meant to be taken at face value, and runs periodograms on the RVs or on the
spectra themselves. It is under active development; comments are welcome.

## What is in it

- **UNICOR** — single-lined RVs by one-dimensional cross-correlation on a
  log-λ grid. Orders are combined by maximum likelihood, and the RV
  uncertainty comes from the curvature of the combined CCF (Zucker 2003;
  Engel, Shahaf & Mazeh 2017).
- **TODCOR** — double-lined (SB2) RVs by two-dimensional correlation: the
  velocities *and* the light ratio of both components, fitted together in the
  Fourier domain, with orders pooled into one fit and Cramér–Rao errors
  (Zucker & Mazeh 1994).
- **TIRAVEL** — a template built from the observations themselves: given
  trial velocities, it aligns the epochs, measures how well they co-add (the
  alignment measure ρ), and returns the maximum-likelihood co-added template
  for UNICOR or TODCOR (Zucker & Mazeh 2006). Only this fixed-velocity step is
  implemented; the optimisation of ρ over the velocities is not.
- **Periodograms** — the generalised Lomb–Scargle (GLS, via astropy) and
  the phase-distance-correlation family:
  - **PDC** — a distance-correlation periodogram of a scalar series such as
    RVs, free of any assumed light-curve shape (Zucker 2018);
  - **error-aware PDC** — PDC with each point's uncertainty in the distance
    (Binnenfeld, Shahaf & Zucker 2024);
  - **USuRPER** — the *Unit Sphere Representation PERiodogram*: PDC on the
    spectra themselves, as points on a unit sphere, so it needs no template
    and responds to shape changes as well as shifts (Binnenfeld, Shahaf &
    Zucker 2020);
  - **partial PDC** — the *shift* periodogram (Doppler shifts with the
    line-shape changes partialled out) and the *shape* periodogram (the
    reverse), which tell orbital motion from stellar activity (Binnenfeld et
    al. 2022).
- **Instruments** — one loader per archive product, with the wavelength frame,
  time system and native sampling of each, and a simulator that produces
  synthetic SB1/SB2 data through the same path. See
  [`docs/instruments.md`](https://github.com/SPARTA-dev/SPARTA/blob/master/docs/instruments.md).

## Installation

SPARTA is installed from a clone of this repository (it is not on PyPI). It
needs Python 3.10 or later; it is tested on 3.10, 3.12 and 3.13.

```bash
git clone https://github.com/SPARTA-dev/SPARTA.git
cd SPARTA
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

With conda, replace the `venv` and `activate` lines by
`conda create -n sparta python=3.12` and `conda activate sparta`.

This installs SPARTA and its two correlation engines, UNICOR and TODCOR, in
editable mode (a `git pull` updates them), together with pytest. To check the
installation, run `pytest` in the `SPARTA` folder (a minute or two). To open
the tutorial notebooks, also `pip install jupyterlab`.

To install only an engine, from the `SPARTA` folder:

```bash
pip install ./packages/unicor                      # UNICOR
pip install ./packages/unicor ./packages/todcor    # TODCOR (needs UNICOR)
```

Do not run `pip install sparta` (that is an unrelated package on PyPI) or
`pip install .` on its own: the engines are not on PyPI yet, so pip must get
them from the clone in the same command, as above.

## Quickstart

Simulate a sun-like star on a 4.3-day orbit as HARPS would record it, measure
its RVs with UNICOR, check them against the injected truth, and recover the
period. Everything is offline — the PHOENIX model is bundled with SPARTA — and
it runs in about a second.

```python
import numpy as np
from sparta import Observations, Template
from sparta.instruments import Simulation

# Twenty epochs of a sun-like star on a 4.3-day circular orbit, K = 25 km/s,
# observed as HARPS would record it (R = 120,000, air wavelengths, S/N 50).
times = np.sort(np.random.default_rng(0).uniform(0, 40, 20))      # days
rv = 12.0 + 25.0 * np.sin(2 * np.pi * times / 4.3)                  # km/s
sim = Simulation(like="HARPS", components=[dict(teff=5800, logg=3.5, feh=0.5)],
                 times=times, rv_kms=rv, snr=50, seed=1)
obs = Observations(survey="SIMULATED", simulation=sim)

# The same PHOENIX model as the template: bundled with SPARTA, so nothing is
# downloaded. HARPS is in air, so convert the (vacuum) model, and match the
# instrumental resolution.
template = Template.from_phoenix(5800, 3.5, 0.5, wave_range=(4950, 5130), air=True)
template.broaden_instrumental(120000)

obs.calc_rv_against_template(template)        # UNICOR, every epoch
residual = obs.vels - obs.truth.rv_kms[:, 0]  # km/s
print(f"RV - truth: rms {np.std(residual) * 1e3:.1f} m/s, "
      f"median eRV {np.median(obs.evels) * 1e3:.1f} m/s")
print(f"chi2 of the residuals: {np.sum((residual / obs.evels) ** 2):.1f} for {len(residual)} epochs")

# The measured RVs are now the time series: search them for the period.
obs.init_periodicity_detector(freq_range=(1 / 50, 1), periodogram_grid_resolution=2000)
pd = obs.periodicity_detector
freq, power = pd.calc_gls()                   # 1/day
pd.calc_pdc()
best_pdc = pd.results_frequency["PDC_unbiased"][np.argmax(pd.results_power["PDC_unbiased"])]
print(f"best period: GLS {1 / freq[np.argmax(power)]:.3f} d, PDC {1 / best_pdc:.3f} d")
```

```text
RV - truth: rms 2.4 m/s, median eRV 2.8 m/s
chi2 of the residuals: 16.2 for 20 epochs
best period: GLS 4.301 d, PDC 4.283 d
```

The template is the very model that was simulated, so the residuals are pure
noise, and their scatter matches the quoted errors (χ² ≈ number of epochs).
With real data, replace the simulation by
`Observations(survey="HARPS", target_visits_lib="path/to/ADP/files")`; the
survey keys are listed in
[`docs/instruments.md`](https://github.com/SPARTA-dev/SPARTA/blob/master/docs/instruments.md).

## Learn

The [`examples/`](https://github.com/SPARTA-dev/SPARTA/blob/master/examples/README.md)
notebooks come in two kinds. The **how-tos** (`examples/howto/`) carry on
from this quickstart: real NRES and HARPS data, a double-lined binary,
TIRAVEL, the periodograms and simulated data, each a short recipe. The
**tutorials** (`examples/tutorials/`) build every method from its atoms and
show why it works: spectra and the log-λ grid, UNICOR, TODCOR, TIRAVEL and the
periodograms.

## Conventions

- **Units.** Velocities in km/s, wavelengths in Å, times in days, periodogram
  frequencies in 1/day. Plain numbers are read in these units; astropy
  Quantities are converted at the API boundary. `Observations` makes the times
  relative to the integer day of the first visit (`first_time`).
- **Velocity convention.** RVs are `'cz'` by default:
  λ_obs/λ_rest = 1 + v/c, the IAU radial-velocity measure and the velocity of
  IRAF fxcor. `velocity_convention='relativistic'` uses
  √((1 + v/c)/(1 − v/c)) instead. For the same spectrum the `'cz'` RV is larger
  by ≈ v²/2c (37.5 m/s at 150 km/s), so use one convention throughout — in the
  RV drivers, `Template.doppler` and `Simulation`.
- **Barycentric correction** is always composed as cz, λ·(1 + bcv/c), whatever
  convention the RVs are read in: that is how pipeline BERVs are defined.
  `Observations` applies it on ingest.
- **Log-λ grid.** Correlation needs a velocity-even grid, uniform in ln λ with
  step ln(1 + dv/c). The engines resample onto one when needed
  (`Spectrum.interpolate(method='log')`) and refuse a grid that only claims to
  be one.
- **Error bars count native pixels.** Resampling onto a finer grid adds
  samples, not information. Every spectrum records its coarsest sampling
  (`Spectrum.native_steps()`; archive-resampled products are seeded with the
  detector step), and UNICOR and TODCOR count `n_eff` independent pixels from
  it, so oversampling does not shrink the errors.
- **Errors are not calibrated after filtering.** The ML and Cramér–Rao errors
  assume white noise on the native pixels. `preprocess` (band-pass filter,
  normalisation, a linear-λ resampling) correlates the noise; in Monte Carlo
  the errors are then 17–38 % (UNICOR) or ~1.4× (TODCOR) too large. Calibrate
  them by simulation, or by the scatter of a constant star.
- **Air or vacuum** depends on the product (listed per product in
  [`docs/instruments.md`](https://github.com/SPARTA-dev/SPARTA/blob/master/docs/instruments.md)).
  PHOENIX models are in vacuum; `Template.from_phoenix(..., air=True)` converts
  them (Morton 2000). Match the template to the data: a frame mismatch is a
  ~80 km/s shift at 5000 Å. `Observations` refuses a directory that mixes
  frames or time systems.
- **Downloads are opt-in.** PHOENIX models come from the local cache; fetching
  another needs `allow_download=True`.
- **Bisector span.** The BIS (`CCF1d.calc_bis`, `Observations.calc_bis`) is
  known to be incorrect and is under review; do not use it yet.

## Documentation

The pages in
[`docs/`](https://github.com/SPARTA-dev/SPARTA/blob/master/docs/index.md)
cover the conventions in full, UNICOR and TODCOR (outputs, likelihoods,
errors), the periodograms, the instruments, the API reference, the known
limitations and a proposed roadmap.
[`docs/open_api.md`](https://github.com/SPARTA-dev/SPARTA/blob/master/docs/open_api.md)
lists the atoms (plain arrays in, arrays out) to build your own analysis from.

## Module map

SPARTA is a monorepo of three installable packages. The two correlation engines
are standalone — plain numpy arrays in, numbers out — and SPARTA builds its
object layer on top of them.

| Package (pip name) | Import | Depends on |
| --- | --- | --- |
| [UNICOR engine](https://github.com/SPARTA-dev/SPARTA/blob/master/packages/unicor/README.md) (`sparta-unicor`) | `unicor` | numpy, scipy |
| [TODCOR engine](https://github.com/SPARTA-dev/SPARTA/blob/master/packages/todcor/README.md) (`sparta-todcor`) | `todcor` | numpy, `unicor` |
| SPARTA (`SPARTA`) | `sparta` | both engines, astropy, specutils, matplotlib (optional: tqdm for progress bars, `pip install tqdm`) |

The `sparta` classes are importable from the top level
(`from sparta import Observations, Template`):

| Module | Contents |
| --- | --- |
| `sparta.spectrum` | `Spectrum`: a multi-order spectrum, its preprocessing and log-λ resampling, and conversion to and from specutils |
| `sparta.preprocess` | the per-order preprocessing steps of `Spectrum` (trim, resample, cosmics, band-pass), on plain arrays |
| `sparta.template` | `Template`: PHOENIX models, arrays or line masks as templates; broadening and cutting to an observation's orders |
| `sparta.ccf1d` | `CCF1d`: UNICOR on a `Spectrum` and a `Template` |
| `sparta.ccf2d` | `CCF2d`: TODCOR on a `Spectrum` and two `Template`s |
| `sparta.tiravel` | `TIRAVEL`: the template built from the observations |
| `sparta.tiravel_ops` | the TIRAVEL alignment measure and co-addition, on plain arrays |
| `sparta.observations` | `Observations`: a time series and the driver of all the engines, epoch by epoch |
| `sparta.timeseries` | `TimeSeries`: the container of times and values the periodograms read |
| `sparta.periodogram` | `PeriodicityDetector`: GLS, PDC, error-aware PDC, USuRPER and the partial periodograms |
| `sparta.pdc` | the PDC math on plain arrays: distance matrices, centering, the frequency scan |
| `sparta.instruments` | per-product loaders by survey key, and `Simulation` ([`docs/instruments.md`](https://github.com/SPARTA-dev/SPARTA/blob/master/docs/instruments.md)) |

The TODCOR derivation and its assumptions are written up in
[`packages/todcor/docs/todcor_method.md`](https://github.com/SPARTA-dev/SPARTA/blob/master/packages/todcor/docs/todcor_method.md).
API changes between versions, with complete old → new rename tables, are in
[`CHANGELOG.md`](https://github.com/SPARTA-dev/SPARTA/blob/master/CHANGELOG.md);
how to run the tests and add an instrument is in
[`CONTRIBUTING.md`](https://github.com/SPARTA-dev/SPARTA/blob/master/CONTRIBUTING.md).

## Citing

If SPARTA helps your research, please cite the papers behind the methods you
use:

- UNICOR: Zucker, S. 2003, [MNRAS, 342, 1291](https://ui.adsabs.harvard.edu/abs/2003MNRAS.342.1291Z/abstract);
  Engel, M., Shahaf, S. & Mazeh, T. 2017, [PASP, 129, 065002](https://ui.adsabs.harvard.edu/abs/2017PASP..129f5002E/abstract)
- TODCOR: Zucker, S. & Mazeh, T. 1994, [ApJ, 420, 806](https://ui.adsabs.harvard.edu/abs/1994ApJ...420..806Z/abstract)
- TIRAVEL: Zucker, S. & Mazeh, T. 2006, [MNRAS, 371, 1513](https://ui.adsabs.harvard.edu/abs/2006MNRAS.371.1513Z/abstract)
- PDC: Zucker, S. 2018, [MNRAS, 474, L86](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474L..86Z/abstract)
- USuRPER: Binnenfeld, A., Shahaf, S. & Zucker, S. 2020, [A&A, 642, A146](https://ui.adsabs.harvard.edu/abs/2020A%26A...642A.146B/abstract)
- Partial PDC: Binnenfeld, A., et al. 2022, [A&A, 659, A189](https://ui.adsabs.harvard.edu/abs/2022A%26A...659A.189B/abstract)
- Error-aware PDC: Binnenfeld, A., Shahaf, S. & Zucker, S. 2024, [A&A, 686, A192](https://doi.org/10.1051/0004-6361/202347764)

## License

MIT; see [`LICENSE`](https://github.com/SPARTA-dev/SPARTA/blob/master/LICENSE).
