# SPARTA

<p>
<a href="https://github.com/SPARTA-dev/SPARTA">
    <img src="https://img.shields.io/badge/GitHub-SPARTA--dev%2FSPARTA-blue?style=flat"></a>
<a href="https://github.com/SPARTA-dev/SPARTA/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue?style=flat"></a>
<a href="https://ui.adsabs.harvard.edu/abs/2020A%26A...642A.146B/abstract">
    <img src="https://img.shields.io/badge/read-USuRPER_paper-yellowgreen?style=flat"></a>
<a href="https://ui.adsabs.harvard.edu/abs/2022A%26A...659A.189B/abstract">
    <img src="https://img.shields.io/badge/read-partial_PDC_paper-yellowgreen?style=flat"></a>
<a href="https://github.com/SPARTA-dev/SPARTA/tree/master/examples">
    <img src="https://img.shields.io/badge/tutorials-notebooks-green?style=flat"></a>
</p>



<br />

**SPARTA** — **SP**ectroscopic v**AR**iabili**T**y **A**nalysis — is a collection of tools designed to analyze periodically-variable spectroscopic observations. Aimed for common astronomical uses, *SPARTA* facilitates analysis of single- and double-lined binaries, high-precision radial velocity extraction, and periodicity searches in complex, high dimensional data. The package is currently under active development. Comments are very welcome.

<br />

------

<br />

## Contents:

**UNICOR**—an engine for the analysis of spectra, using 1-d CCF. Includes maximum-likelihood analysis of multi-order spectra,  and detection of systematic shifts ([ref](https://ui.adsabs.harvard.edu/abs/2003MNRAS.342.1291Z/abstract), [ref](https://ui.adsabs.harvard.edu/abs/2017PASP..129f5002E/abstract)).

**TODCOR**—an engine for double-lined (SB2) spectra, using a 2-d CCF. Fits the radial velocities *and* the relative amplitudes of both components simultaneously, in the Fourier domain, and pools spectral orders by adding sufficient statistics ([ref](https://ui.adsabs.harvard.edu/abs/1994ApJ...420..806Z/abstract)). See [`examples/todcor/`](examples/todcor/).

**USuRPER**—**U**nit **S**phere **R**epresentation **PER**iodogram—a phase-distance correlation (PDC) based periodogram, designed for very high-dimensional data like spectra ([ref](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474L..86Z/abstract)).

**TIRAVEL**—**T**emplate-**I**ndependent **RA**dial **VEL**ocity: builds a template out of the observations themselves. Given a set of trial velocities it aligns the epochs, measures how well they co-add (the alignment measure ρ), and returns the maximum-likelihood co-added template for use by UNICOR or TODCOR ([ref](https://ui.adsabs.harvard.edu/abs/2006MNRAS.371.1513Z/abstract)). Currently the fixed-velocity step: ρ is evaluated at the supplied velocities, and is the objective a velocity optimizer (not yet implemented) would maximise.

### Near-future extension plans:

TIRAVEL velocity optimization (maximise ρ over the velocities). Further 2-d CCF work (template-pair search).

<br />



## Repository layout

SPARTA is a monorepo of three installable packages. The two correlation engines
are standalone — plain numpy arrays in, numbers out, only numpy and scipy
required — and SPARTA builds its `Spectrum` / `Template` / `Observations` layer on
top of them.

| Package (pip name) | Import | Path | Depends on |
| --- | --- | --- | --- |
| UNICOR engine (`sparta-unicor`) | `unicor` | [`packages/unicor`](packages/unicor) | numpy, scipy |
| TODCOR engine (`sparta-todcor`) | `todcor` | [`packages/todcor`](packages/todcor) | numpy, scipy, `unicor` |
| SPARTA (`SPARTA`) | `sparta` | [`src/sparta`](src/sparta) | both engines, astropy, specutils, matplotlib, tqdm |

Inside `sparta` the layout is flat, one module per concern, and the classes are
importable from the top level (`from sparta import Observations, Template`):

| Module | Contents |
| --- | --- |
| `sparta.spectrum`, `sparta.template` | `Spectrum`, `Template` (PHOENIX cache in `template_files/`) |
| `sparta.ccf1d`, `sparta.ccf2d` | `CCF1d` (UNICOR), `CCF2d` (TODCOR) |
| `sparta.tiravel`, `sparta.tiravel_ops` | `TIRAVEL` and its array-level measure |
| `sparta.observations`, `sparta.timeseries` | `Observations` (the series-level driver), `TimeSeries` (container) |
| `sparta.periodogram`, `sparta.pdc` | `PeriodicityDetector` (GLS, PDC, USuRPER) and the PDC array math |
| `sparta.instruments` | per-instrument loaders and the simulator |

The engines carry a `sparta-` prefix on PyPI because `unicor` there is an
unrelated project; likewise `pip install sparta` installs an unrelated RDF
library. Install from this repository as below.

## Installation

Requires Python ≥ 3.10. Clone the repository, then, from its root:

```bash
# Everything, editable, for development (one pip call so the engines resolve locally):
pip install -r requirements.txt

# Or a regular install of the full package:
pip install ./packages/unicor ./packages/todcor .
```

Just one engine:

```bash
pip install ./packages/unicor                      # UNICOR only
pip install ./packages/unicor ./packages/todcor    # TODCOR (needs unicor)
```

Straight from GitHub, without cloning:

```bash
R=git+https://github.com/SPARTA-dev/SPARTA
pip install "sparta-unicor @ $R#subdirectory=packages/unicor"                 # UNICOR
pip install "sparta-unicor @ $R#subdirectory=packages/unicor" \
            "sparta-todcor @ $R#subdirectory=packages/todcor"                 # + TODCOR
pip install "sparta-unicor @ $R#subdirectory=packages/unicor" \
            "sparta-todcor @ $R#subdirectory=packages/todcor" "SPARTA @ $R"   # everything
```

The engines are not on PyPI yet, so list them in the same `pip` call as anything
that depends on them.

Run the tests with `pytest` from the repository root (all three packages), or
`pytest packages/todcor` for one engine.
