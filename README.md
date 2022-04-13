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

**USuRPER**—**U**nit **S**phere **R**epresentation **PER**iodogram—a phase-distance correlation (PDC) based periodogram, designed for very high-dimensional data like spectra ([ref](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474L..86Z/abstract)).

### Near-future extension plans:

Template independent radial velocity extraction ([ref](https://ui.adsabs.harvard.edu/abs/2006MNRAS.371.1513Z/abstract)). Analysis of composite spectra using a 2-d CCF ([ref](https://ui.adsabs.harvard.edu/abs/1994ApJ...420..806Z/abstract)). 

<br />



## Installation instructions: 

SPARTA is compatible with Python 3.7. The complete requirement list is available in the requirements text file. **Download or clone** SPARTA package from GitHub. Unpack the zip file in a designated folder. 

**Install using pip:** 

```
pip install -U -r requirements.txt
pip install -e [path to setup.py]
```
