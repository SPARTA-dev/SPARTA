"""SPARTA -- SPectroscopic vARiabiliTy Analysis.

Radial velocities and periodicity searches for spectroscopic time series:

``Spectrum``, ``Template``
    (multi-order) spectra and reference spectra (PHOENIX models, broadening).
``CCF1d``
    UNICOR: single-template cross-correlation RVs (Zucker 2003).
``CCF2d``
    TODCOR: two-template RVs of double-lined binaries (Zucker & Mazeh 1994).
``TIRAVEL``
    a template built from the observations themselves (Zucker & Mazeh 2006).
``Observations``
    a spectroscopic time series and the driver of all of the above, epoch by
    epoch; ``TimeSeries`` is its container of times and values.
``PeriodicityDetector``
    GLS, PDC (Zucker 2018) and USuRPER (Binnenfeld et al. 2020) periodograms.
``instruments``
    per-instrument archive loaders and a simulator, by survey key.
``plotting``
    the figure style: data as black steps, models red, one extra series blue,
    maps as black unfilled contours.

The array-level engines are the standalone ``unicor`` and ``todcor``
packages; the PDC math is in ``sparta.pdc`` and the TIRAVEL measure in
``sparta.tiravel_ops``.

Conventions: wavelengths in Angstrom, velocities in km/s, times in days,
periodogram frequencies in 1/day.
"""
from sparta import instruments, plotting
from sparta.ccf1d import CCF1d
from sparta.ccf2d import CCF2d
from sparta.observations import Observations
from sparta.periodogram import PeriodicityDetector
from sparta.spectrum import Spectrum
from sparta.template import Template
from sparta.timeseries import TimeSeries
from sparta.tiravel import TIRAVEL

__version__ = "0.3.0"

__all__ = ["Spectrum", "Template", "Observations", "TimeSeries", "CCF1d", "CCF2d",
           "TIRAVEL", "PeriodicityDetector", "instruments", "plotting"]
