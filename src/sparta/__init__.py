"""SPARTA -- SPectroscopic vARiabiliTy Analysis.

Radial velocities and periodicity searches for spectroscopic time series.

The atoms
---------
Every step is a function on plain arrays that you can call, print and plot
on its own, and all of them import from ``sparta`` itself:

``sparta.preprocess``, ``phoenix_spectrum``
    per-order preprocessing (trim, resample to a ln-lambda grid, cosmics,
    band-pass, the resampling noise model) and the PHOENIX model reader.
``sparta.tiravel_ops``
    TIRAVEL: align epochs, the alignment measure rho, the co-added template
    (Zucker & Mazeh 2006).
``sparta.pdc``
    PDC and USuRPER: distance matrices, centring, the periodogram and its
    p-values (Zucker 2018; Binnenfeld et al. 2020).

The RV engines are the standalone ``unicor`` (single-template CCF, Zucker
2003) and ``todcor`` (two-template fit, Zucker & Mazeh 1994) packages;
import their atoms from them. docs/open_api.md lists every atom with its
units and assumptions.

The objects
-----------
Conveniences assembled from the atoms, for the common recipes:

``Spectrum``, ``Template``
    (multi-order) spectra and reference spectra (PHOENIX models, broadening).
``CCF1d``, ``CCF2d``, ``TIRAVEL``
    UNICOR, TODCOR and TIRAVEL on those objects.
``Observations``
    a spectroscopic time series and the driver of all of the above, epoch by
    epoch; ``TimeSeries`` is its container of times and values.
``PeriodicityDetector``
    GLS, PDC and USuRPER periodograms.
``instruments``
    per-instrument archive loaders and a simulator, by survey key.
``plotting``
    the figure style: data as black steps, models red, one extra series blue,
    maps as grey-scale images with the peak black.

Import cost
-----------
Nothing is imported by ``import sparta`` itself: every name above is resolved
on first access (PEP 562). matplotlib arrives with the first plot, specutils
with the first spline resampling or ``Spectrum`` <-> specutils conversion.

Conventions: wavelengths in Angstrom, velocities in km/s, times in days,
periodogram frequencies in 1/day.
"""
import importlib
import importlib.util

__version__ = "0.3.0"

# public name -> the module that defines it. A name equal to its module's last
# component is that module itself (``instruments``, ``plotting``).
_LAZY_EXPORTS = {
    # The objects.
    "Spectrum": "sparta.spectrum",
    "Template": "sparta.template",
    "Observations": "sparta.observations",
    "TimeSeries": "sparta.timeseries",
    "CCF1d": "sparta.ccf1d",
    "CCF2d": "sparta.ccf2d",
    "TIRAVEL": "sparta.tiravel",
    "PeriodicityDetector": "sparta.periodogram",
    "instruments": "sparta.instruments",
    "plotting": "sparta.plotting",

    # Preprocessing atoms: one order's wavelength and flux arrays in, arrays out.
    "resample_flux": "sparta.preprocess",
    "trim_margins": "sparta.preprocess",
    "resample_to_linear_grid": "sparta.preprocess",
    "resample_to_log_grid": "sparta.preprocess",
    "coarsest_native_step": "sparta.preprocess",
    "remove_cosmics": "sparta.preprocess",
    "bandpass": "sparta.preprocess",
    "resampling_noise_psd": "sparta.preprocess",
    "native_phases": "sparta.preprocess",
    "phoenix_spectrum": "sparta.template",

    # TIRAVEL atoms.
    "rest_frame_wavelengths": "sparta.tiravel_ops",
    "common_grid": "sparta.tiravel_ops",
    "align_spectra": "sparta.tiravel_ops",
    "normalize_rows": "sparta.tiravel_ops",
    "correlation_matrix": "sparta.tiravel_ops",
    "alignment": "sparta.tiravel_ops",
    "coadd": "sparta.tiravel_ops",
    "TiravelResult": "sparta.tiravel_ops",
    "build_tiravel": "sparta.tiravel_ops",

    # PDC / USuRPER atoms.
    "scalar_distance_matrix": "sparta.pdc",
    "error_aware_distance_matrix": "sparta.pdc",
    "spectral_distance_matrix": "sparta.pdc",
    "phase_distance_matrix": "sparta.pdc",
    "double_centering": "sparta.pdc",
    "unbiased_u_centering": "sparta.pdc",
    "inner_prod": "sparta.pdc",
    "norm_z": "sparta.pdc",
    "partial_centering": "sparta.pdc",
    "pdc_power": "sparta.pdc",
    "pdc_pvalue": "sparta.pdc",
    "pdc_power_level": "sparta.pdc",
    "get_wv_grid": "sparta.pdc",
    "resample_spec_to_common_grid": "sparta.pdc",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    """Resolve a public name on first access (PEP 562) and cache it.

    Keeps ``import sparta`` cheap: no submodule, and so neither matplotlib
    nor specutils, is imported until a name that needs it is asked for.
    """
    path = _LAZY_EXPORTS.get(name)
    if path is None and not name.startswith("_") and importlib.util.find_spec(f"{__name__}.{name}"):
        # A submodule (``sparta.pdc``, ``sparta.spectrum``, ...): reachable after a
        # bare ``import sparta``, as when that import loaded everything.
        path = f"{__name__}.{name}"
    if path is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}; the public names are "
            f"in sparta.__all__ (the unicor and todcor atoms import from those packages)"
        )
    module = importlib.import_module(path)
    value = module if path == f"{__name__}.{name}" else getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    """The public names, ``__all__``, whether or not they are resolved yet."""
    return sorted(__all__)
