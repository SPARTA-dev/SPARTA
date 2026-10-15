"""The public ``import sparta`` surface, pinned.

Every change to what users import must be a deliberate edit of the lists
below rather than a side effect. Checked in a fresh interpreter, so
submodules that other tests import cannot leak into ``dir(sparta)``.

The layout is flat: one module per concern (``sparta.spectrum``,
``sparta.ccf1d``, ...), its class re-exported at the top level. The class and
its module differ in case only for ``Observations``/``observations`` and
``TIRAVEL``/``tiravel``. The atoms of ``sparta.preprocess``, ``sparta.pdc``,
``sparta.tiravel_ops`` and ``phoenix_spectrum`` are re-exported too. Every
top-level name is resolved lazily (PEP 562), so ``dir(sparta)`` is exactly
``__all__``: submodules are reached by importing them.

The classes' public methods and their parameter names are pinned too (0.3.0
made them snake_case, see CHANGELOG.md), and ``__version__`` must be the
newest CHANGELOG section.
"""
import functools
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
_PATH = os.pathsep.join(str(ROOT / p) for p in ("src", "packages/unicor/src",
                                                  "packages/todcor/src"))

ALL = ["Spectrum", "Template", "Observations", "TimeSeries", "CCF1d", "CCF2d",
       "TIRAVEL", "PeriodicityDetector", "instruments", "plotting",
       "resample_flux", "trim_margins", "resample_to_linear_grid", "resample_to_log_grid",
       "coarsest_native_step", "remove_cosmics", "bandpass", "filtered_noise_fraction",
       "resampling_noise_psd",
       "native_phases", "phoenix_spectrum",
       "rest_frame_wavelengths", "common_grid", "align_spectra", "normalize_rows",
       "correlation_matrix", "alignment", "coadd", "TiravelResult", "build_tiravel",
       "scalar_distance_matrix", "error_aware_distance_matrix", "spectral_distance_matrix",
       "phase_distance_matrix", "double_centering", "unbiased_u_centering", "inner_prod",
       "norm_z", "partial_centering", "pdc_power", "pdc_pvalue", "pdc_power_level",
       "get_wv_grid", "resample_spec_to_common_grid"]

# name -> "class|function <module>.<qualname>" or "module <name>"
TOP_LEVEL = {
    "CCF1d": "class sparta.ccf1d.CCF1d",
    "CCF2d": "class sparta.ccf2d.CCF2d",
    "Observations": "class sparta.observations.Observations",
    "PeriodicityDetector": "class sparta.periodogram.PeriodicityDetector",
    "Spectrum": "class sparta.spectrum.Spectrum",
    "TIRAVEL": "class sparta.tiravel.TIRAVEL",
    "Template": "class sparta.template.Template",
    "TimeSeries": "class sparta.timeseries.TimeSeries",
    "instruments": "module sparta.instruments",
    "plotting": "module sparta.plotting",
    **{n: f"function sparta.preprocess.{n}"
       for n in ("resample_flux", "trim_margins", "resample_to_linear_grid",
                 "resample_to_log_grid", "coarsest_native_step", "remove_cosmics", "bandpass",
                 "filtered_noise_fraction", "resampling_noise_psd", "native_phases")},
    "phoenix_spectrum": "function sparta.template.phoenix_spectrum",
    **{n: f"function sparta.tiravel_ops.{n}"
       for n in ("rest_frame_wavelengths", "common_grid", "align_spectra", "normalize_rows",
                 "correlation_matrix", "alignment", "coadd", "build_tiravel")},
    "TiravelResult": "class sparta.tiravel_ops.TiravelResult",
    **{n: f"function sparta.pdc.{n}"
       for n in ("scalar_distance_matrix", "error_aware_distance_matrix",
                 "spectral_distance_matrix", "phase_distance_matrix", "double_centering",
                 "unbiased_u_centering", "inner_prod", "norm_z", "partial_centering",
                 "pdc_power", "pdc_pvalue", "pdc_power_level", "get_wv_grid",
                 "resample_spec_to_common_grid")},
}

# The deep import paths the examples and docs use: module -> names that must
# resolve there (existence, not an exhaustive list).
DEEP = {
    "sparta.spectrum": ["Spectrum", "resample_flux"],
    "sparta.preprocess": ["bandpass", "coarsest_native_step", "filtered_noise_fraction",
                          "native_phases",
                          "remove_cosmics", "resample_flux", "resample_to_linear_grid",
                          "resample_to_log_grid", "resampling_noise_psd", "trim_margins"],
    "sparta.template": ["Template", "phoenix_spectrum"],
    "sparta.ccf1d": ["CCF1d"],
    "sparta.ccf2d": ["CCF2d"],
    "sparta.tiravel": ["TIRAVEL"],
    "sparta.tiravel_ops": ["TiravelResult", "align_spectra", "alignment",
                           "build_tiravel", "coadd", "common_grid",
                           "correlation_matrix", "normalize_rows",
                           "rest_frame_wavelengths"],
    "sparta.timeseries": ["TimeSeries", "is_numeric_scalar"],
    "sparta.periodogram": ["PeriodicityDetector"],
    "sparta.pdc": ["double_centering", "error_aware_distance_matrix", "get_wv_grid",
                   "inner_prod", "norm_z", "partial_centering", "pdc_power",
                   "pdc_power_level", "pdc_pvalue", "phase_distance_matrix", "resample_spec_to_common_grid",
                   "scalar_distance_matrix", "spectral_distance_matrix",
                   "unbiased_u_centering"],
    "sparta.observations": ["Observations"],
    "sparta.plotting": ["DATA_COLOR", "DLNL_LABEL", "EXTRA_COLOR", "GREY", "MAP_CMAP",
                        "MODEL_COLOR", "SIGMA_LEVELS", "STYLE", "apply_style", "dlnl_norm",
                        "map_colorbar", "map_image", "mark_fit", "mark_truth", "plot_curve",
                        "plot_data", "plot_extra", "plot_model", "plot_points"],
    "sparta.instruments": ["Component", "Instrument", "LoadedSpectrum", "MODULES", "Product",
                           "REGISTRY", "Recommended", "Simulation", "Truth", "get",
                           "render_markdown"],
    "sparta.instruments.simulated": ["Component", "Simulation", "Truth", "component_spectra",
                                     "component_templates", "simulate", "upper_envelope"],
}

# class -> public method -> its parameter names (without self/cls).
METHODS = {
    "Spectrum": {
        "barycentric_correction": [],
        "filter_spectrum": ["continuum_scale_kms", "noise_scale_kms", "order"],
        "from_spectrum1d": ["spectra", "bjd", "bcv", "name"],
        "interpolate": ["delta", "method"],
        "native_steps": [],
        "orders": "property",
        "preprocess": ["n_trim", "clean_margins", "remove_nans", "delta", "cosmic_sigma",
                       "continuum_scale_kms", "noise_scale_kms"],
        "remove_cosmics": ["sigma_upper"],
        "select_orders": ["orders", "remove"],
        "trim": ["n_trim", "n_trim_left", "n_trim_right", "clean_margins", "remove_nans",
                 "margin_fraction", "margin_window"],
    },
    "Template": {
        "broaden_instrumental": ["resolution"],
        "broaden_rotational": ["vsini", "epsilon"],
        "cut_multiorder": ["wv_bounds"],
        "cut_multiorder_like": ["spectrum", "margins"],
        "doppler": ["vel", "wv", "velocity_convention"],
        "from_arrays": ["wave", "flux"],
        "from_mask": ["path", "wave_range", "allow_download"],
        "from_phoenix": ["teff", "logg", "feh", "alpha", "wave_range", "air",
                         "allow_download"],
        "integrate_spec": ["integration_ratio", "wv", "sp"],
        "save_template_model": ["title", "path"],
    },
    "CCF1d": {
        "calc_bis": ["x", "y", "bisect_val", "n_ord"],
        "combine": [],
        "cross_correlate": ["spectrum", "template", "dv", "vel_bound", "errors_per_order",
                            "velocity_convention", "output"],
        "extract_rv": ["x", "y", "vel", "n_ord"],
        "plot": ["plot_combined", "plot_orders", "orders", "alpha"],
        "subpixel_peak": ["vels", "ccf", "v"],
    },
    "CCF2d": {
        "best_fit_spectrum": ["order"],
        "cross_correlate": ["spectrum", "template1", "template2", "dv", "vel_bound", "dvel",
                            "v1_grid", "v2_grid", "sigma", "nonneg", "amplitudes",
                            "orders", "keep_windows", "velocity_convention", "output"],
        "plot": ["marginals", "figsize"],
    },
    "TIRAVEL": {
        "build_template": ["spectra", "velocities", "orders", "reference", "margin", "kind",
                           "keep_aligned"],
        "plot_template": ["order", "show_epochs", "alpha", "figsize"],
        "plot_weights": ["times", "figsize"],
    },
    "Observations": {
        "build_tiravel_template": ["velocities", "kwargs"],
        "calc_bis": ["bisect_val", "use_combined"],
        "calc_rv_against_template": ["template", "dv", "vel_bound", "errors_per_order",
                                     "combine_ccfs", "velocity_convention", "output",
                                     "progress"],
        "calc_rv_against_two_templates": ["template1", "template2", "dv", "vel_bound", "dvel",
                                          "sigma", "nonneg", "velocity_convention", "output",
                                          "progress", "kwargs"],
        "clean_time_series": ["max_vel", "min_vel", "max_time", "nan_flag", "sample_rate"],
        "init_periodicity_detector": ["freq_range", "periodogram_grid_resolution"],
        "preprocess": ["n_trim", "clean_margins", "remove_nans", "delta", "cosmic_sigma",
                       "continuum_scale_kms", "noise_scale_kms", "progress"],
        "select_orders": ["orders", "remove"],
    },
    "TimeSeries": {
        "plot_velocities": ["figsize"],
    },
    "TiravelResult": {},
    "PeriodicityDetector": {
        "calc_gls": [],
        "calc_partial_periodogram": ["partial_type", "velocity_convention"],
        "calc_pdc": ["calc_biased_flag", "calc_unbiased_flag", "errors"],
        "calc_pdc_pval": ["values", "inverse_sf"],
        "calc_usurper": ["calc_biased_flag", "calc_unbiased_flag"],
        "plot_periodograms": ["plot_vals", "annotate_pval", "figsize"],
    },
}

_PROBE = r"""
import importlib, inspect, json, sys
import sparta

def kind(obj):
    if inspect.ismodule(obj):
        return "module " + obj.__name__
    if inspect.isclass(obj):
        return "class " + obj.__module__ + "." + obj.__qualname__
    if inspect.isfunction(obj):
        return "function " + obj.__module__ + "." + obj.__qualname__
    return type(obj).__name__

def params(member):
    if isinstance(member, property):
        return "property"
    func = member.__func__ if isinstance(member, (classmethod, staticmethod)) else member
    return [p for p in inspect.signature(func).parameters if p not in ("self", "cls")]

top = {n: kind(getattr(sparta, n)) for n in dir(sparta) if not n.startswith("_")}
methods = {n: {m: params(v) for m, v in vars(getattr(sparta, n)).items()
               if not m.startswith("_") and (callable(v) or isinstance(v, property)
                                             or isinstance(v, (classmethod, staticmethod)))}
           for n in sparta.__all__ if inspect.isclass(getattr(sparta, n))}
deep = {m: [n for n in json.loads(sys.argv[1])[m] if hasattr(importlib.import_module(m), n)]
        for m in json.loads(sys.argv[1])}
print(json.dumps({"top": top, "deep": deep, "all": sparta.__all__,
                  "methods": methods, "version": sparta.__version__}))
"""


@functools.lru_cache(maxsize=None)
def _probe():
    env = dict(os.environ, PYTHONPATH=_PATH + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run([sys.executable, "-c", _PROBE, json.dumps(DEEP)],
                          capture_output=True, text=True, env=env, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_import_sparta_surface_is_pinned():
    probe = _probe()
    assert probe["top"] == TOP_LEVEL
    assert probe["deep"] == DEEP
    assert probe["all"] == ALL


def test_public_methods_and_parameters_are_pinned():
    assert _probe()["methods"] == METHODS


def test_version_is_the_newest_changelog_section():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    newest = re.search(r"^## \[?(\d+\.\d+\.\d+)", changelog, flags=re.M)
    assert newest is not None, "CHANGELOG.md has no '## <version>' section"
    assert _probe()["version"] == newest.group(1)
