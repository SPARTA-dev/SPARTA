"""UNICOR: one-dimensional cross-correlation radial velocities.

A standalone, numpy + scipy engine for measuring radial velocities of
single-lined spectra: correlate an observation against a template on a uniform
log-lambda grid, combine orders by maximum likelihood (Zucker 2003, MNRAS 342,
1291), and read the RV, its uncertainty and the bisector span off the peak.

Everything takes plain floats / arrays: velocities in km/s, wavelengths in
Angstrom. SPARTA's ``Spectrum`` / ``Template`` / ``CCF1d`` objects are a layer
on top of this package, not a requirement for it.

Layout
------
``unicor.dsp`` - log-lambda grids, resampling, apodisation, shifts, broadening,
                 Fourier helpers (shared with the ``todcor`` package).
``unicor.ccf`` - CCF vs velocity, multi-order combination, sub-pixel peak,
                 ML uncertainty, bisector span.
"""

from .ccf import (
    bisector_span, ccf_at, ccf_vs_velocity, combine_ccfs, rv_uncertainty,
    subpixel_peak,
)
from .dsp import (
    C_KMS, broaden_gaussian, broaden_gaussian_residual, broaden_rotational,
    broaden_rotational_residual, doppler_shift_wave, dv_from_wave,
    log_lambda_grid, normalize, parseval_weights, resample, rfft_positive,
    rotational_kernel, shift_kms, tukey_window, zero_lag_correlation,
)

__version__ = "0.1.0"

__all__ = [
    'ccf_vs_velocity', 'combine_ccfs', 'subpixel_peak', 'ccf_at',
    'rv_uncertainty', 'bisector_span',
    'C_KMS', 'dv_from_wave', 'log_lambda_grid', 'resample', 'tukey_window',
    'shift_kms', 'doppler_shift_wave', 'rfft_positive', 'parseval_weights',
    'rotational_kernel', 'broaden_rotational', 'broaden_gaussian',
    'broaden_rotational_residual', 'broaden_gaussian_residual', 'normalize',
    'zero_lag_correlation',
]
