"""TODCOR: two-dimensional correlation for double-lined (SB2) spectra.

A standalone numpy engine (on unicor's grid primitives) that fits the
velocities *and* relative amplitudes of both components of an SB2
simultaneously on a 2-D ``(v1, v2)`` grid -- Zucker & Mazeh (1994, ApJ 420,
806) recast as a Gaussian profile likelihood in the Fourier domain.

On a uniform log-lambda grid a Doppler shift is a translation, so in the Fourier
domain it is a phase factor: the templates are transformed **once** and reused
at every trial velocity. The amplitudes enter linearly and are solved in closed
form (2x2 weighted least squares) at each node; profiling them out gives the
classic TODCOR correlation map.

Everything takes plain floats / arrays: velocities and the log-lambda shift in
km/s, ``zeta`` (the Fourier conjugate) in s/km, kernel widths as FWHM in km/s.
Velocities follow the 'cz' Doppler convention (lambda_obs/lambda_rest = 1 + v/c)
unless 'relativistic' is requested (``velocity_convention``; see
:func:`unicor.dsp.doppler_factor`).
SPARTA's ``CCF2d`` (Spectrum + two Templates) is a layer on top.

Layout
------
``todcor.estimator`` - sufficient statistics, window pooling, profiling, CRLB.
``todcor.simulate``  - the forward model: composites and best-fit reconstructions.

Grid, FFT and broadening primitives come from :mod:`unicor.dsp` and are
re-exported here, so ``todcor`` is a complete namespace for the method.
"""

from unicor.dsp import (
    C_KMS, broaden_gaussian_residual, broaden_rotational_residual, doppler_factor,
    log_grid_step, log_lambda_grid, parseval_weights, resample, rfft_positive,
    rotational_kernel, shift_kms, tukey_window, velocity_from_log_shift,
)

from .estimator import (
    chi2_from_stats, combine_stats, crlb_from_map, profile_from_stats,
    resampled_noise_psd, todcor_map, window_stats,
)
from .simulate import composite_flux, model_flux, shift_template_flux

__version__ = "0.1.0"

__all__ = [
    'window_stats', 'combine_stats', 'chi2_from_stats', 'profile_from_stats',
    'crlb_from_map', 'todcor_map', 'resampled_noise_psd',
    'shift_template_flux', 'composite_flux', 'model_flux',
    'C_KMS', 'log_lambda_grid', 'log_grid_step', 'resample', 'tukey_window',
    'shift_kms', 'doppler_factor', 'velocity_from_log_shift',
    'rfft_positive', 'parseval_weights', 'rotational_kernel',
    'broaden_gaussian_residual', 'broaden_rotational_residual',
]
