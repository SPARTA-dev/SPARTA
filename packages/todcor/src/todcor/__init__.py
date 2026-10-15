"""TODCOR: two-dimensional correlation for double-lined (SB2) spectra.

A standalone numpy engine (on unicor's grid primitives) that fits the
velocities *and* relative amplitudes of both components of an SB2
simultaneously on a 2-D ``(v1, v2)`` grid -- Zucker & Mazeh (1994, ApJ 420,
806) recast as a Gaussian profile likelihood.

Each window (one echelle order) is fitted on its observed pixels only. The two
templates are resampled onto the window's uniform log-lambda grid *extended* on
each side by the largest trial shift, so a Doppler shift is a translation of
real template pixels into the window (applied as a Fourier phase factor of the
extended array) and nothing wraps around; there is no taper. Each window has a
free additive offset (its continuum level), profiled out. The amplitudes enter
linearly and are solved in closed form at each node -- one ``(a1, a2)`` for all
windows (``amplitudes='shared'``) or one per window (``'per_order'``);
profiling them out gives two equivalent (v1, v2) maps -- the classic TODCOR
two-dimensional correlation R (``output='ccf'``, the default primary map) and
the Gaussian log-likelihood (``output='loglike'``), which share their peak.

Everything takes plain floats / arrays: velocities in km/s, kernel widths as
FWHM in km/s.
Velocities follow the 'cz' Doppler convention (lambda_obs/lambda_rest = 1 + v/c)
unless 'relativistic' is requested (``velocity_convention``; see
:func:`unicor.dsp.doppler_factor`).
SPARTA's ``CCF2d`` (Spectrum + two Templates) is a layer on top.

Layout
------
``todcor.estimator`` - sufficient statistics, window pooling, profiling, CRLB;
                     and the per-window atoms of a multi-window fit
                     (``prepare_window``, ``refine_grid``, ``joint_fit``,
                     ``window_noise``), which SPARTA's ``CCF2d`` composes.
``todcor.simulate``  - the forward model: composites and best-fit reconstructions.

Grid and broadening primitives come from :mod:`unicor.dsp` and are
re-exported here, so ``todcor`` is a complete namespace for the method.
"""

from unicor.dsp import (
    C_KMS, broaden_gaussian_residual, broaden_rotational_residual, doppler_factor,
    log_grid_step, log_lambda_grid, resample, rotational_kernel, shift_kms,
    velocity_from_log_shift,
)

from .estimator import (
    TodcorFit, TodcorWindow, chi2_from_stats, combine_stats, crlb_from_map, joint_fit,
    prepare_window, profile_from_stats, profile_per_window, refine_grid,
    resampled_noise_psd, todcor_map, window_noise, window_stats,
)
from .simulate import composite_flux, model_flux, shift_template_flux

__version__ = "0.2.0"

__all__ = [
    'window_stats', 'combine_stats', 'chi2_from_stats', 'profile_from_stats',
    'profile_per_window',
    'crlb_from_map', 'todcor_map', 'resampled_noise_psd',
    'TodcorWindow', 'prepare_window', 'refine_grid', 'TodcorFit', 'joint_fit', 'window_noise',
    'shift_template_flux', 'composite_flux', 'model_flux',
    'C_KMS', 'log_lambda_grid', 'log_grid_step', 'resample',
    'shift_kms', 'doppler_factor', 'velocity_from_log_shift', 'rotational_kernel',
    'broaden_gaussian_residual', 'broaden_rotational_residual',
]
