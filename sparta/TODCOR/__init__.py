"""TODCOR: two-dimensional correlation for double-lined (SB2) spectra.

The two-component counterpart of UNICOR. UNICOR correlates one observation
against one template and reads a 1-D CCF peak; TODCOR fits the velocities *and*
relative amplitudes of both components simultaneously on a 2-D ``(v1, v2)``
grid -- Zucker & Mazeh (1994, ApJ 420, 806) recast as a Gaussian profile
likelihood in the Fourier domain.

On a uniform log-lambda grid a Doppler shift is a translation, so in the Fourier
domain it is a phase factor: the templates are transformed **once** and reused
at every trial velocity. The amplitudes enter linearly and are solved in closed
form (2x2 weighted least squares) at each node; profiling them out gives the
classic TODCOR correlation map.

Layout
------
``CCF2d``       - the SPARTA-object class (Spectrum + two Templates -> map, RVs).
``todcor_ops``  - the estimator: sufficient statistics, profiling, CRLB.
``simulate``    - the forward model: composites and best-fit reconstructions.

Grid, FFT and broadening primitives are shared with UNICOR and live in
:mod:`sparta.spectral_ops`.
"""

from sparta.spectral_ops import (
    C_KMS, broaden_gaussian_residual, broaden_rotational_residual,
    log_lambda_grid, parseval_weights, resample, rfft_positive,
    rotational_kernel, shift_kms, tukey_window,
)

from .CCF2d import CCF2d
from .simulate import composite_flux, model_flux, shift_template_flux
from .todcor_ops import (
    combine_stats, crlb_from_map, profile_from_stats, todcor_map, window_stats,
)

# The spectral_ops primitives are re-exported (not redefined) so that
# ``sparta.TODCOR`` is a complete namespace for the method -- grid, apodisation,
# Fourier weights, broadening, estimator -- as the examples use it.
__all__ = [
    'CCF2d',
    'window_stats', 'combine_stats', 'profile_from_stats', 'crlb_from_map',
    'todcor_map',
    'shift_template_flux', 'composite_flux', 'model_flux',
    'C_KMS', 'log_lambda_grid', 'resample', 'tukey_window', 'shift_kms',
    'rfft_positive', 'parseval_weights', 'rotational_kernel',
    'broaden_gaussian_residual', 'broaden_rotational_residual',
]
