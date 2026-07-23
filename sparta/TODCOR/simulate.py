#                   -------------------------------------
#                     simulate.py (SPARTA TODCOR forward)
#                   -------------------------------------
# The forward model for the two-component (double-lined) engine: build a
# composite spectrum from two templates at known velocities and amplitudes, and
# reconstruct the best-fit composite for residual plots.
#
# Kept separate from todcor_ops.py so that "makes data" and "fits data" never
# share a file: composite_flux shifts by physical resampling (simulating an
# observation), while model_flux uses the Fourier phase shift (matching the fit
# exactly).
#
# Conventions: wavelengths in Angstrom, velocities in km/s, normalized flux
# (continuum = 1).
#
# Dependencies: numpy, sparta.spectral_ops.
"""Forward model for the SPARTA TODCOR engine."""

import numpy as np

from sparta.spectral_ops import C_KMS, resample, shift_kms


# =============================================================================
# =============================================================================
def shift_template_flux(wave_t, flux_t, wave_grid, v, fill=1.0):
    """Resample a template, Doppler-shifted by ``v`` (km/s), onto ``wave_grid``.

    A line emitted at rest wavelength ``lam`` appears at ``lam * (1 + v/c)``, so
    the shifted template sampled at ``wave_grid`` is the rest template evaluated
    at ``wave_grid / (1 + v/c)``.

    Parameters
    ----------
    wave_t, flux_t : ndarray
        Rest-frame template (Angstrom; normalized flux, continuum 1).
    wave_grid : ndarray
        Output wavelength grid (Angstrom).
    v : float
        Radial velocity (km/s); positive red-shifts.
    fill : float
        Value used where the shifted template does not cover ``wave_grid``
        (default 1.0, the continuum).

    Returns
    -------
    flux : ndarray
        The shifted template sampled on ``wave_grid``.
    """
    return resample(wave_t, flux_t, wave_grid / (1.0 + v / C_KMS), fill=fill)


# =============================================================================
# =============================================================================
def composite_flux(wave_grid, templates, velocities, amplitudes):
    """Build a noiseless composite (continuum-normalised) by physical shifting.

    Parameters
    ----------
    wave_grid : ndarray
        Output wavelength grid (Angstrom).
    templates : sequence of (wave, flux)
        Each component's rest-frame template (flux normalized to continuum 1).
    velocities : sequence of float
        Radial velocity of each component (km/s).
    amplitudes : sequence of float
        Linear scaling (light/dilution fraction) of each component's *lines*.

    Returns
    -------
    flux : ndarray
        ``1 + sum_i amplitudes[i] * (shifted_template_i - 1)``.
    """
    g = np.zeros_like(np.asarray(wave_grid, float))
    for (wave_t, flux_t), v, a in zip(templates, velocities, amplitudes):
        g += a * (shift_template_flux(wave_t, flux_t, wave_grid, v) - 1.0)
    return 1.0 + g


# =============================================================================
# =============================================================================
def model_flux(wave_grid, dv, g1, g2, v1, v2, a1, a2):
    """Reconstruct the best-fit composite flux on the grid (for residual plots).

    Uses the same Fourier phase-shift model as the fit, so the returned flux is
    exactly ``1 + a1*shift(g1, v1) + a2*shift(g2, v2)`` evaluated via FFT --
    the circular shift the estimator actually assumes, not a resampled one.

    Parameters
    ----------
    wave_grid : ndarray
        The uniform log-lambda grid (Angstrom); only its length is used.
    dv : float
        Velocity step of the grid (km/s).
    g1, g2 : ndarray
        Template residuals on that grid (continuum 0).
    v1, v2 : float
        Component velocities (km/s).
    a1, a2 : float
        Component amplitudes (dimensionless).

    Returns
    -------
    flux : ndarray
        The modelled composite flux (continuum 1).
    """
    n = len(wave_grid)
    zeta = np.fft.rfftfreq(n, d=dv)
    G1 = np.fft.rfft(np.asarray(g1, float))
    G2 = np.fft.rfft(np.asarray(g2, float))
    s1, s2 = shift_kms(v1), shift_kms(v2)
    M = (a1 * G1 * np.exp(-2j * np.pi * zeta * s1)
         + a2 * G2 * np.exp(-2j * np.pi * zeta * s2))
    return 1.0 + np.fft.irfft(M, n=n)
