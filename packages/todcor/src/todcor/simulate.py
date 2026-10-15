#                   -------------------------------------
#                     todcor/simulate.py (TODCOR forward)
#                   -------------------------------------
# The forward model for the two-component (double-lined) engine: build a
# composite spectrum from two templates at known velocities and amplitudes, and
# reconstruct the best-fit composite for residual plots.
#
# Kept separate from estimator.py so that "makes data" and "fits data" never
# share a file: composite_flux shifts by physical resampling (simulating an
# observation), while model_flux uses the fit's own Fourier shift of the
# extended templates (matching the fit exactly).
#
# Conventions: wavelengths in Angstrom, velocities in km/s ('cz' Doppler
# convention unless velocity_convention='relativistic'), normalized flux
# (continuum = 1).
#
# Dependencies: numpy, unicor.dsp, the estimator's shift.
"""Forward model for the TODCOR engine."""

import numpy as np

from unicor.dsp import doppler_factor, resample

from .estimator import _shifted_rows


# =============================================================================
# =============================================================================
def shift_template_flux(wave_t, flux_t, wave_grid, v, fill=1.0, velocity_convention='cz',
                        kind='linear'):
    """Resample a template, Doppler-shifted by ``v`` (km/s), onto ``wave_grid``.

    A line emitted at rest wavelength ``lam`` appears at ``lam * D(v)``, with
    ``D`` the Doppler factor of ``velocity_convention``
    (:func:`unicor.dsp.doppler_factor`; ``1 + v/c`` under 'cz'), so the shifted
    template sampled at ``wave_grid`` is the rest template evaluated at
    ``wave_grid / D(v)``.

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
    velocity_convention : {'cz', 'relativistic'}
        Doppler convention of ``v``. Default 'cz'.
    kind : {'linear', 'cubic'}
        Interpolation of the template (:func:`unicor.dsp.resample`). Linear
        is enough for a template sampled much finer than its lines; use
        'cubic' when it is not (e.g. PHOENIX at ~0.6 km/s per pixel), where
        linear interpolation biases the shift at the m/s level.

    Returns
    -------
    flux : ndarray
        The shifted template sampled on ``wave_grid``.
    """
    return resample(wave_t, flux_t,
                    np.asarray(wave_grid, float) / doppler_factor(v, velocity_convention),
                    fill=fill, kind=kind)


# =============================================================================
# =============================================================================
def composite_flux(wave_grid, templates, velocities, amplitudes, velocity_convention='cz',
                   kind='linear'):
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
    velocity_convention : {'cz', 'relativistic'}
        Doppler convention of ``velocities``. Default 'cz'.
    kind : {'linear', 'cubic'}
        Interpolation of the templates, see :func:`shift_template_flux`.

    Returns
    -------
    flux : ndarray
        ``1 + sum_i amplitudes[i] * (shifted_template_i - 1)``.
    """
    g = np.zeros_like(np.asarray(wave_grid, float))
    for (wave_t, flux_t), v, a in zip(templates, velocities, amplitudes):
        g += a * (shift_template_flux(wave_t, flux_t, wave_grid, v,
                                      velocity_convention=velocity_convention,
                                      kind=kind) - 1.0)
    return 1.0 + g


# =============================================================================
# =============================================================================
def model_flux(g1, g2, n, dv, v1, v2, a1, a2, velocity_convention='cz'):
    """The best-fit composite on a window's pixels, as the fit models it.

    The two templates, given on the window's grid extended on each side (as
    :func:`todcor.estimator.prepare_window` makes them), are shifted with the
    same Fourier phase factors as :func:`todcor.estimator.window_stats` and
    read on the window's ``n`` pixels. The window's free offset (its
    continuum level) is the caller's: for the fitted one, add the mean of
    ``data - model``.

    Parameters
    ----------
    g1, g2 : ndarray
        The template residuals on the extended grid, as :func:`todcor.estimator.prepare_window`
        makes them (length ``n + 2e`` or ``n + 2e + 1``).
    n : int
        Pixels in the window.
    dv : float
        Velocity step of the grid (km/s), ``c`` times its actual ln(lambda)
        step.
    v1, v2 : float
        Component velocities (km/s).
    a1, a2 : float
        Component amplitudes (dimensionless).
    velocity_convention : {'cz', 'relativistic'}
        Doppler convention of ``v1``, ``v2``. Default 'cz'.

    Returns
    -------
    ndarray, shape (n,)
        ``a1 * G1(v1) + a2 * G2(v2)`` on the window (flux units, no offset).
    """
    rows1 = _shifted_rows(g1, n, dv, [v1], velocity_convention)[0]
    rows2 = _shifted_rows(g2, n, dv, [v2], velocity_convention)[0]
    return a1 * rows1 + a2 * rows2
