"""Reusable spectral operations for SPARTA (velocity-space primitives).

Pure functions on plain numpy arrays: no I/O, no GUI, no network, no astropy
Quantity in the inner loops. Shared by the UNICOR CCF/RV engine and the
USuRPER periodograms, so broadening, resampling, shifting, and correlation
live in one place instead of being tangled into Template/Spectrum.

Conventions
-----------
- Velocities and velocity steps in km/s; wavelengths in Angstrom.
- Broadening operates on normalized flux (continuum = 1) via the residual
  ``flux - 1``, so the continuum is preserved and the edges zero-pad naturally.

Provenance
----------
The rotational (Gray 2005) and Gaussian broadening kernels are adapted from the
author's ``specbin.todcor.dsp`` (Binaries project) and reimplemented here to
keep SPARTA self-contained (removes the PyAstronomy dependency).
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate as _correlate

C_KMS = 299792.458  # speed of light, km/s
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def dv_from_wave(wave):
    """Velocity step (km/s) of a wavelength grid, from the median d(lambda)/lambda.

    Exact for a log-lambda grid; a close approximation over a narrow linear grid.
    """
    wave = np.asarray(wave, float)
    return C_KMS * np.median(np.diff(wave) / wave[:-1])


def rotational_kernel(dv, vsini, epsilon=0.6):
    """Unit-sum rotational broadening kernel on a velocity grid (Gray 2005).

    Parameters
    ----------
    dv : float
        Velocity step of the grid (km/s).
    vsini : float
        Projected rotational velocity (km/s).
    epsilon : float
        Linear limb-darkening coefficient (0-1).

    Returns
    -------
    kernel : ndarray
        Odd-length, unit-sum kernel over ``[-vsini, vsini]`` sampled at ``dv``.
    """
    n = int(np.floor(vsini / dv))
    v = np.arange(-n, n + 1) * dv
    x = np.clip(1.0 - (v / vsini) ** 2, 0.0, None)
    num = 2.0 * (1.0 - epsilon) * np.sqrt(x) + 0.5 * np.pi * epsilon * x
    kernel = num / (np.pi * vsini * (1.0 - epsilon / 3.0))
    total = kernel.sum()
    return kernel / total if total > 0 else kernel


def broaden_rotational(flux, dv, vsini, epsilon=0.6):
    """Rotationally broaden normalized flux (continuum = 1) by ``vsini`` km/s."""
    if vsini <= 0:
        return np.asarray(flux, float)
    kernel = rotational_kernel(dv, vsini, epsilon)
    residual = np.asarray(flux, float) - 1.0
    return 1.0 + np.convolve(residual, kernel, mode="same")


def broaden_gaussian(flux, dv, fwhm_kms):
    """Gaussian-broaden normalized flux by ``fwhm_kms`` km/s (instrumental).

    For resolving power ``R``, use ``fwhm_kms = c / R``.
    """
    if fwhm_kms <= 0:
        return np.asarray(flux, float)
    sigma_pix = (fwhm_kms * _FWHM_TO_SIGMA) / dv
    residual = np.asarray(flux, float) - 1.0
    return 1.0 + gaussian_filter1d(residual, sigma_pix, mode="constant", cval=0.0)


def doppler_shift_wave(wave, v):
    """Doppler-shift a wavelength array by velocity ``v`` (km/s), classical.

    Returns ``wave * (1 + v/c)``; positive ``v`` red-shifts (increases lambda).
    """
    return np.asarray(wave, float) * (1.0 + v / C_KMS)


def resample(wave, flux, wave_grid):
    """Linearly interpolate ``flux(wave)`` onto ``wave_grid`` (edge-clamped)."""
    return np.interp(wave_grid, np.asarray(wave, float), np.asarray(flux, float))


def normalize(flux):
    """Mean-subtract and L2-normalize a flux vector (zero vector -> unchanged)."""
    f = np.asarray(flux, float)
    f = f - f.mean()
    norm = np.sqrt((f ** 2).sum())
    return f / norm if norm > 0 else f


def zero_lag_correlation(a, b):
    """Unshifted (zero-lag) correlation: the inner product of two vectors.

    On L2-normalized inputs this is the cosine similarity used by USuRPER; no
    Fourier transform or velocity scan is involved.
    """
    return float(np.sum(np.asarray(a, float) * np.asarray(b, float)))


def ccf_vs_velocity(template, signal, dv, vmax):
    """Normalized cross-correlation of ``template`` and ``signal`` vs velocity.

    Both arrays must be sampled on the SAME uniform log-lambda grid with
    velocity step ``dv`` (km/s), so a shift of k pixels is k*dv km/s. Both are
    mean-subtracted, so this is the Pearson correlation of the line residuals
    (no continuum-DC taper). The scan is symmetric over ``[-vmax, vmax]``, so
    the whole axis is uniformly valid -- unlike the legacy asymmetric,
    unequal-length correlation (see KNOWN_TRAPS.md).

    Returns
    -------
    v : ndarray
        Symmetric velocity lags (km/s): ``arange(-K, K+1) * dv``.
    ccf : ndarray
        Normalized correlation in [-1, 1]. Peaks at the velocity by which
        ``signal`` is red-shifted relative to ``template``.
    """
    t = np.asarray(template, float)
    s = np.asarray(signal, float)
    if t.shape != s.shape:
        raise ValueError("template and signal must share the same grid/length")
    t = t - t.mean()
    s = s - s.mean()
    norm = np.sqrt((t ** 2).sum() * (s ** 2).sum())
    full = _correlate(s, t, mode="full", method="fft")
    mid = len(t) - 1
    K = min(int(round(vmax / dv)), mid)
    ccf = full[mid - K: mid + K + 1] / norm
    v = np.arange(-K, K + 1) * dv
    return v, ccf
