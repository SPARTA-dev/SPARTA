"""Velocity-space spectral primitives on plain numpy arrays.

Log-lambda grids, resampling, apodisation, Doppler shifts, rotational and
Gaussian broadening, and the Fourier helpers used by the correlation engines.
No I/O, no network, no astropy: the shared numerical layer under UNICOR
(:mod:`unicor.ccf`), TODCOR (the ``todcor`` package), and SPARTA's TIRAVEL and
USuRPER engines.

Conventions
-----------
- Velocities and velocity steps in km/s; wavelengths in Angstrom.
- Broadening operates on normalized flux (continuum = 1) via the residual
  ``flux - 1``, so the continuum is preserved and the edges zero-pad naturally.
- The Fourier conjugate of a log-lambda (velocity) axis, ``zeta``, is in s/km.

Provenance
----------
The rotational (Gray 2005) and Gaussian broadening kernels, and the log-lambda
grid / Parseval-weight helpers, are adapted from the author's ``specbin.todcor``
(Binaries project); they replace the former PyAstronomy dependency.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import tukey as _tukey

C_KMS = 299792.458  # speed of light, km/s
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def dv_from_wave(wave):
    """Velocity step (km/s) of a wavelength grid, from the median d(lambda)/lambda.

    Exact for a log-lambda grid; a close approximation over a narrow linear grid.
    """
    wave = np.asarray(wave, float)
    return C_KMS * np.median(np.diff(wave) / wave[:-1])


def log_lambda_grid(lam_lo, lam_hi, dv):
    """Uniform log-lambda (constant-velocity-step) wavelength grid.

    A Doppler shift is a pure translation on this grid, which is what makes the
    cross-correlation (UNICOR) and the Fourier phase-shift model (TODCOR) valid.

    Parameters
    ----------
    lam_lo, lam_hi : float
        Wavelength bounds (Angstrom); ``lam_hi`` is included only if it falls on
        a grid node.
    dv : float
        Velocity step of the grid (km/s).

    Returns
    -------
    wave : ndarray
        Geometrically spaced wavelength grid (Angstrom).
    u : ndarray
        Velocity coordinate ``u = c * ln(wave / lam_lo)`` (km/s): uniform step
        ``dv``, starting at 0.
    """
    dln = dv / C_KMS
    n = int(np.floor(np.log(lam_hi / lam_lo) / dln)) + 1
    k = np.arange(n)
    return lam_lo * np.exp(k * dln), k * dv


def tukey_window(n, alpha=0.1):
    """Tukey (tapered-cosine) apodisation window of length ``n``.

    A mild taper (small ``alpha``) suppresses FFT edge ringing. Choose the
    analysis window so the spectra sit at their continuum near the edges; the
    taper then acts only on already-flat regions, and the circular (Fourier)
    shift model stays an accurate stand-in for a physical shift.

    Note the taper makes the per-pixel noise heteroscedastic
    (``sigma_n = w_n * sigma``), which the white-noise weighting downstream
    does not model -- only the overall scale is recoverable, from reduced-chi2.
    """
    return _tukey(n, alpha)


def shift_kms(v):
    """Log-lambda shift ``s = c * ln(1 + v/c)`` (km/s) for radial velocity ``v``.

    The relativistically consistent translation distance along the log-lambda
    axis: a line at rest wavelength ``lam`` moves to ``lam * (1 + v/c)``, i.e.
    by ``s`` in the velocity coordinate ``u = c * ln(lam / lam_0)``.
    """
    return C_KMS * np.log1p(np.asarray(v, float) / C_KMS)


def rfft_positive(g, dv):
    """Real FFT of ``g`` on positive frequencies only, DC dropped.

    Dropping the DC bin also removes the mean of ``g``, so any residual
    continuum offset is projected out for free.

    Parameters
    ----------
    g : ndarray
        Real signal on a uniform grid (in practice a residual flux, continuum 0).
    dv : float
        Grid step (km/s) -- sets the frequency axis.

    Returns
    -------
    G : ndarray (complex)
        ``np.fft.rfft(g)[1:]``.
    zeta : ndarray
        Conjugate frequencies ``np.fft.rfftfreq(N, d=dv)[1:]`` (s/km).
    """
    return np.fft.rfft(g)[1:], np.fft.rfftfreq(len(g), d=dv)[1:]


def parseval_weights(n, sigma=1.0):
    """Per-mode weights making a Fourier chi2 equal the real-space chi2.

    For a real signal of length ``n`` with white, homoscedastic per-pixel noise
    ``sigma``, the positive interior modes carry ``2 / (n * sigma**2)`` while the
    DC and (even ``n``) Nyquist modes carry half that. Then, for a real residual
    ``r`` with transform ``R``::

        sum_k w_k |R_k|**2  ==  sum_n r_n**2 / sigma**2

    Returns
    -------
    w : ndarray
        Weights aligned with :func:`rfft_positive` output, i.e. DC dropped.
    """
    w = np.full(len(np.fft.rfftfreq(n, d=1.0)), 2.0 / n)
    w[0] = 1.0 / n                       # DC (dropped below)
    if n % 2 == 0:
        w[-1] = 1.0 / n                  # Nyquist
    return (w / sigma ** 2)[1:]


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


def broaden_rotational_residual(residual, dv, vsini, epsilon=0.6):
    """Rotationally broaden a *residual* flux array (continuum = 0).

    The primitive form: convolution is linear, so broadening the residual is
    what actually happens; the continuum is untouched by construction and the
    edges zero-pad naturally. UNICOR works in normalized flux and calls
    :func:`broaden_rotational`; TODCOR works in residuals and calls this.
    """
    if vsini <= 0:
        return np.asarray(residual, float)
    kernel = rotational_kernel(dv, vsini, epsilon)
    return np.convolve(np.asarray(residual, float), kernel, mode="same")


def broaden_gaussian_residual(residual, dv, fwhm_kms):
    """Gaussian-broaden a *residual* flux array (continuum = 0) by ``fwhm_kms``.

    For resolving power ``R``, use ``fwhm_kms = c / R``. See
    :func:`broaden_rotational_residual` on the residual vs flux forms.
    """
    if fwhm_kms <= 0:
        return np.asarray(residual, float)
    sigma_pix = (fwhm_kms * _FWHM_TO_SIGMA) / dv
    return gaussian_filter1d(np.asarray(residual, float), sigma_pix,
                             mode="constant", cval=0.0)


def broaden_rotational(flux, dv, vsini, epsilon=0.6):
    """Rotationally broaden normalized flux (continuum = 1) by ``vsini`` km/s."""
    return 1.0 + broaden_rotational_residual(
        np.asarray(flux, float) - 1.0, dv, vsini, epsilon)


def broaden_gaussian(flux, dv, fwhm_kms):
    """Gaussian-broaden normalized flux by ``fwhm_kms`` km/s (instrumental).

    For resolving power ``R``, use ``fwhm_kms = c / R``.
    """
    return 1.0 + broaden_gaussian_residual(
        np.asarray(flux, float) - 1.0, dv, fwhm_kms)


def doppler_shift_wave(wave, v):
    """Doppler-shift a wavelength array by velocity ``v`` (km/s), classical.

    Returns ``wave * (1 + v/c)``; positive ``v`` red-shifts (increases lambda).
    """
    return np.asarray(wave, float) * (1.0 + v / C_KMS)


def resample(wave, flux, wave_grid, fill=None, kind='linear'):
    """Interpolate ``flux(wave)`` onto ``wave_grid``.

    Parameters
    ----------
    wave, flux : ndarray
        Source grid (Angstrom, need not be uniform) and its flux.
    wave_grid : ndarray
        Target wavelength grid (Angstrom).
    fill : float or None
        Value for target points outside the source coverage. ``None`` (default)
        clamps to the edge values; pass ``1.0`` to fill with the normalized
        continuum, which is what the shift-and-resample forward model wants so a
        template shifted off its own span does not smear its edge pixel.
    kind : {'linear', 'cubic'}
        ``'linear'`` (default) is the fast path used by the CCF engines, where
        the grids are heavily oversampled. ``'cubic'`` uses a natural cubic
        spline; prefer it when a spectrum is shifted by many pixels and the
        interpolation error would otherwise show up as lost correlation.
    """
    wave = np.asarray(wave, float)
    flux = np.asarray(flux, float)

    if kind == 'linear':
        if fill is None:
            return np.interp(wave_grid, wave, flux)
        return np.interp(wave_grid, wave, flux, left=fill, right=fill)

    if kind != 'cubic':
        raise ValueError(f"kind must be 'linear' or 'cubic', got {kind!r}")

    out = CubicSpline(wave, flux, extrapolate=False)(wave_grid)
    outside = np.isnan(out)
    if outside.any():
        # CubicSpline refuses to extrapolate; apply the requested edge policy.
        edge = np.interp(np.asarray(wave_grid, float)[outside], wave, flux) \
            if fill is None else fill
        out[outside] = edge
    return out


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

