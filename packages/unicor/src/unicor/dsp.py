"""Velocity-space spectral primitives on plain numpy arrays.

Log-lambda grids, resampling, apodisation, Doppler shifts, rotational and
Gaussian broadening, and the Fourier helpers used by the correlation engines.
No I/O, no network, no astropy: the shared numerical layer under UNICOR
(:mod:`unicor.ccf`), TODCOR (the ``todcor`` package), and SPARTA's TIRAVEL and
USuRPER engines.

Conventions
-----------
- Velocities and velocity steps in km/s; wavelengths in Angstrom.
- Radial velocities follow one of two Doppler conventions (``convention`` /
  ``velocity_convention``; see :func:`doppler_factor`), 'cz' by default.
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

    Exact for a log-lambda grid; for a linear grid, whose velocity step falls
    from the blue end to the red end, it is the mid-grid value.

    Parameters
    ----------
    wave : array_like
        Increasing wavelength grid (Angstrom), at least two points.

    Returns
    -------
    float
        ``c * median(diff(wave) / wave[:-1])`` (km/s).
    """
    wave = np.asarray(wave, float)
    return C_KMS * np.median(np.diff(wave) / wave[:-1])



def log_grid_dln(wave, rtol=1e-3):
    """The ln(lambda) step of a uniform log-lambda grid, after verifying it.

    The CCF and the Fourier shift model are only valid on a velocity-even grid;
    a grid that merely *claims* to be one (e.g. after pixels were dropped) must
    be caught before correlating, not after.

    A velocity read off a lag must use this *actual* step: a grid built with
    ``ln(1 + dv/c)`` (``Spectrum.interpolate``) and one built with
    ``dv/c`` (:func:`log_lambda_grid`) differ by ``dv/2c`` relative.

    Parameters
    ----------
    wave : ndarray
        Wavelength grid (Angstrom).
    rtol : float
        Allowed relative spread of the ln(lambda) step.

    Returns
    -------
    float
        ``ln(wave[-1] / wave[0]) / (N - 1)`` (dimensionless): the mean step,
        which -- unlike a single difference -- carries no round-off from the
        logarithms of neighbouring nodes.

    Raises
    ------
    ValueError
        If ``wave`` is not strictly increasing or not uniform in ln(lambda).
    """
    wave = np.asarray(wave, float)
    dln = np.diff(np.log(wave))
    if dln.size == 0 or np.any(dln <= 0):
        raise ValueError("wavelength grid must be strictly increasing")
    step = np.median(dln)
    if np.ptp(dln) > rtol * step:
        raise ValueError(
            f"wavelength grid is not uniform in ln(lambda): step varies by "
            f"{np.ptp(dln) / step:.2g} (relative), tolerance {rtol:g}. "
            "Resample it onto a log-lambda grid first.")
    return float(np.log(wave[-1] / wave[0]) / dln.size)


def log_grid_step(wave, rtol=1e-3):
    """Velocity step (km/s) of a uniform log-lambda grid, after verifying it.

    Parameters
    ----------
    wave : ndarray
        Wavelength grid (Angstrom).
    rtol : float
        Allowed relative spread of the ln(lambda) step.

    Returns
    -------
    float
        ``c * log_grid_dln(wave)`` (km/s), see :func:`log_grid_dln`.

    Raises
    ------
    ValueError
        If ``wave`` is not strictly increasing or not uniform in ln(lambda).
    """
    return C_KMS * log_grid_dln(wave, rtol)


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

    Parameters
    ----------
    n : int
        Window length (pixels).
    alpha : float
        Tapered fraction of the window (0 = rectangular, 1 = Hann).

    Returns
    -------
    ndarray
        (n,) window, 1 in the flat part and 0 at the ends.
    """
    return _tukey(n, alpha)


# ln(lambda_obs / lambda_rest) as a function of beta = v/c, and its inverse,
# per convention. Every Doppler shift in unicor and todcor goes through this
# table (via shift_kms / doppler_factor / velocity_from_log_shift).
_LOG_DOPPLER = {
    'cz': (np.log1p, np.expm1),               # 1 + beta
    'relativistic': (np.arctanh, np.tanh),    # sqrt((1 + beta) / (1 - beta))
}


def _log_doppler(convention):
    """The (beta -> ln factor, ln factor -> beta) pair of ``convention``."""
    try:
        return _LOG_DOPPLER[convention]
    except (KeyError, TypeError):
        raise ValueError(f"velocity convention must be one of {tuple(_LOG_DOPPLER)}, "
                         f"got {convention!r}") from None


def shift_kms(v, convention='cz'):
    """Log-lambda shift ``s = c * ln(lambda_obs / lambda_rest)`` (km/s) for velocity ``v``.

    The translation distance along the velocity coordinate
    ``u = c * ln(lambda / lambda_0)`` of a uniform log-lambda grid: a line at
    rest wavelength ``lambda`` moves to ``lambda * doppler_factor(v)``, i.e. by
    ``s`` in ``u``. ``s = c * ln(1 + v/c)`` under 'cz' and ``c * artanh(v/c)``
    (the rapidity) under 'relativistic'; computed with ``log1p`` /
    ``arctanh``, so it keeps full precision for small ``v``.

    Parameters
    ----------
    v : float or ndarray
        Radial velocity (km/s); positive red-shifts.
    convention : {'cz', 'relativistic'}
        Doppler convention, see :func:`doppler_factor`.

    Returns
    -------
    float or ndarray
        The shift ``s`` (km/s), same shape as ``v``.

    Raises
    ------
    ValueError
        If ``convention`` is unknown, or ``v`` has no Doppler factor under it
        (``v <= -c`` for 'cz', ``|v| >= c`` for 'relativistic').
    """
    forward, _ = _log_doppler(convention)
    beta = np.asarray(v, float) / C_KMS
    with np.errstate(divide='ignore', invalid='ignore'):
        s = C_KMS * forward(beta)
    # A NaN velocity (e.g. a failed RV) propagates as NaN, as numpy would.
    if np.any(np.isfinite(beta) & ~np.isfinite(s)):
        raise ValueError(
            f"velocity at or beyond the speed of light has no {convention!r} Doppler "
            f"factor: need {'v > -c' if convention == 'cz' else '|v| < c'} (km/s)")
    return s


def doppler_factor(v, convention='cz'):
    """Wavelength ratio ``lambda_obs / lambda_rest`` for radial velocity ``v``.

    Two conventions are supported:

    'cz' (the default)
        ``lambda_obs / lambda_rest = 1 + v/c``, i.e. ``v = c z``. The
        radial-velocity measure of the IAU (2000) resolution (Lindegren &
        Dravins 2003, A&A 401, 1185) and the velocity of IRAF ``fxcor``: the
        measured shift expressed as a velocity, with no attempt to separate
        the transverse-Doppler, gravitational or convective contributions.
    'relativistic'
        ``lambda_obs / lambda_rest = sqrt((1 + v/c) / (1 - v/c))``: the special-
        relativistic Doppler factor of a source receding along the line of
        sight at speed ``v``.

    The two agree to first order; for the same spectrum the 'cz' velocity
    exceeds the 'relativistic' one by ``~v**2 / 2c`` (37.5 m/s at 150 km/s).

    Parameters
    ----------
    v : float or ndarray
        Radial velocity (km/s); positive red-shifts.
    convention : {'cz', 'relativistic'}
        Doppler convention.

    Returns
    -------
    float or ndarray
        The (dimensionless) factor, same shape as ``v``.

    Raises
    ------
    ValueError
        As :func:`shift_kms`.
    """
    return np.exp(shift_kms(v, convention) / C_KMS)


def velocity_from_log_shift(dln, convention='cz'):
    """Radial velocity (km/s) of a shift ``dln = ln(lambda_obs / lambda_rest)``.

    The inverse of :func:`shift_kms` (``dln = s / c``). On a uniform
    log-lambda grid of step ``Delta``, a shift of ``k`` pixels is
    ``dln = k * Delta``, and its velocity is

    - 'cz': ``c * expm1(dln)``, where ``expm1(x) = exp(x) - 1`` computed
      without the cancellation that ruins ``exp(x) - 1`` for tiny ``x``;
    - 'relativistic': ``c * tanh(dln)``.

    Reading the shift as ``c * dln`` instead (i.e. as ``k * dv``) is off by
    ``-v**2 / 2c`` from 'cz' and by ``+v**3 / 3c**2`` from 'relativistic'.

    Parameters
    ----------
    dln : float or ndarray
        Natural-log wavelength shift (dimensionless).
    convention : {'cz', 'relativistic'}
        Doppler convention, see :func:`doppler_factor`.

    Returns
    -------
    float or ndarray
        Radial velocity (km/s), same shape as ``dln``.

    Raises
    ------
    ValueError
        If ``convention`` is unknown.
    """
    _, inverse = _log_doppler(convention)
    return C_KMS * inverse(np.asarray(dln, float))


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

    Parameters
    ----------
    n : int
        Length of the real signal (pixels).
    sigma : float
        Per-pixel noise (flux units).

    Returns
    -------
    w : ndarray
        Weights (1 / flux units^2) aligned with :func:`rfft_positive`
        output, i.e. DC dropped.
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

    Parameters
    ----------
    residual : array_like
        Flux minus continuum, on a uniform velocity (log-lambda) grid.
    dv : float
        Velocity step of the grid (km/s).
    vsini : float
        Projected rotational velocity (km/s); <= 0 returns the input.
    epsilon : float
        Linear limb-darkening coefficient (0-1).

    Returns
    -------
    ndarray
        The broadened residual, same length (``numpy.convolve`` 'same').
    """
    if vsini <= 0:
        return np.asarray(residual, float)
    kernel = rotational_kernel(dv, vsini, epsilon)
    return np.convolve(np.asarray(residual, float), kernel, mode="same")


def broaden_gaussian_residual(residual, dv, fwhm_kms):
    """Gaussian-broaden a *residual* flux array (continuum = 0) by ``fwhm_kms``.

    For resolving power ``R``, use ``fwhm_kms = c / R``. See
    :func:`broaden_rotational_residual` on the residual vs flux forms.

    Parameters
    ----------
    residual : array_like
        Flux minus continuum, on a uniform velocity (log-lambda) grid.
    dv : float
        Velocity step of the grid (km/s).
    fwhm_kms : float
        Kernel FWHM (km/s); <= 0 returns the input.

    Returns
    -------
    ndarray
        The broadened residual, zero-padded beyond the ends.
    """
    if fwhm_kms <= 0:
        return np.asarray(residual, float)
    sigma_pix = (fwhm_kms * _FWHM_TO_SIGMA) / dv
    return gaussian_filter1d(np.asarray(residual, float), sigma_pix,
                             mode="constant", cval=0.0)


def broaden_rotational(flux, dv, vsini, epsilon=0.6):
    """Rotationally broaden normalized flux (continuum = 1) by ``vsini`` km/s.

    ``1 + broaden_rotational_residual(flux - 1, ...)``: continuum 1 is
    assumed beyond the ends.

    Parameters
    ----------
    flux : array_like
        Normalized flux on a uniform velocity (log-lambda) grid.
    dv : float
        Velocity step of the grid (km/s).
    vsini : float
        Projected rotational velocity (km/s).
    epsilon : float
        Linear limb-darkening coefficient (0-1).

    Returns
    -------
    ndarray
        The broadened flux.
    """
    return 1.0 + broaden_rotational_residual(
        np.asarray(flux, float) - 1.0, dv, vsini, epsilon)


def broaden_gaussian(flux, dv, fwhm_kms):
    """Gaussian-broaden normalized flux by ``fwhm_kms`` km/s (instrumental).

    ``1 + broaden_gaussian_residual(flux - 1, ...)``: continuum 1 is assumed
    beyond the ends. For resolving power ``R``, use ``fwhm_kms = c / R``.

    Parameters
    ----------
    flux : array_like
        Normalized flux on a uniform velocity (log-lambda) grid.
    dv : float
        Velocity step of the grid (km/s).
    fwhm_kms : float
        Kernel FWHM (km/s).

    Returns
    -------
    ndarray
        The broadened flux.
    """
    return 1.0 + broaden_gaussian_residual(
        np.asarray(flux, float) - 1.0, dv, fwhm_kms)


def doppler_shift_wave(wave, v, convention='cz'):
    """Doppler-shift a wavelength array by radial velocity ``v``.

    Parameters
    ----------
    wave : ndarray
        Rest-frame wavelengths (Angstrom).
    v : float
        Radial velocity (km/s); positive red-shifts (increases lambda).
    convention : {'cz', 'relativistic'}
        Doppler convention, see :func:`doppler_factor`; under 'cz' this is
        ``wave * (1 + v/c)``.

    Returns
    -------
    ndarray
        Observed-frame wavelengths (Angstrom).
    """
    return np.asarray(wave, float) * doppler_factor(v, convention)


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
        the grids are heavily oversampled. ``'cubic'`` uses a (not-a-knot)
        cubic spline; prefer it when a spectrum is shifted by many pixels and the
        interpolation error would otherwise show up as lost correlation.

    Returns
    -------
    ndarray
        The flux at ``wave_grid``.

    Raises
    ------
    ValueError
        If ``wave`` is not strictly increasing, or ``kind`` is unknown.
    """
    wave = np.asarray(wave, float)
    flux = np.asarray(flux, float)
    # np.interp silently returns garbage on a decreasing or repeated axis.
    if np.any(np.diff(wave) <= 0):
        raise ValueError("source wavelengths must be strictly increasing")

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
    """Mean-subtract and L2-normalize a flux vector.

    Parameters
    ----------
    flux : array_like
        1-D flux.

    Returns
    -------
    ndarray
        ``(flux - mean) / ||flux - mean||``; just mean-subtracted if that
        norm is zero.
    """
    f = np.asarray(flux, float)
    f = f - f.mean()
    norm = np.sqrt((f ** 2).sum())
    return f / norm if norm > 0 else f


def zero_lag_correlation(a, b):
    """Unshifted (zero-lag) correlation: the inner product of two vectors.

    On L2-normalized inputs this is the cosine similarity used by USuRPER; no
    Fourier transform or velocity scan is involved.

    Parameters
    ----------
    a, b : array_like
        Vectors of equal length.

    Returns
    -------
    float
        ``sum(a * b)``.
    """
    return float(np.sum(np.asarray(a, float) * np.asarray(b, float)))

