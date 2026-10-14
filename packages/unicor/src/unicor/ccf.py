"""One-dimensional cross-correlation: CCF, multi-order combination, RV + error.

The array-level UNICOR engine. Every function takes plain floats / numpy
arrays -- velocities in km/s, the CCF dimensionless -- so it can be used
without SPARTA's ``Spectrum`` / ``Template`` objects (which wrap it in
:class:`sparta.ccf1d.CCF1d`).

Typical use, for one order already on a uniform log-lambda grid ``wave``::

    K = round(vmax / dv)                                  # lags on each side
    template_flux = resample(t_wave, t_flux, extend_log_grid(wave, K))
    v, c = ccf_vs_velocity(template_flux, observed_flux, wave)
    rv, peak = subpixel_peak(v, c)
    erv = rv_uncertainty(v, c, rv, n_eff)

The template must cover ``wave`` extended by K steps on each side, so that every
lag correlates all the observed pixels (see :func:`ccf_vs_velocity`).

Lag k shifts by ``k * Delta`` in ln(lambda), ``Delta`` the grid's actual
ln-step, and its velocity follows the Doppler convention
(``velocity_convention``, 'cz' by default; see :func:`unicor.dsp.doppler_factor`).
The velocity axis is therefore not exactly uniform -- under 'cz' its step grows
by ``Delta`` (relative) per lag -- and every function below takes it as given
(none assumes equal spacing).

For several orders sharing the velocity lags ``v``::

    combined = combine_ccfs(ccf_matrix, n_eff=n_eff_per_order)  # Zucker (2003)
    rv, peak = subpixel_peak(v, combined)
    erv = rv_uncertainty(v, combined, rv, n_eff_per_order.sum())

``n_eff`` is the number of *independent* spectral bins: the number of pixels
on the native (detector) sampling, not on a finer resampled grid -- and not the
number of velocity lags (:func:`effective_bins`).

Per-order atoms
---------------
The same steps, one call per order -- the loop body of SPARTA's ``CCF1d``::

    dln, v = lag_axis(waves, dv, K)                     # shared lag velocities
    orders = [ccf_order(w, f, t_wave, t_flux, K, dv_kms=dv, native_step_kms=nat)
              for w, f, nat in zip(waves, fluxes, native_steps)]
    combined = combine_ccfs([o.ccf for o in orders], n_eff=[o.n_eff for o in orders])
    rv = measure_rv(v, combined, sum(o.n_eff for o in orders))

:func:`ccf_order` returns a :class:`CCFOrder`, :func:`measure_rv` an
:class:`RVMeasurement`; :func:`template_coverage` says which observed pixels
the template serves at every lag.

Two output modes
----------------
:func:`ccf_vs_velocity` and :func:`combine_ccfs` take ``output='ccf'`` (the
default: the correlation itself) or ``output='loglike'``: the Zucker (2003)
log-likelihood ``log L = -(N/2) log(1 - C**2)`` (:func:`loglike_from_ccf`),
relative to no correlation (C = 0). It is monotonic in ``C**2`` at fixed N, so
both curves peak at the same lag; read the RV and its error off the CCF
(:func:`subpixel_peak`, :func:`rv_uncertainty`) whichever curve you keep.
"""
from dataclasses import dataclass

import numpy as np
from scipy import interpolate
from scipy.signal import correlate as _correlate

from .dsp import C_KMS, log_grid_dln, resample, velocity_from_log_shift

OUTPUTS = ('ccf', 'loglike')


def check_output(output):
    """Validate an ``output`` mode: 'ccf' (the correlation) or 'loglike'.

    Parameters
    ----------
    output : str
        The requested mode.

    Raises
    ------
    ValueError
        If ``output`` is not one of ``OUTPUTS``.
    """
    if output not in OUTPUTS:
        raise ValueError(f"output must be one of {OUTPUTS}, got {output!r}")


def extend_log_grid(wave, n_lags):
    """A uniform log-lambda grid continued by ``n_lags`` steps on each side.

    The template for :func:`ccf_vs_velocity` must be sampled on this grid: its
    central ``len(wave)`` nodes are ``wave`` itself, so lag k is exactly k grid
    steps, with no sub-pixel phase offset between template and observation.

    Parameters
    ----------
    wave : ndarray
        Observed wavelength grid (Angstrom), uniform in ln(lambda).
    n_lags : int
        Number of velocity lags K (>= 1) to scan on each side of zero; the
        scan half-width is ``K * dv`` km/s.

    Returns
    -------
    ndarray
        ``wave[0] * exp(dln * arange(-K, N + K))`` (Angstrom), length
        ``N + 2K``, with ``dln`` the mean ln(lambda) step of ``wave``.

    Raises
    ------
    ValueError
        If ``n_lags`` is not a positive integer, or ``wave`` is not a strictly
        increasing grid uniform in ln(lambda).
    """
    if int(n_lags) != n_lags or n_lags < 1:
        raise ValueError(f"n_lags must be a positive integer, got {n_lags}")
    wave = np.asarray(wave, float)
    dln = log_grid_dln(wave)
    n, k = wave.size, int(n_lags)
    return wave[0] * np.exp(dln * np.arange(-k, n + k))


def ccf_vs_velocity(template, signal, wave, velocity_convention='cz', output='ccf',
                    n_eff=None):
    """Pearson cross-correlation of ``signal`` against a template, vs velocity.

    ``signal`` holds N pixels on the uniform log-lambda grid ``wave``;
    ``template`` holds N + 2K pixels of the same grid continued by K steps on
    each side (:func:`extend_log_grid`). At lag k the N observed pixels are
    correlated with the N-pixel template segment shifted by k, each
    mean-subtracted and normalised by its own norm -- so every lag is a full
    Pearson correlation over the same N pixels.

    Lag k multiplies wavelength by ``exp(k * Delta)``, ``Delta`` the actual
    ln(lambda) step of ``wave``, so its velocity is
    ``velocity_from_log_shift(k * Delta, velocity_convention)``:
    ``c * expm1(k * Delta)`` under 'cz', ``c * tanh(k * Delta)`` under
    'relativistic'. The step is measured from ``wave`` rather than taken from
    a nominal ``dv``, because grids built as ``ln(1 + dv/c)`` and as ``dv/c``
    differ by ``dv/2c`` relative.

    A template on the observed span alone (correlated in 'full' mode with one
    global norm) overlaps only N - |k| pixels at lag k: the CCF is tapered by
    about ``1 - |k|/N`` and its peak is pulled toward v = 0, by tens to
    hundreds of m/s per order for rotating stars, the same sign in every
    order. The extended template removes that pull, and makes the number of
    bins behind every lag the constant N that :func:`rv_uncertainty` assumes.

    Parameters
    ----------
    template : ndarray, shape (N + 2K,)
        Template flux on the observed grid extended by K >= 1 steps per side.
    signal : ndarray, shape (N,)
        Observed flux on ``wave``.
    wave : ndarray, shape (N,)
        The signal's wavelength grid (Angstrom), uniform in ln(lambda).
    velocity_convention : {'cz', 'relativistic'}
        Doppler convention mapping lags to velocities, see
        :func:`unicor.dsp.doppler_factor`. Default 'cz'.
    output : {'ccf', 'loglike'}
        Return the correlation (default) or the Zucker (2003)
        log-likelihood ``-(n_eff/2) log(1 - C**2)`` (:func:`loglike_from_ccf`).
    n_eff : float, optional
        Effective number of independent spectral bins behind the signal
        (dimensionless, > 0; native pixels, see :func:`rv_uncertainty`).
        Required for ``output='loglike'``, unused otherwise.

    Returns
    -------
    v : ndarray, shape (2K + 1,)
        Velocity of each lag k = -K..K (km/s), increasing, 0 at k = 0.
    ccf : ndarray, shape (2K + 1,)
        Pearson correlation in [-1, 1] (``output='ccf'``), or its
        log-likelihood (``output='loglike'``, dimensionless, >= 0). Peaks at
        the velocity by which ``signal`` is red-shifted relative to
        ``template``.

    Raises
    ------
    ValueError
        If ``output`` is unknown, or 'loglike' without ``n_eff``;
        if ``len(template) - len(signal)`` is not a positive even number 2K;
        if ``wave`` is not an N-node grid uniform in ln(lambda); if
        ``velocity_convention`` is unknown; if either flux array is not finite;
        or if ``signal`` or the template segment of any lag is constant (the
        correlation is then 0/0 -- typically a template without lines there).
    """
    check_output(output)
    if output == 'loglike' and n_eff is None:
        raise ValueError("output='loglike' needs n_eff, the number of independent "
                         "spectral bins behind the signal")
    t = np.asarray(template, float)
    s = np.asarray(signal, float)
    if t.ndim != 1 or s.ndim != 1:
        raise ValueError("template and signal must be one-dimensional")
    n = s.size
    excess = t.size - n
    if excess < 2 or excess % 2:
        raise ValueError(
            f"template must extend the signal grid by K >= 1 pixels on each side "
            f"(length N + 2K); got {t.size} template vs {n} signal pixels. "
            "Resample the template onto extend_log_grid(wave, K).")
    k = excess // 2
    wave = np.asarray(wave, float)
    if wave.shape != s.shape:
        raise ValueError(f"wave must be the signal's wavelength grid ({n} nodes), "
                         f"got shape {wave.shape}")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(s))):
        raise ValueError("template and signal must be finite (mask NaN pixels first)")

    s = s - s.mean()
    s_norm2 = (s ** 2).sum()
    if s_norm2 == 0:
        raise ValueError("signal is constant: there are no lines to correlate")
    v = velocity_from_log_shift(np.arange(-k, k + 1) * log_grid_dln(wave),
                                velocity_convention)

    # Subtracting the global mean leaves each segment's Pearson coefficient
    # unchanged and keeps the running sums small, so the segment variance
    # below does not cancel catastrophically for un-normalised flux.
    t = t - t.mean()
    # out[m] = sum_i t[i + m] s[i], m = 0..2K; s has zero mean, so the
    # segment mean drops out of the numerator.
    numer = _correlate(t, s, mode="valid", method="fft")
    s1 = np.concatenate(([0.0], np.cumsum(t)))
    s2 = np.concatenate(([0.0], np.cumsum(t ** 2)))
    sum_t = s1[n:] - s1[:-n]
    sum_t2 = s2[n:] - s2[:-n]
    seg_var = sum_t2 - sum_t ** 2 / n
    # A flat segment leaves only the cumsum round-off (~1e-16 of sum_t2).
    flat = seg_var <= 1e-10 * sum_t2
    if np.any(flat):
        raise ValueError(
            f"the template is constant over the observed span at {flat.sum()} of "
            f"{flat.size} lags; the correlation is undefined there (narrow the "
            "velocity scan or use a template with lines across the order)")
    ccf = numer / np.sqrt(seg_var * s_norm2)
    # Segment m is the template shifted by K - m pixels: reverse to ascending lag.
    ccf = ccf[::-1]
    if output == 'loglike':
        return v, loglike_from_ccf(ccf, n_eff)
    return v, ccf


def loglike_from_ccf(ccf, n_eff):
    """Zucker (2003) log-likelihood of a CCF: ``-(N/2) log(1 - C**2)``.

    For white noise on N independent bins and a template matched up to an
    amplitude, maximising the likelihood over that amplitude and the noise
    level leaves ``log L(v) = -(N/2) log(1 - C(v)**2)`` plus a constant
    (Zucker 2003, MNRAS 342, 1291, §2), taken here as zero where C = 0. It
    rises monotonically with ``C**2``, so it peaks where ``|C|`` does, and its
    curvature at the peak is the one :func:`rv_uncertainty` uses.

    Parameters
    ----------
    ccf : ndarray, shape (n_lags,) or (n_orders, n_lags)
        Normalized CCF(s); clipped to [-1, 1] first.
    n_eff : float or array_like, shape (n_orders,)
        Effective number of independent spectral bins (dimensionless, > 0):
        a scalar, or one per order (row) of ``ccf``.

    Returns
    -------
    ndarray, shape of ``ccf``
        The log-likelihood (dimensionless, >= 0); +inf where ``|C| = 1``
        (a perfect, noiseless match).

    Raises
    ------
    ValueError
        If ``n_eff`` is not finite and positive, or does not match the rows.
    """
    C = np.clip(np.asarray(ccf, float), -1.0, 1.0)
    n_eff = np.asarray(n_eff, float)
    if not np.all(np.isfinite(n_eff) & (n_eff > 0)):
        raise ValueError(f"n_eff must be finite and positive, got {n_eff}")
    if n_eff.ndim == 1:
        if C.ndim != 2 or n_eff.shape != C.shape[:1]:
            raise ValueError(f"n_eff must hold one value per row of ccf ({C.shape}), "
                             f"got shape {n_eff.shape}")
        n_eff = n_eff[:, None]
    elif n_eff.ndim != 0:
        raise ValueError(f"n_eff must be a scalar or one value per order, got shape {n_eff.shape}")
    with np.errstate(divide='ignore'):
        return -0.5 * n_eff * np.log1p(-C ** 2)


def combine_ccfs(ccfs, n_eff=None, output='ccf'):
    """Combine per-order CCFs into one, by maximum likelihood (Zucker 2003, §3.1).

    Order ``i`` with ``N_i`` independent bins contributes
    ``log L_i = -N_i/2 log(1 - C_i**2)``; the combined CCF is defined so that
    ``-N_tot/2 log(1 - ML**2)`` is the summed log-likelihood, i.e.

        ML = sqrt(1 - prod_i (1 - C_i**2) ** (N_i / N_tot)),  N_tot = sum_i N_i.

    With equal ``N_i`` this is Zucker's ``sqrt(1 - prod(1 - C_i**2) ** (1/M))``.

    Parameters
    ----------
    ccfs : ndarray, shape (n_orders, n_lags)
        Per-order normalized CCFs on a common velocity-lag axis.
    n_eff : array_like, shape (n_orders,), optional
        Effective number of independent spectral bins of each order
        (dimensionless, > 0). None weights all orders equally (``output='ccf'``
        only).
    output : {'ccf', 'loglike'}
        Return the ML-combined CCF (default) or the summed log-likelihood
        ``sum_i -(N_i/2) log(1 - C_i**2) = -(N_tot/2) log(1 - ML**2)``.

    Returns
    -------
    combined : ndarray, shape (n_lags,)
        The ML-combined CCF, in [0, 1] -- pair it with ``N_tot = sum(n_eff)``
        in :func:`rv_uncertainty` -- or the summed log-likelihood
        (dimensionless, >= 0).

    Raises
    ------
    ValueError
        If ``output`` is unknown, ``n_eff`` does not hold one finite, positive
        value per order, or is None with ``output='loglike'``.
    """
    check_output(output)
    # Clip to [-1, 1] so a per-order value that overshoots numerically cannot
    # drive 1 - C**2 negative (which would make the combined CCF NaN).
    C = np.clip(np.atleast_2d(np.asarray(ccfs, float)), -1.0, 1.0)
    n_orders = C.shape[0]
    if n_eff is None:
        weights = np.full(n_orders, 1.0 / n_orders)
    else:
        n_eff = np.asarray(n_eff, float)
        if n_eff.shape != (n_orders,):
            raise ValueError(f"n_eff must have one entry per order ({n_orders}), "
                             f"got shape {n_eff.shape}")
        if not np.all(np.isfinite(n_eff) & (n_eff > 0)):
            raise ValueError(f"n_eff must be finite and positive, got {n_eff}")
        weights = n_eff / n_eff.sum()
    if output == 'loglike':
        if n_eff is None:
            raise ValueError("output='loglike' needs n_eff: the log-likelihood "
                             "scales with each order's number of independent bins")
        return loglike_from_ccf(C, n_eff).sum(axis=0)
    return np.sqrt(1 - np.prod((1 - C ** 2) ** weights[:, None], axis=0))


def subpixel_peak(v, ccf, npts=5):
    """Sub-pixel CCF peak from a least-squares parabola around the maximum.

    Parameters
    ----------
    v : ndarray
        Velocity lags (km/s), increasing; need not be uniform.
    ccf : ndarray
        CCF values at ``v``.
    npts : int
        Odd number (>= 3) of points around the argmax used in the fit.

    Returns
    -------
    v_peak : float
        Velocity of the parabola vertex (km/s).
    ccf_peak : float
        CCF value at the vertex.

    Raises
    ------
    ValueError
        If the maximum is within ``npts // 2`` lags of either end of ``v`` (the
        peak may lie outside the scan), the fitted parabola is not concave, or
        the CCF is not finite.
    """
    if npts < 3:
        raise ValueError("npts must be at least 3")
    if npts % 2 != 1:
        raise ValueError("npts must be odd, to centre the fit on the peak")
    v = np.asarray(v, float)
    ccf = np.asarray(ccf, float)

    if not np.all(np.isfinite(ccf)):
        raise ValueError("CCF contains NaN/inf")
    x_n = int(np.argmax(ccf))
    half = npts // 2
    # Near the ends the window would wrap (negative indices) or overrun; the
    # true maximum may lie outside the scanned range, so there is no peak.
    if x_n < half or x_n > len(ccf) - 1 - half:
        raise ValueError("CCF peak is at the edge of the velocity range; "
                         "widen the scan")
    # Fit about the central node: in absolute velocity the columns 1, x, x^2
    # are nearly collinear far from v = 0 (x ~ 38 km/s on 0.05 km/s steps lost
    # ~0.2 m/s to round-off). Any spacing works, uniform or not.
    x0 = v[x_n]
    dx = v[x_n - half: x_n + half + 1] - x0
    y = ccf[x_n - half: x_n + half + 1]

    design = np.column_stack([np.ones_like(dx), dx, dx ** 2])
    pars, *_ = np.linalg.lstsq(design, y, rcond=None)
    if not pars[2] < 0:
        raise ValueError("CCF is not concave around its maximum")
    ccf_peak = pars[0] - pars[1] * pars[1] / (4 * pars[2])
    v_peak = x0 - pars[1] / (2 * pars[2])
    return v_peak, ccf_peak


def ccf_at(v, ccf, v0):
    """CCF value at velocity ``v0`` by 3-point (quadratic) Lagrange interpolation.

    Parameters
    ----------
    v : ndarray
        Velocity lags (km/s).
    ccf : ndarray
        CCF values at ``v``.
    v0 : float
        Velocity (km/s) at which to evaluate.

    Returns
    -------
    float
        Interpolated CCF value.

    Raises
    ------
    ValueError
        If ``v0`` lies outside ``[v[0], v[-1]]``.
    """
    v = np.asarray(v, float)
    ccf = np.asarray(ccf, float)
    if not v[0] <= v0 <= v[-1]:
        raise ValueError(f"v0 = {v0} km/s is outside the velocity range")
    # Centre the 3-point stencil on the nearest lag, kept inside the array.
    x_n = int(np.clip(np.argmin(np.abs(v - v0)), 1, len(v) - 2))
    x = v[x_n - 1: x_n + 2]
    y = ccf[x_n - 1: x_n + 2]

    return (y[0] * (v0 - x[1]) * (v0 - x[2]) / (x[0] - x[1]) / (x[0] - x[2]) +
            y[1] * (v0 - x[0]) * (v0 - x[2]) / (x[1] - x[0]) / (x[1] - x[2]) +
            y[2] * (v0 - x[0]) * (v0 - x[1]) / (x[2] - x[0]) / (x[2] - x[1]))


def rv_uncertainty(v, ccf, v_peak, n_eff):
    """Maximum-likelihood RV uncertainty from the CCF curvature (Zucker 2003, §2.3).

    ``sigma**2 = -[N * C''(v) C(v) / (1 - C(v)**2)]**-1`` at the peak, with
    ``C''`` from an interpolating quartic spline. ``N`` is the number of
    independent spectral bins behind ``ccf`` -- for a multi-order CCF combined
    by :func:`combine_ccfs`, the total over orders. It is a property of the
    spectrum, so the result does not depend on the velocity range scanned.

    Parameters
    ----------
    v : ndarray
        Velocity lags (km/s).
    ccf : ndarray
        CCF values at ``v`` (single-order or combined).
    v_peak : float
        Velocity of the CCF maximum (km/s).
    n_eff : float
        Effective number of independent spectral bins (dimensionless, > 0):
        native pixels, not pixels of a finer resampled grid.

    Returns
    -------
    float
        1-sigma RV uncertainty (km/s); NaN when ``v_peak`` is not a true
        maximum of the spline (non-negative curvature).

    Raises
    ------
    ValueError
        If ``ccf`` is not finite or ``n_eff`` is not finite and positive.
    """
    v = np.asarray(v, float)
    if not np.all(np.isfinite(ccf)):
        raise ValueError("CCF contains NaN/inf")
    if not (np.isfinite(n_eff) and n_eff > 0):
        raise ValueError(f"n_eff must be finite and positive, got {n_eff}")

    spl = interpolate.UnivariateSpline(v, ccf, k=4, s=0)
    spl_vv = spl.derivative(n=2)

    # The variance is -1 / (ml_factor * n_eff) and is only real at a true CCF
    # maximum (ml_factor < 0); guard the degenerate cases rather than emitting
    # a silent NaN from sqrt().
    ml_factor = spl_vv(v_peak) * spl(v_peak) / (1 - spl(v_peak) ** 2)
    denom = ml_factor * n_eff
    var = -1.0 / denom if denom != 0 else np.nan
    return float(np.sqrt(var)) if (np.isfinite(var) and var > 0) else np.nan


def bisector_span(v, ccf, n_eff, levels=(0.35, 0.95)):
    """Bisector span (BIS) of a single-peaked CCF and its uncertainty.

    .. warning:: This formula is known to be incorrect and is under review:
       the bisector midpoints are built from the CCF *values* at the level
       crossings (``ccf_at`` at each root), not from the crossing velocities,
       so the result is not a velocity span. It is pinned, unchanged, for a
       later fix; do not use it scientifically.

    Parameters
    ----------
    v : ndarray
        Velocity lags (km/s).
    ccf : ndarray
        CCF values at ``v``.
    n_eff : float
        Effective number of independent spectral bins behind ``ccf`` (see
        :func:`rv_uncertainty`); sets the uncertainty.
    levels : (float, float)
        Lower and upper bisector levels, as fractions of the CCF maximum.

    Returns
    -------
    bis, ebis : float
        The span as currently computed (see the warning: not yet a velocity)
        and its uncertainty (km/s); NaN when either level does not cut the
        CCF exactly twice.
    """
    v = np.asarray(v, float)
    ccf = np.asarray(ccf, float)
    y_low = np.max(ccf) * levels[0]
    y_high = np.max(ccf) * levels[1]

    roots_low = interpolate.sproot(interpolate.splrep(v, ccf - y_low), mest=2)
    roots_high = interpolate.sproot(interpolate.splrep(v, ccf - y_high), mest=2)

    if len(roots_low) != 2 or len(roots_high) != 2:
        return np.nan, np.nan

    low = [ccf_at(v, ccf, roots_low[0]), ccf_at(v, ccf, roots_low[1])]
    high = [ccf_at(v, ccf, roots_high[0]), ccf_at(v, ccf, roots_high[1])]

    try:
        v_peak, _ = subpixel_peak(v, ccf)
        err = rv_uncertainty(v, ccf, v_peak, n_eff)
    except ValueError:
        # The bisector is defined, but its error needs a located peak.
        err = np.nan

    bis = (low[1] + low[0]) / 2 - (high[1] + high[0]) / 2
    return bis, np.sqrt(2) * err


# =============================================================================
# Per-order atoms
# =============================================================================
@dataclass(frozen=True, eq=False)
class CCFOrder:
    """
    One order's cross-correlation with a template (:func:`ccf_order`).

    Parameters
    ----------
    velocity : ndarray, shape (2K + 1,)
        Lag velocities (km/s) on the order's own correlated grid, increasing,
        0 at the central lag.
    ccf : ndarray, shape (2K + 1,)
        Pearson correlation at each lag (dimensionless, in [-1, 1]).
    loglike : ndarray, shape (2K + 1,)
        Zucker (2003) log-likelihood ``-(n_eff/2) ln(1 - ccf**2)``
        (dimensionless, >= 0; :func:`loglike_from_ccf`).
    rv : float
        Velocity of the sub-pixel CCF peak (km/s); NaN if the peak cannot be
        located (e.g. at the edge of the scan).
    erv : float
        ML uncertainty of ``rv`` (km/s); NaN unless requested, or if ``rv``
        is NaN or not a true maximum.
    peak : float
        CCF value at the peak (dimensionless); NaN with ``rv``.
    n_eff : float
        Effective number of independent spectral bins behind ``ccf``
        (dimensionless, > 0; :func:`effective_bins` of the correlated pixels).
    lo, hi : int
        The correlated observed pixels, ``wave[lo:hi]``
        (:func:`template_coverage`).
    """
    velocity: np.ndarray
    ccf: np.ndarray
    loglike: np.ndarray
    rv: float
    erv: float
    peak: float
    n_eff: float
    lo: int
    hi: int


@dataclass(frozen=True)
class RVMeasurement:
    """
    A radial velocity read off a CCF (:func:`measure_rv`).

    Parameters
    ----------
    rv : float
        Velocity of the sub-pixel CCF peak (km/s).
    erv : float
        Its ML uncertainty (km/s); NaN when ``rv`` is not a true maximum of
        the spline through the CCF.
    peak : float
        CCF value at the peak (dimensionless).
    """
    rv: float
    erv: float
    peak: float


def template_coverage(w_ext, t_wave, n_lags):
    """Observed pixels whose template is known at every lag of the scan.

    Parameters
    ----------
    w_ext : ndarray
        The order's log-lambda grid extended by ``n_lags`` steps on each side
        (Angstrom), see :func:`extend_log_grid`.
    t_wave : ndarray
        The template's wavelengths (Angstrom, increasing).
    n_lags : int
        Number of lags K on each side of zero.

    Returns
    -------
    lo, hi : int
        Observed pixels ``lo:hi`` need exactly the template nodes
        ``w_ext[lo: hi + 2K]``, all inside the template's span.

    Raises
    ------
    ValueError
        If no observed pixel is covered at every lag.
    """
    # A millionth of a step absorbs the round-off of exp() when the template
    # sits on the observed grid itself.
    tol = 1e-6 * np.log(w_ext[1] / w_ext[0])
    ln_ext = np.log(w_ext)
    inside = np.flatnonzero((ln_ext >= np.log(t_wave[0]) - tol) &
                            (ln_ext <= np.log(t_wave[-1]) + tol))
    n_obs = len(w_ext) - 2 * n_lags
    # The covered nodes are contiguous; pixel i needs nodes i .. i + 2K.
    lo, hi = (inside[0], inside[-1] - 2 * n_lags + 1) if inside.size else (0, 0)
    if hi <= lo:
        raise ValueError(
            f"the template ({t_wave[0]:.2f}-{t_wave[-1]:.2f} A) does not cover any "
            f"pixel of the order ({w_ext[n_lags]:.2f}-{w_ext[n_lags + n_obs - 1]:.2f} A) "
            f"across the +-{n_lags}-lag scan; use a wider template or a smaller vel_bound")
    return lo, hi


def lag_axis(waves, dv_kms, n_lags, velocity_convention='cz'):
    """The velocity of each lag, shared by orders on one log-lambda step.

    Lag k is ``k * Delta`` in ln(lambda) only if every order really is
    velocity-even with the declared step (dropping pixels breaks that
    silently), and the orders' CCFs share one velocity axis only if they share
    ``Delta``. Both are verified here.

    Parameters
    ----------
    waves : sequence of ndarray
        Each order's wavelength grid (Angstrom), uniform in ln(lambda).
    dv_kms : float
        The declared grid step (km/s); every order's measured step
        ``c * Delta_i`` must match it to a relative 1e-3.
    n_lags : int
        Number of lags K (>= 1) on each side of zero.
    velocity_convention : {'cz', 'relativistic'}
        Doppler convention mapping lags to velocities, see
        :func:`unicor.dsp.doppler_factor`. Default 'cz'.

    Returns
    -------
    dln : float
        The common ln(lambda) step ``Delta`` (dimensionless): the median of the
        orders' measured steps.
    v : ndarray, shape (2K + 1,)
        Velocity of lags k = -K..K (km/s), increasing, 0 at k = 0.

    Raises
    ------
    ValueError
        If ``n_lags`` is not a positive integer; an order is not uniform in
        ln(lambda) or its step is not ``dv_kms``; the orders' steps differ by
        more than a thousandth of a lag across the scan; or
        ``velocity_convention`` is unknown.
    """
    if int(n_lags) != n_lags or n_lags < 1:
        raise ValueError(f"n_lags must be a positive integer, got {n_lags}")
    dln = np.array([log_grid_dln(w) for w in waves])
    for i, step in enumerate(C_KMS * dln):
        if not np.isclose(step, dv_kms, rtol=1e-3):
            raise ValueError(
                f"order {i}: the wavelength grid step is {step:.6g} km/s but "
                f"{dv_kms:.6g} km/s is declared")
    dln_common = float(np.median(dln))
    # Tolerate a thousandth of a lag of misregistration at the scan edge.
    if np.max(np.abs(dln / dln_common - 1)) * n_lags > 1e-3:
        raise ValueError(
            "the orders' ln(lambda) steps differ (relative spread "
            f"{np.ptp(dln) / dln_common:.2g}), so their CCFs cannot share one "
            "velocity axis; resample them onto one log-lambda grid step")
    v = velocity_from_log_shift(np.arange(-n_lags, n_lags + 1) * dln_common,
                                velocity_convention)
    return dln_common, v


def effective_bins(n_pix, dv_kms, native_step_kms):
    """Independent spectral bins behind ``n_pix`` pixels on a ``dv_kms`` grid.

    Resampling onto a grid finer than the native (detector) sampling adds
    samples but no information, so ``n_pix * min(1, dv / native)`` bins are
    counted; a grid coarser than the native one keeps every pixel.

    Parameters
    ----------
    n_pix : int or array_like
        Pixels correlated (dimensionless, >= 0).
    dv_kms : float
        Step of the correlated grid (km/s, > 0).
    native_step_kms : float or array_like
        Native sampling step (km/s, > 0), the coarsest the data have had.

    Returns
    -------
    float or ndarray
        The effective number of bins (dimensionless), broadcast over the
        inputs.

    Raises
    ------
    ValueError
        If ``n_pix`` is negative, or a step is not finite and positive.
    """
    n_pix = np.asarray(n_pix, float)
    native_step_kms = np.asarray(native_step_kms, float)
    if np.any(n_pix < 0):
        raise ValueError(f"n_pix must be non-negative, got {n_pix}")
    if not (np.isfinite(dv_kms) and dv_kms > 0
            and np.all(np.isfinite(native_step_kms) & (native_step_kms > 0))):
        raise ValueError("dv_kms and native_step_kms must be finite and positive (km/s), "
                         f"got {dv_kms} and {native_step_kms}")
    return n_pix * np.minimum(1.0, dv_kms / native_step_kms)


def ccf_order(wave, flux, t_wave, t_flux, n_lags, *, dv_kms, native_step_kms,
              velocity_convention='cz', errors=False):
    """Cross-correlate one order with a template and read its RV.

    The template is resampled (linearly) onto the order's log-lambda grid
    continued by K = ``n_lags`` steps on each side (:func:`extend_log_grid`);
    only the observed pixels it covers at every lag are correlated
    (:func:`template_coverage`) -- it is never extrapolated -- and each lag
    is a Pearson correlation over those pixels (:func:`ccf_vs_velocity`). The
    RV is the sub-pixel peak (:func:`subpixel_peak`) on the velocity axis of
    the correlated pixels.

    Parameters
    ----------
    wave : ndarray, shape (N,)
        The order's wavelengths (Angstrom), uniform in ln(lambda) with step
        ``dv_kms``.
    flux : ndarray, shape (N,)
        The order's flux (finite).
    t_wave : ndarray
        Template wavelengths (Angstrom, strictly increasing), at rest.
    t_flux : ndarray
        Template flux at ``t_wave``.
    n_lags : int
        Number of lags K (>= 1) on each side of zero; the scan half-width is
        ``K * dv_kms`` km/s.
    dv_kms : float
        Step of the order's grid (km/s); with ``native_step_kms`` it sets
        ``n_eff`` (:func:`effective_bins`).
    native_step_kms : float
        The order's native sampling step (km/s).
    velocity_convention : {'cz', 'relativistic'}
        Doppler convention mapping lags to velocities. Default 'cz'.
    errors : bool
        Also compute the ML uncertainty of the RV (:func:`rv_uncertainty`);
        otherwise ``erv`` is NaN.

    Returns
    -------
    CCFOrder
        The CCF, its log-likelihood, RV, uncertainty, peak, ``n_eff`` and the
        correlated pixel range. ``rv``, ``erv`` and ``peak`` are NaN when the
        peak cannot be located (e.g. beyond the scan).

    Raises
    ------
    ValueError
        If the template covers no observed pixel at every lag or its
        wavelengths are not strictly increasing (one per flux value), the
        grid is not uniform in ln(lambda), a flux is not finite or constant,
        or ``velocity_convention`` is unknown (see :func:`ccf_vs_velocity`).
    """
    wave = np.asarray(wave, float)
    flux = np.asarray(flux, float)
    t_wave = np.asarray(t_wave, float)
    t_flux = np.asarray(t_flux, float)

    w_ext = extend_log_grid(wave, n_lags)
    t_ext = resample(t_wave, t_flux, w_ext)
    lo, hi = template_coverage(w_ext, t_wave, n_lags)
    v, ccf = ccf_vs_velocity(t_ext[lo: hi + 2 * n_lags], flux[lo:hi], wave[lo:hi],
                             velocity_convention)
    n_eff = effective_bins(hi - lo, dv_kms, native_step_kms)

    # A peak that cannot be located (e.g. at the scan edge) is a NaN RV, not
    # an error: the CCF itself is valid and still enters a combination.
    try:
        rv, peak = subpixel_peak(v, ccf)
        erv = rv_uncertainty(v, ccf, rv, n_eff) if errors else np.nan
    except (ValueError, np.linalg.LinAlgError):
        rv, erv, peak = np.nan, np.nan, np.nan

    return CCFOrder(velocity=v, ccf=ccf, loglike=loglike_from_ccf(ccf, n_eff), rv=rv,
                    erv=erv, peak=peak, n_eff=n_eff, lo=lo, hi=hi)


def measure_rv(v, ccf, n_eff):
    """Radial velocity, its ML uncertainty and the peak of a CCF.

    The sub-pixel peak (:func:`subpixel_peak`) and the curvature error
    (:func:`rv_uncertainty`) there -- for a combined CCF
    (:func:`combine_ccfs`), pass the total ``n_eff`` of its orders.

    Parameters
    ----------
    v : ndarray
        Velocity lags (km/s), increasing.
    ccf : ndarray
        CCF values at ``v`` (single-order or combined).
    n_eff : float
        Effective number of independent spectral bins behind ``ccf``
        (dimensionless, > 0).

    Returns
    -------
    RVMeasurement
        ``rv`` and ``erv`` (km/s) and ``peak`` (the CCF at ``rv``).

    Raises
    ------
    ValueError
        If the peak cannot be located (at the scan edge, not concave, or a
        non-finite CCF), or ``n_eff`` is not finite and positive.
    numpy.linalg.LinAlgError
        If the parabola fit around the peak fails.
    """
    rv, peak = subpixel_peak(v, ccf)
    return RVMeasurement(rv=rv, erv=rv_uncertainty(v, ccf, rv, n_eff), peak=peak)
