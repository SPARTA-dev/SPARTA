"""One-dimensional cross-correlation: CCF, multi-order combination, RV + error.

The array-level UNICOR engine. Every function takes plain floats / numpy
arrays -- velocities in km/s, the CCF dimensionless -- so it can be used
without SPARTA's ``Spectrum`` / ``Template`` objects (which wrap it in
:class:`sparta.UNICOR.CCF1d`).

Typical use, for one order already on a common log-lambda grid::

    v, c = ccf_vs_velocity(template_flux, observed_flux, dv, vmax)
    rv, peak = subpixel_peak(v, c)
    erv = rv_uncertainty(v, c, rv, n_eff)

and, for several orders sharing the velocity lags ``v``::

    combined = combine_ccfs(ccf_matrix, n_eff=n_eff_per_order)  # Zucker (2003)
    rv, peak = subpixel_peak(v, combined)
    erv = rv_uncertainty(v, combined, rv, n_eff_per_order.sum())

``n_eff`` is the number of *independent* spectral bins: the number of pixels
on the native (detector) sampling, not on a finer resampled grid -- and not the
number of velocity lags.
"""
import numpy as np
from scipy import interpolate
from scipy.signal import correlate as _correlate


def ccf_vs_velocity(template, signal, dv, vmax):
    """Normalized cross-correlation of ``template`` and ``signal`` vs velocity.

    Both arrays must be sampled on the SAME uniform log-lambda grid with
    velocity step ``dv`` (km/s), so a shift of k pixels is k*dv km/s. Both are
    mean-subtracted, so this is the Pearson correlation of the line residuals
    (no continuum-DC taper). The scan is symmetric over ``[-vmax, vmax]``, so
    the whole axis is uniformly valid.

    Parameters
    ----------
    template, signal : ndarray
        Flux on a shared uniform log-lambda grid.
    dv : float
        Velocity step of that grid (km/s).
    vmax : float
        Half-width of the velocity scan (km/s); clipped to the array length.

    Returns
    -------
    v : ndarray
        Symmetric velocity lags (km/s): ``arange(-K, K+1) * dv``.
    ccf : ndarray
        Normalized correlation in [-1, 1]. Peaks at the velocity by which
        ``signal`` is red-shifted relative to ``template``.
    """
    if not dv > 0:
        raise ValueError(f"dv must be positive, got {dv}")
    t = np.asarray(template, float)
    s = np.asarray(signal, float)
    if t.shape != s.shape:
        raise ValueError("template and signal must share the same grid/length")
    t = t - t.mean()
    s = s - s.mean()
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(s))):
        raise ValueError("template and signal must be finite (mask NaN pixels first)")
    norm = np.sqrt((t ** 2).sum() * (s ** 2).sum())
    if norm == 0:
        # A flat array has no lines to correlate -- typically a template that
        # does not cover the observed range and was clamped to a constant.
        raise ValueError("template or signal is constant; check wavelength coverage")
    full = _correlate(s, t, mode="full", method="fft")
    mid = len(t) - 1
    K = min(int(round(vmax / dv)), mid)
    ccf = full[mid - K: mid + K + 1] / norm
    v = np.arange(-K, K + 1) * dv
    return v, ccf


def combine_ccfs(ccfs, n_eff=None):
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
        (dimensionless, > 0). None weights all orders equally.

    Returns
    -------
    combined : ndarray, shape (n_lags,)
        The ML-combined CCF, in [0, 1]; pair it with ``N_tot = sum(n_eff)`` in
        :func:`rv_uncertainty`.

    Raises
    ------
    ValueError
        If ``n_eff`` does not hold one finite, positive value per order.
    """
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
    return np.sqrt(1 - np.prod((1 - C ** 2) ** weights[:, None], axis=0))


def subpixel_peak(v, ccf, npts=5):
    """Sub-pixel CCF peak from a least-squares parabola around the maximum.

    Parameters
    ----------
    v : ndarray
        Velocity lags (km/s).
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
    x = v[x_n - half: x_n + half + 1]
    y = ccf[x_n - half: x_n + half + 1]

    design = np.array([[1, xi, xi ** 2] for xi in x])
    cov = np.linalg.inv(design.T @ design)
    pars = cov @ (design.T @ y)
    if not pars[2] < 0:
        raise ValueError("CCF is not concave around its maximum")
    ccf_peak = pars[0] - pars[1] * pars[1] / (4 * pars[2])
    v_peak = -pars[1] / (2 * pars[2])
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
        Bisector span and its uncertainty (km/s); NaN when either level does not cut
        the CCF exactly twice.
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
