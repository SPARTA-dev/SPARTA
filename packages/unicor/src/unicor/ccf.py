"""One-dimensional cross-correlation: CCF, multi-order combination, RV + error.

The array-level UNICOR engine. Every function takes plain floats / numpy
arrays -- velocities in km/s, the CCF dimensionless -- so it can be used
without SPARTA's ``Spectrum`` / ``Template`` objects (which wrap it in
:class:`sparta.UNICOR.CCF1d`).

Typical use, for one order already on a common log-lambda grid::

    v, c = ccf_vs_velocity(template_flux, observed_flux, dv, vmax)
    rv, peak = subpixel_peak(v, c)
    erv = rv_uncertainty(v, c, rv, n_orders=1)

and, for several orders sharing the velocity lags ``v``::

    combined = combine_ccfs(ccf_matrix)          # Zucker (2003) ML combination
    rv, peak = subpixel_peak(v, combined)
    erv = rv_uncertainty(v, combined, rv, n_orders=len(ccf_matrix))
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


def combine_ccfs(ccfs):
    """Combine per-order CCFs into one, by maximum likelihood (Zucker 2003, §3.1).

    Parameters
    ----------
    ccfs : ndarray, shape (n_orders, n_lags)
        Per-order normalized CCFs on a common velocity-lag axis.

    Returns
    -------
    combined : ndarray, shape (n_lags,)
        ``sqrt(1 - prod(1 - C_i**2) ** (1 / n_orders))``.
    """
    # Clip to [-1, 1] so a per-order value that overshoots numerically cannot
    # drive 1 - C**2 negative (which would make the combined CCF NaN).
    C = np.clip(np.atleast_2d(np.asarray(ccfs, float)), -1.0, 1.0)
    n_orders = C.shape[0]
    return np.sqrt(1 - (np.prod(1 - C ** 2, axis=0)) ** (1 / n_orders))


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
    """
    if npts < 3:
        raise ValueError("npts must be at least 3")
    if npts % 2 != 1:
        raise ValueError("npts must be odd, to centre the fit on the peak")
    v = np.asarray(v, float)
    ccf = np.asarray(ccf, float)

    x_n = np.argmax(ccf)
    indlist = [int(x_n - npts // 2 + k) for k in np.arange(npts)]
    x = v[indlist]
    y = ccf[indlist]

    design = np.array([[1, xi, xi ** 2] for xi in x])
    cov = np.linalg.inv(design.T @ design)
    pars = cov @ (design.T @ y)
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
    """
    v = np.asarray(v, float)
    ccf = np.asarray(ccf, float)
    x_n = np.argmin(np.abs(v - v0))

    indlist = [int(x_n - 1), int(x_n), int(x_n + 1)]
    x = v[indlist]
    y = ccf[indlist]

    return (y[0] * (v0 - x[1]) * (v0 - x[2]) / (x[0] - x[1]) / (x[0] - x[2]) +
            y[1] * (v0 - x[0]) * (v0 - x[2]) / (x[1] - x[0]) / (x[1] - x[2]) +
            y[2] * (v0 - x[0]) * (v0 - x[1]) / (x[2] - x[0]) / (x[2] - x[1]))


def rv_uncertainty(v, ccf, v_peak, n_orders):
    """Maximum-likelihood RV uncertainty from the CCF curvature (Zucker 2003).

    Parameters
    ----------
    v : ndarray
        Velocity lags (km/s).
    ccf : ndarray
        CCF values at ``v`` (single-order or combined).
    v_peak : float
        Velocity of the CCF maximum (km/s).
    n_orders : int
        Number of orders that went into ``ccf``.

    Returns
    -------
    float
        1-sigma RV uncertainty (km/s); NaN when ``v_peak`` is not a true
        maximum of the spline (non-negative curvature).
    """
    v = np.asarray(v, float)
    n_vels = len(v)

    spl = interpolate.UnivariateSpline(v, ccf, k=4, s=0)
    spl_vv = spl.derivative(n=2)

    # The variance is -1 / (ml_factor * n_orders * n_vels) and is only real at
    # a true CCF maximum (ml_factor < 0); guard the degenerate cases rather
    # than emitting a silent NaN from sqrt().
    ml_factor = spl_vv(v_peak) * spl(v_peak) / (1 - spl(v_peak) ** 2)
    denom = ml_factor * n_orders * n_vels
    var = -1.0 / denom if denom != 0 else np.nan
    return float(np.sqrt(var)) if (np.isfinite(var) and var > 0) else np.nan


def bisector_span(v, ccf, n_orders, levels=(0.35, 0.95)):
    """Bisector span (BIS) of a single-peaked CCF and its uncertainty.

    Parameters
    ----------
    v : ndarray
        Velocity lags (km/s).
    ccf : ndarray
        CCF values at ``v``.
    n_orders : int
        Number of orders that went into ``ccf`` (sets the RV uncertainty).
    levels : (float, float)
        Lower and upper bisector levels, as fractions of the CCF maximum.

    Returns
    -------
    bis, ebis : float
        Bisector span and its uncertainty; NaN when either level does not cut
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

    v_peak, _ = subpixel_peak(v, ccf)
    err = rv_uncertainty(v, ccf, v_peak, n_orders)

    bis = (low[1] + low[0]) / 2 - (high[1] + high[0]) / 2
    return bis, np.sqrt(2) * err
