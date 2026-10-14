"""Phase-distance-correlation (PDC) math: distance matrices, centering, the scan.

Pure array functions -- no SPARTA objects. The PDC periodogram correlates the
pairwise distances of the values with the pairwise phase distances at each
trial frequency (Zucker 2018). The values' distance matrix is one of

* ``scalar_distance_matrix`` -- |y_i - y_j|, e.g. RVs (PDC);
* ``error_aware_distance_matrix`` -- the same with per-point uncertainties
  (Binnenfeld, Shahaf & Zucker 2024, A&A 686, A192, Eq. 3);
* ``spectral_distance_matrix`` -- the chord distance between spectra on the
  unit sphere (USuRPER, Binnenfeld, Shahaf & Zucker 2020).

It is centered once (``double_centering`` for the biased statistic,
``unbiased_u_centering`` for the unbiased one, ``partial_centering`` for the
partial periodograms of Binnenfeld et al. 2022) and ``pdc_power`` scans it
against the phase distances over a frequency grid. ``pdc_pvalue`` turns an
unbiased power into its single-frequency p-value (Shen et al. 2019) and
``pdc_power_level`` a p-value into the power it needs.

Conventions: times in days, frequencies in 1/day, wavelengths in Angstrom.
The unbiased U-centering uses 1/(n(n-3)) and /(n-2), /(n-1) factors --
singular or negative for n <= 3, which raise.
"""

import numpy as np
from scipy import special
from scipy.stats import chi2
from unicor import dsp


# =============================================================================
# Distance matrices
# =============================================================================
def scalar_distance_matrix(y):
    '''
    Pairwise absolute differences of scalar values.

    Parameters
    ----------
    y : array_like
        (n,) values (e.g. RVs, km/s).

    Returns
    -------
    ndarray
        (n, n) array of |y_i - y_j|, in the unit of ``y``.
    '''
    y = np.asarray(y, float)
    return np.abs(y[:, None] - y[None, :])


def error_aware_distance_matrix(y, errors):
    '''
    Pairwise distances of scalar values with Gaussian uncertainties.

    Binnenfeld, Shahaf & Zucker (2024), Eq. 3, as published:
    sqrt(s * (e^(-x^2) + x*erf(x) - (e_i + e_j)/(sqrt(2) s))), with
    s^2 = e_i^2 + e_j^2 and x = (y_i - y_j)/(sqrt(2) s) -- without a sqrt(pi)
    on the erf term. An energy-distance derivation suggests one; the authors
    confirmed (2026-10-14) that the published form is intended. Negative
    arguments are clipped to zero and the diagonal is zero.

    Parameters
    ----------
    y : array_like
        (n,) values.
    errors : array_like
        (n,) their 1-sigma uncertainties, in the unit of ``y``.

    Returns
    -------
    ndarray
        (n, n) array of distances, in units of sqrt(unit of ``y``).

    Raises
    ------
    ValueError
        If ``errors`` does not have one entry per value.
    '''
    y = np.asarray(y, float)
    err = np.asarray(errors, float)
    if err.shape != y.shape:
        raise ValueError(f"the error-aware distance needs one error per value; got "
                         f"{err.shape} errors for {y.shape} values.")
    y1, y2 = y[:, None], y[None, :]
    e1, e2 = err[:, None], err[None, :]
    den = np.sqrt(2 * (e1 ** 2 + e2 ** 2))
    x = (y1 - y2) / den
    yv = (e1 + e2) / den
    ev = np.sqrt(e1 ** 2 + e2 ** 2) * (np.exp(-x ** 2) + x * special.erf(x) - yv)
    a = np.sqrt(np.clip(ev, 0.0, None))
    np.fill_diagonal(a, 0.0)
    return a


def spectral_distance_matrix(waves, fluxes):
    '''
    Chord distances between spectra on the unit sphere (USuRPER).

    The spectra are resampled onto their common grid, mean-subtracted and
    L2-normalized; the distance of two unit vectors with zero-lag correlation
    r is sqrt(2 (1 - r)).

    Parameters
    ----------
    waves : list of ndarray
        1-D increasing wavelength arrays (Angstrom).
    fluxes : list of ndarray
        The matching 1-D flux arrays.

    Returns
    -------
    ndarray
        (n, n) array of distances in [0, 2], zero diagonal.
    '''
    gram = _gram(resample_spec_to_common_grid(waves, fluxes, factor=1))
    d = np.sqrt(np.clip(1.0 - gram, 0.0, None)) * np.sqrt(2)
    np.fill_diagonal(d, 0.0)
    return d


def phase_distance_matrix(times, f):
    '''
    The PDC phase-distance kernel at one trial frequency.

    Parameters
    ----------
    times : array_like
        (n,) observation times (days).
    f : float
        Trial frequency (1/day), > 0.

    Returns
    -------
    ndarray
        (n, n) array d (1/f - d), with d = (t_i - t_j) mod 1/f (days^2).
    '''
    t = np.asarray(times, float)[None, :]
    d = (t.T - t) % (1 / f)
    return d * (1 / f - d)


def _gram(sp_matrix):
    '''Pairwise inner products of the (L2-normalized, bounded) spectrum columns.

    Apple's Accelerate BLAS raises spurious FP-exception flags during matmul
    under numpy>=2 even for finite, small inputs; the Gram of unit vectors is
    bounded in [-1, 1] so a real exception is impossible. Silence the flags.
    '''
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return sp_matrix.T @ sp_matrix


# =============================================================================
# Centering and the U-centered inner product
# =============================================================================
def double_centering(a):
    '''
    Double-centered distance matrix (the biased distance correlation).

    Parameters
    ----------
    a : ndarray
        (n, n) distance matrix.

    Returns
    -------
    ndarray
        ``a`` minus its row and column means, plus the grand mean.
    '''
    return a - np.mean(a, axis=0)[None, :] - np.mean(a, axis=1)[:, None] + np.mean(a)


def unbiased_u_centering(a, n):
    '''
    U-centered distance matrix (the unbiased distance correlation).

    Row and column sums are divided by (n - 2), the grand sum by
    (n - 1)(n - 2), and the diagonal is set to zero (Szekely & Rizzo 2014).

    Parameters
    ----------
    a : array_like
        (n, n) distance matrix.
    n : int
        Its size.

    Returns
    -------
    ndarray
        (n, n) U-centered matrix, zero diagonal.

    Raises
    ------
    ValueError
        For n <= 3, where the /(n-2) factor here and the 1/(n(n-3)) of the
        U-centered inner product are singular or negative.
    '''
    if n <= 3:
        raise ValueError(
            f"The unbiased (U-centered) PDC needs at least 4 epochs; got n = {n}. "
            "Its 1/(n(n-3)) and /(n-2) factors are singular or negative for n <= 3.")
    a = np.asarray(a, float)
    row_sum = a.sum(axis=1) / (n - 2)
    col_sum = a.sum(axis=0) / (n - 2)
    tot_sum = col_sum.sum() / (n - 1)
    mat_unbiased = a - row_sum[:, None] - col_sum[None, :] + tot_sum
    np.fill_diagonal(mat_unbiased, 0.0)
    return mat_unbiased


def inner_prod(a, b):
    '''
    Unbiased U-centered inner product of two distance matrices.

    Parameters
    ----------
    a, b : ndarray
        (n, n) U-centered distance matrices.

    Returns
    -------
    float
        sum(a * b) / (n (n - 3)).
    '''
    n = len(a)
    return np.sum(a * b) / (n * (n - 3))


def norm_z(a, b):
    '''
    The component of ``a`` orthogonal to ``b`` under the U-centered inner
    product (used to build the partial / semi-partial distance correlation).

    Parameters
    ----------
    a, b : ndarray
        (n, n) U-centered distance matrices.

    Returns
    -------
    ndarray
        a - (<a, b> / <b, b>) b.
    '''
    factor = inner_prod(a, b) / inner_prod(b, b)
    return a - factor * b


def partial_centering(a, c):
    '''
    U-center a distance matrix and a control one, and remove the control's
    component from the first (partial distance correlation).

    Parameters
    ----------
    a : ndarray
        (n, n) distance matrix of the quantity of interest.
    c : ndarray
        (n, n) distance matrix of the control quantity.

    Returns
    -------
    A, C : ndarray
        U-centered ``a`` made orthogonal to C, and U-centered ``c``: the
        arguments of ``pdc_power(times, freqs, A, control=C)``.
    '''
    n = len(a)
    control = unbiased_u_centering(c, n)
    return norm_z(unbiased_u_centering(a, n), control), control


# =============================================================================
# The frequency scan
# =============================================================================
def pdc_power(times, freqs, a_centered, unbiased=True, control=None):
    '''
    PDC periodogram: correlation of a centered distance matrix with the phase
    distances, at every trial frequency.

    Parameters
    ----------
    times : array_like
        (n,) observation times (days).
    freqs : array_like
        (m,) trial frequencies (1/day), > 0.
    a_centered : ndarray
        (n, n) the values' distance matrix, centered the way the phase
        matrices will be: ``double_centering`` for unbiased=False,
        ``unbiased_u_centering`` for unbiased=True, or the first output of
        ``partial_centering`` with ``control``.
    unbiased : bool
        U-centered (True) or double-centered (False) statistic.
    control : ndarray, optional
        The U-centered control matrix (the second output of
        ``partial_centering``): every phase matrix is made orthogonal to it
        -- the partial periodogram. Unbiased only.

    Returns
    -------
    ndarray
        (m,) PDC power (dimensionless).

    Raises
    ------
    ValueError
        For a control with the biased statistic, or n <= 3 with the unbiased
        one.
    '''
    if control is not None and not unbiased:
        raise ValueError("the partial periodogram is defined for the unbiased "
                         "(U-centered) statistic only.")
    n = len(times)
    power = np.empty(len(freqs))
    for k, f in enumerate(freqs):
        b = phase_distance_matrix(times, f)
        if not unbiased:
            power[k] = _centered_correlation(a_centered, double_centering(b))
        elif control is None:
            power[k] = _centered_correlation(a_centered, unbiased_u_centering(b, n))
        else:
            B = norm_z(unbiased_u_centering(b, n), control)
            power[k] = (inner_prod(a_centered, B)
                        / (np.sqrt(inner_prod(a_centered, a_centered))
                           * np.sqrt(inner_prod(B, B))))
    return power


def _centered_correlation(A, B):
    '''Cosine of two centered matrices: sum(A B) / sqrt(sum(A^2) sum(B^2)).'''
    return np.sum(A * B) / np.sqrt(np.sum(A ** 2) * np.sum(B ** 2))


# =============================================================================
# p-values
# =============================================================================
def pdc_pvalue(power, n):
    '''
    Single-frequency p-value of an unbiased PDC power.

    The chi-square approximation of the distance-correlation test (Shen et
    al. 2019, arXiv:1912.12150): p = chi2_1.sf(n * power + 1). It holds for
    the unbiased (U-centered) statistics at one trial frequency; it is not a
    false-alarm probability over a frequency grid.

    Parameters
    ----------
    power : array_like
        Unbiased PDC / USuRPER / partial power (dimensionless).
    n : int
        Number of epochs (> 0).

    Returns
    -------
    ndarray
        The p-values, shaped like ``power``.

    Raises
    ------
    ValueError
        If ``n`` is not positive.
    '''
    _require_epochs(n)
    return chi2.sf(np.asarray(power, float) * n + 1, 1)


def pdc_power_level(pvalue, n):
    '''
    The unbiased PDC power whose single-frequency p-value is ``pvalue``.

    The inverse of :func:`pdc_pvalue`: (chi2_1.isf(p) - 1) / n.

    Parameters
    ----------
    pvalue : array_like
        p-values in (0, 1].
    n : int
        Number of epochs (> 0).

    Returns
    -------
    ndarray
        The power levels (dimensionless), shaped like ``pvalue``.

    Raises
    ------
    ValueError
        If ``n`` is not positive.
    '''
    _require_epochs(n)
    return (chi2.isf(np.asarray(pvalue, float), 1) - 1) / n


def _require_epochs(n):
    '''Raise ValueError unless the number of epochs ``n`` is positive.'''
    if not n > 0:
        raise ValueError(f"the PDC p-value needs a positive number of epochs, got n = {n}")


# =============================================================================
# The common wavelength grid of the spectral distance
# =============================================================================
def get_wv_grid(waves, factor):
    '''
    The uniform wavelength grid common to all spectra: their overlap, sampled at
    the finest median pixel step.

    Parameters
    ----------
    waves : list of ndarray
        1-D increasing wavelength arrays (Angstrom).
    factor : float
        Multiplies the grid step.

    Returns
    -------
    ndarray
        1-D wavelength grid (Angstrom).

    Raises
    ------
    ValueError
        If the spectra do not overlap (fewer than two common grid points).
    '''
    min_wv = np.max([np.min(w) for w in waves])
    max_wv = np.min([np.max(w) for w in waves])
    delta_wv = np.min([np.median(np.diff(w)) for w in waves])

    grid = np.arange(start=min_wv, stop=max_wv, step=delta_wv*factor)
    if len(grid) < 2:
        raise ValueError(
            f"The spectra do not overlap in wavelength (common range "
            f"{min_wv:.4f}-{max_wv:.4f} Angstrom); no common grid to resample onto.")
    return grid


def resample_spec_to_common_grid(waves, fluxes, factor):
    '''
    Resample spectra onto their common grid and normalize each one.

    The spectra are linearly interpolated (``unicor.resample``).

    Parameters
    ----------
    waves : list of ndarray
        1-D increasing wavelength arrays (Angstrom).
    fluxes : list of ndarray
        The matching 1-D flux arrays.
    factor : float
        Multiplies the common-grid step (see ``get_wv_grid``).

    Returns
    -------
    ndarray
        (n_grid, n_spectra); each column mean-subtracted and L2-normalized.
    '''
    wv_grid   = get_wv_grid(waves, factor)
    sp_matrix = np.zeros((len(wv_grid), len(waves)))
    for i, (w, f) in enumerate(zip(waves, fluxes)):
        sp_matrix[:, i] = dsp.normalize(dsp.resample(w, f, wv_grid))
    return sp_matrix
