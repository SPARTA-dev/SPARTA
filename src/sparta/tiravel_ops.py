"""The TIRAVEL alignment measure, optimal weights, and co-added template.

Method
------
Given `K` observed spectra of one target and a trial set of radial velocities
`v_1..v_K`, shift every spectrum to its rest frame and resample them onto one
common wavelength grid.  Form the `K x K` correlation matrix `R` of the aligned
spectra, take its largest eigenvalue `lambda_max`, and report

    rho = (lambda_max - 1) / (K - 1)

`rho` measures how well the whole set is described by a *single* principal
component -- that is, how well the spectra are aligned.  It runs from 0 (the
spectra share nothing beyond their own variance) to 1 (they are identical up to
scale).  In the full TIRAVEL method `rho` is **maximised over the K velocities**;
this module evaluates it at a supplied set, and builds the template that goes
with it.

The eigenvector of `lambda_max` gives, in Zucker & Mazeh's words, "the optimal
weights with which the individual spectra should be summed": co-adding the
normalized aligned spectra with those weights is the maximum-likelihood estimate
of the common (unknown) template.

Zero point
----------
Only the velocity *differences* enter `R`: adding a constant to every velocity
leaves `R`, `lambda_max` and `rho` unchanged, and simply shifts the resulting
template by the same amount.  TIRAVEL velocities are therefore **relative**, and
the rest frame of the returned template is whatever zero point the supplied
velocities carry.  An absolute scale needs an external anchor (a library
template, a telluric standard, ...).

Normalization
-------------
Every aligned spectrum is mean-subtracted per order (removing each order's own
continuum offset) and then L2-normalized across all orders jointly, so `R` is a
true correlation matrix (unit diagonal, entries in [-1, 1]) and the orders
contribute in proportion to their information rather than their flux scale.

Conventions
-----------
Pure functions on plain numpy arrays: no SPARTA objects, no astropy Quantity,
no I/O. Velocities in km/s, wavelengths in Angstrom; spectra enter as rows of
a (K, M) matrix -- K epochs, M pixels of one common rest-frame grid.

The whole build
---------------
:func:`build_tiravel` chains the steps for multi-order epochs given as
per-epoch, per-order arrays -- the body of SPARTA's
``TIRAVEL.build_template``::

    res = build_tiravel(waves, fluxes, velocities)   # waves[k][I], fluxes[k][I]
    res.rho, res.weights, res.grids, res.template_flux

It returns a :class:`TiravelResult`.

Reference
---------
Zucker, S. & Mazeh, T. 2006, MNRAS 371, 1513 (astro-ph/0607293).
"""

from dataclasses import dataclass

import numpy as np

from unicor.dsp import C_KMS, resample


# =============================================================================
# =============================================================================
def rest_frame_wavelengths(wave, velocity):
    """Map observed wavelengths to the source's rest frame.

    A line emitted at ``lam_rest`` is observed at ``lam_rest * (1 + v/c)``, so
    de-shifting is the exact division -- not the ``*(1 - v/c)`` first-order form.

    Parameters
    ----------
    wave : ndarray
        Observed wavelengths (Angstrom).
    velocity : float
        Radial velocity of the source (km/s); positive is red-shifted.

    Returns
    -------
    ndarray
        Rest-frame wavelengths (Angstrom).
    """
    return np.asarray(wave, float) / (1.0 + velocity / C_KMS)


# =============================================================================
# =============================================================================
def common_grid(waves, velocities, reference=0, margin=0):
    """The rest-frame wavelength grid shared by every epoch.

    Takes the reference epoch's rest-frame grid and clips it to the wavelength
    range all epochs cover once de-shifted, so no epoch is ever extrapolated.
    Reusing an existing grid rather than inventing one means the reference epoch
    is not resampled at all.

    Parameters
    ----------
    waves : sequence of ndarray
        One observed wavelength array (Angstrom) per epoch.
    velocities : sequence of float
        One radial velocity (km/s) per epoch.
    reference : int
        Index of the epoch whose sampling defines the grid.
    margin : int
        Extra pixels trimmed from each end of the overlap, as insurance against
        edge effects in the interpolation.

    Returns
    -------
    ndarray
        The common rest-frame grid (Angstrom).

    Raises
    ------
    ValueError
        If ``margin`` is negative or the de-shifted epochs have no wavelength
        range in common.
    """
    if margin < 0:
        raise ValueError(f'margin must be >= 0 pixels, got {margin}')
    rest = [rest_frame_wavelengths(w, v) for w, v in zip(waves, velocities)]
    lo = max(float(r.min()) for r in rest)
    hi = min(float(r.max()) for r in rest)
    if not lo < hi:
        raise ValueError(
            f'the de-shifted epochs do not overlap (common range {lo:.3f} to '
            f'{hi:.3f} Angstrom); check the supplied velocities')

    grid = rest[reference]
    grid = grid[(grid >= lo) & (grid <= hi)]
    if margin:
        grid = grid[margin:len(grid) - margin]
    if len(grid) < 4:
        raise ValueError(
            f'only {len(grid)} pixels are common to all epochs; the velocity '
            'spread is too large for the wavelength range, or margin is too big')
    return grid


# =============================================================================
# =============================================================================
def align_spectra(waves, fluxes, velocities, grid, kind='cubic'):
    """Shift each epoch to rest and resample it onto ``grid``.

    Parameters
    ----------
    waves, fluxes : sequence of ndarray
        One observed wavelength array (Angstrom) and flux array per epoch.
    velocities : sequence of float
        One radial velocity (km/s) per epoch.
    grid : ndarray
        The common rest-frame grid (Angstrom), e.g. from :func:`common_grid`.
    kind : {'cubic', 'linear'}
        Interpolation used for the resampling. Cubic by default: the shifts run
        to many pixels, and linear interpolation would smooth the lines enough
        to depress the very correlations being measured.

    Returns
    -------
    ndarray, shape (K, len(grid))
        The aligned spectra, one epoch per row.
    """
    return np.vstack([
        resample(rest_frame_wavelengths(w, v), f, grid, kind=kind)
        for w, f, v in zip(waves, fluxes, velocities)])


# =============================================================================
# =============================================================================
def normalize_rows(blocks):
    """Mean-subtract each block and L2-normalize the epochs across all blocks.

    Parameters
    ----------
    blocks : sequence of ndarray
        One (K, M_b) matrix per order/block, all sharing the same K epochs.
        Each block's own mean is removed per epoch (its continuum offset), then
        every epoch's concatenated row is scaled to unit L2 norm, so orders
        contribute in proportion to their line content rather than their flux
        scale.

    Returns
    -------
    ndarray, shape (K, sum(M_b))
        The normalized, concatenated rows. A zero row (a flat spectrum) is left
        at zero rather than divided by zero.
    """
    centred = [np.asarray(b, float) - np.mean(b, axis=1, keepdims=True)
               for b in blocks]
    stacked = np.hstack(centred) if len(centred) > 1 else centred[0]
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    return np.divide(stacked, norms, out=np.zeros_like(stacked),
                     where=norms > 0)


# =============================================================================
# =============================================================================
def correlation_matrix(normalized):
    """Pairwise correlation matrix of normalized epoch rows.

    For mean-subtracted, L2-normalized rows the Pearson correlation is just the
    inner product, so the whole matrix is one Gram product -- ``O(K^2 M)`` in
    BLAS rather than ``K^2`` separate cross-correlations. It is symmetric by
    construction, which is what lets :func:`alignment` use ``eigh``.

    Parameters
    ----------
    normalized : ndarray, shape (K, M)
        Output of :func:`normalize_rows`.

    Returns
    -------
    ndarray, shape (K, K)
        Correlation matrix, unit diagonal, entries in [-1, 1].
    """
    normalized = np.asarray(normalized, float)
    # Apple's Accelerate BLAS raises spurious FP-exception flags during matmul
    # under numpy>=2; the Gram of unit vectors is bounded so this is safe.
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        R = normalized @ normalized.T
    np.fill_diagonal(R, 1.0)
    return np.clip(R, -1.0, 1.0)


# =============================================================================
# =============================================================================
def alignment(R):
    """The TIRAVEL alignment measure and the optimal co-addition weights.

    ``rho = (lambda_max - 1) / (K - 1)`` quantifies how well one principal
    component describes the whole set: 1 when the aligned spectra are identical
    up to scale, ~0 when they share nothing. It is the quantity the full method
    maximises over the velocities.

    ``R`` is symmetric, so ``numpy.linalg.eigh`` is used: it returns *real,
    ascending* eigenvalues, which removes the ordering trap of the general
    ``eig`` (whose output is unsorted, so column 0 is often not the principal
    eigenvector at all).

    Parameters
    ----------
    R : ndarray, shape (K, K)
        Correlation matrix from :func:`correlation_matrix`.

    Returns
    -------
    dict
        ``rho`` : the alignment measure.
        ``lambda_max`` : the largest eigenvalue.
        ``weights`` : (K,) the corresponding eigenvector, unit norm, sign-fixed
        so the bulk of the weight is positive. These are the weights the
        spectra should be summed with -- *not* their squares.
        ``eigenvalues`` : (K,) all eigenvalues, descending (they sum to K).
        ``explained`` : ``lambda_max / K``, the fraction of the total variance
        carried by the leading component.
        ``n_eff`` : the participation ratio ``1 / sum(weights**4)`` -- roughly
        how many epochs actually contribute to the template.
    """
    R = np.asarray(R, float)
    K = R.shape[0]
    if K < 2:
        raise ValueError(f'TIRAVEL needs at least 2 epochs, got {K}')

    eigenvalues, eigenvectors = np.linalg.eigh(R)          # ascending, real
    lambda_max = float(eigenvalues[-1])
    weights = eigenvectors[:, -1]

    # The eigenvector's overall sign is arbitrary; fix it so the co-added
    # template comes out with the same polarity as the input spectra.
    if weights.sum() < 0:
        weights = -weights

    return {
        'rho': (lambda_max - 1.0) / (K - 1.0),
        'lambda_max': lambda_max,
        'weights': weights,
        'eigenvalues': eigenvalues[::-1],
        'explained': lambda_max / K,
        'n_eff': float(1.0 / np.sum(weights ** 4)),
    }


# =============================================================================
# =============================================================================
def coadd(matrix, weights):
    """Weighted co-addition of aligned spectra -> the optimal template.

    ``sum_i weights[i] * matrix[i]``: the projection of the set onto its
    leading principal component, which is the maximum-likelihood estimate of
    the common template the spectra are noisy, shifted copies of.

    Parameters
    ----------
    matrix : ndarray, shape (K, M)
        Aligned (and normally normalized) spectra, one epoch per row.
    weights : ndarray, shape (K,)
        From :func:`alignment`.

    Returns
    -------
    ndarray, shape (M,)
        The co-added template. Its absolute scale is arbitrary -- correlation
        against it is scale-invariant.
    """
    # Silence Accelerate's spurious matmul FP flags under numpy>=2 (see
    # correlation_matrix / sparta.pdc._gram).
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        return np.asarray(weights, float) @ np.asarray(matrix, float)


# =============================================================================
# The whole build
# =============================================================================
@dataclass(frozen=True, eq=False)
class TiravelResult:
    """
    The TIRAVEL template and alignment of a set of epochs (:func:`build_tiravel`).

    Parameters
    ----------
    rho : float
        The alignment measure ``(lambda_max - 1)/(K - 1)`` (dimensionless, in
        [0, 1]; :func:`alignment`).
    weights : ndarray, shape (K,)
        Co-addition weights (unit norm, bulk positive).
    correlation : ndarray, shape (K, K)
        The epochs' correlation matrix (:func:`correlation_matrix`), all
        orders pooled.
    diagnostics : dict
        'lambda_max', 'eigenvalues', 'explained' and 'n_eff', as returned by
        :func:`alignment`.
    velocities : ndarray, shape (K,)
        The trial velocities used (km/s), as floats.
    grids : list of ndarray
        The common rest-frame grid of each order used (Angstrom), in the
        order of ``orders``.
    template_flux : list of ndarray
        The co-added template on each grid (:func:`coadd`; dimensionless,
        mean-free, arbitrary overall scale).
    aligned : list of ndarray
        Each order's aligned epochs, shape (K, len(grid)): mean-subtracted
        per order, every epoch of unit L2 norm across all orders jointly
        (:func:`normalize_rows`).
    """
    rho: float
    weights: np.ndarray
    correlation: np.ndarray
    diagnostics: dict
    velocities: np.ndarray
    grids: list
    template_flux: list
    aligned: list


def build_tiravel(waves, fluxes, velocities, orders=None, reference=0, margin=10,
                  kind='cubic'):
    """Build the TIRAVEL template of a set of epochs at given velocities.

    Every epoch is Doppler-shifted to its rest frame and resampled onto one
    common grid per order (:func:`common_grid`, :func:`align_spectra`). The
    epochs are mean-subtracted per order and L2-normalized across all orders
    (:func:`normalize_rows`), their K x K correlation matrix is one Gram
    product (:func:`correlation_matrix`), and its leading eigenpair gives
    both ``rho`` and the weights (:func:`alignment`) with which each order's
    epochs are co-added (:func:`coadd`). Orders are pooled into one
    correlation matrix, so a single weight vector describes the whole
    spectrum.

    Parameters
    ----------
    waves : sequence of sequence of ndarray
        ``waves[k][I]``: the observed wavelengths (Angstrom) of epoch k,
        order I. K >= 2 epochs, all with the same number of orders.
    fluxes : sequence of sequence of ndarray
        ``fluxes[k][I]``: the matching flux (finite); not modified.
    velocities : array_like
        K trial radial velocities (km/s). Only their differences matter: a
        constant added to all leaves ``rho`` unchanged and shifts the
        template by the same amount.
    orders : sequence of int, optional
        Order indices to use, in this order. Default all.
    reference : int
        Index of the epoch whose rest-frame sampling defines each common grid.
    margin : int
        Pixels trimmed from each end of the common range (>= 0).
    kind : {'cubic', 'linear'}
        Resampling of the shifted epochs (see :func:`align_spectra`).

    Returns
    -------
    TiravelResult
        ``rho``, the weights, the correlation matrix and its diagnostics, the
        velocities, and per order the grid, the template flux and the aligned
        epochs.

    Raises
    ------
    ValueError
        For mismatched or too few epochs, non-finite velocities or flux,
        epochs with different numbers of orders, or epochs that do not
        overlap once de-shifted (:func:`common_grid`).
    """
    if len(waves) != len(velocities):
        raise ValueError(
            f'got {len(waves)} spectra but {len(velocities)} velocities')
    if len(waves) < 2:
        raise ValueError('TIRAVEL needs at least 2 epochs')
    if len(fluxes) != len(waves):
        raise ValueError(f'one flux list per epoch needed: got {len(fluxes)} for '
                         f'{len(waves)} epochs')

    velocities = np.asarray(velocities, float)
    if not np.all(np.isfinite(velocities)):
        raise ValueError('the supplied velocities contain NaN/inf; drop '
                         'those epochs before building a template')

    n_ord = len(waves[0])
    if any(len(w) != n_ord for w in waves):
        raise ValueError('all epochs must have the same number of orders')
    for k, (w, f) in enumerate(zip(waves, fluxes)):
        if len(f) != len(w):
            raise ValueError(f'epoch {k}: one flux array per order needed, got '
                             f'{len(f)} for {len(w)} orders')
    if orders is None:
        orders = range(n_ord)

    # Align every epoch, order by order, onto that order's common grid.
    grids, aligned = [], []
    for I in orders:
        order_waves = [np.asarray(w[I], float) for w in waves]
        order_fluxes = [np.asarray(f[I], float) for f in fluxes]
        # A single NaN would otherwise zero that epoch's norm (linear) or
        # stop the spline with an opaque message (cubic).
        for k, f in enumerate(order_fluxes):
            if not np.all(np.isfinite(f)):
                raise ValueError(f'epoch {k}, order {I}: flux contains NaN/inf; '
                                 'mask or interpolate those pixels first')
        grid = common_grid(order_waves, velocities, reference=reference, margin=margin)
        grids.append(grid)
        aligned.append(align_spectra(order_waves, order_fluxes, velocities, grid,
                                     kind=kind))

    # One normalization, one correlation matrix, one weight vector for the
    # whole spectrum: the orders are different views of the same alignment.
    normalized = normalize_rows(aligned)
    correlation = correlation_matrix(normalized)
    stats = alignment(correlation)
    weights = stats.pop('weights')
    rho = stats.pop('rho')

    # Split the normalized rows back per order and co-add each.
    sizes = np.cumsum([0] + [len(g) for g in grids])
    blocks = [normalized[:, a:b] for a, b in zip(sizes[:-1], sizes[1:])]
    return TiravelResult(rho=rho, weights=weights, correlation=correlation,
                         diagnostics=stats, velocities=velocities, grids=grids,
                         template_flux=[coadd(block, weights) for block in blocks],
                         aligned=blocks)
