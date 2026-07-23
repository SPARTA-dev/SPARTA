#                   ------------------------------------
#                     todcor_ops.py (SPARTA TODCOR core)
#                   ------------------------------------
# The Fourier-domain TODCOR estimator: pure functions on plain numpy arrays,
# no SPARTA objects, no astropy Quantity, no I/O. The Spectrum/Template
# orchestration lives in CCF2d.py; the forward model in simulate.py; the shared
# grid/FFT/broadening primitives in sparta.spectral_ops.
#
# Conventions: velocities and the log-lambda shift s in km/s; zeta (the Fourier
# conjugate) in s/km; spectra on a uniform log-lambda grid of step dv (km/s);
# residual flux g = flux - continuum.
#
# Reference: Zucker, S. & Mazeh, T. 1994, ApJ 420, 806 -- "Study of
# spectroscopic binaries with TODCOR. I. A new two-dimensional correlation
# algorithm."
#
# Dependencies: numpy, sparta.spectral_ops.
"""The Fourier-domain TODCOR estimator (the derivation).

Method
------
A composite, continuum-normalised spectrum of a double-lined system is
modelled, in the residual flux ``g = flux - 1``, as the sum of two
Doppler-shifted, scaled template spectra::

    g(u) = a1 * g1(u - s1) + a2 * g2(u - s2)

where ``u`` is a velocity coordinate (km/s) on a uniform log-lambda grid and
``s_i = c * ln(1 + v_i / c)`` is the shift of component *i*.  On the log-lambda
grid a Doppler shift is a translation, so in the Fourier domain it becomes a
phase factor (the shift theorem) and the *templates themselves* are transformed
once and reused at every trial velocity::

    G_tilde(zeta; v1, v2) = a1 * G1(zeta) * exp(-2j*pi*zeta*s1)
                          + a2 * G2(zeta) * exp(-2j*pi*zeta*s2)

The amplitudes ``(a1, a2)`` enter linearly, so for every trial ``(v1, v2)`` they
are solved in closed form by 2x2 weighted least squares.  Profiling them out
gives a log-likelihood surface over ``(v1, v2)`` -- the classic TODCOR
correlation map (Zucker & Mazeh 1994), here cast as a Gaussian likelihood.

Sufficient statistics
---------------------
For one spectral window the weighted (Parseval) chi^2 at any trial
``(v1, v2, a1, a2)`` is a fixed quadratic form in just **six** numbers -- the
window's *sufficient statistics*.  With per-mode weight ``w_k`` (see
:func:`sparta.spectral_ops.parseval_weights`) and shifts
``s_i = c*ln(1 + v_i/c)``::

    S11 = sum_k w_k |G1_k|^2                                   (scalar)
    S22 = sum_k w_k |G2_k|^2                                   (scalar)
    SD  = sum_k w_k |D_k|^2                                    (scalar)
    b1(v1) = Re sum_k w_k conj(G1_k) D_k exp(+2j*pi*zeta_k*s1) (n1,)
    b2(v2) = Re sum_k w_k conj(G2_k) D_k exp(+2j*pi*zeta_k*s2) (n2,)
    A12(v1,v2) = Re sum_k w_k conj(G1_k) G2_k
                              exp(-2j*pi*zeta_k*(s2 - s1))      (n1, n2)

Given these six the raw spectrum is never needed again: expanding
``sum_k w_k |D_k - a1 G1_k e^{-i phi1} - a2 G2_k e^{-i phi2}|^2`` gives::

    chi^2 = SD - 2(a1 b1 + a2 b2) + a1^2 S11 + 2 a1 a2 A12 + a2^2 S22

which :func:`window_stats` returns and :func:`profile_from_stats` minimises.

Profiling and marginalising the amplitudes
------------------------------------------
The 2x2 normal equations ``N a = b`` with ``N = [[S11, A12], [A12, S22]]`` and
``b = (b1, b2)`` solve in closed form (``det = S11*S22 - A12^2``)::

    a1 = (S22 b1 - A12 b2) / det        a2 = (S11 b2 - A12 b1) / det

Substituting the profile amplitudes gives ``logL_profile = -chi^2/2``.
*Marginalising* the amplitudes under a flat (improper) prior on
``(a1, a2) in R^2`` -- a Gaussian integral with precision ``N`` -- adds the
usual ``-1/2 ln|det|`` (the amplitude-posterior width)::

    logL_marginal = logL_profile - 1/2 ln|det|

When ``nonneg=True`` the amplitudes are constrained to the physical quadrant
``a >= 0`` (2-variable NNLS), which removes the sign-flip degeneracy but makes
the exact marginal an *orthant-truncated* Gaussian integral
(``... - 1/2 ln|det| + ln Phi_N(a >= 0)``).  We do not evaluate that; in
constrained mode ``logL_marginal`` is simply set equal to ``logL_profile``
(dropping **both** the ``-1/2 ln|det|`` and the truncation term).

Combining windows
-----------------
The six statistics are indexed by *velocity* (through ``s_i``), not by Fourier
mode, so windows on different grids / instruments / resolutions combine by
**summation** of the six statistics (:func:`combine_stats`) into one joint fit
with **shared ``(v1, v2, a1, a2)``**.  This is valid only *within a single
epoch* -- pooled windows must share one true ``(v1, v2)``; do NOT combine
across epochs (orbital motion changes the velocities).  In SPARTA the natural
windows are the spectral **orders** of one observation.

Assumptions (exact only under these)
------------------------------------
* **White, homoscedastic per-pixel noise** ``sigma`` on the analysis grid.  The
  Tukey taper (:func:`sparta.spectral_ops.tukey_window`) makes the noise
  heteroscedastic and interpolation in
  :func:`sparta.spectral_ops.resample` *correlates* (colours) it; only the
  overall *scale* is compensated (reduced-chi^2 rescaling,
  ``sigma *= sqrt(chi2_red)``), not the correlation -- so CRLB errors are
  mildly optimistic.
* **Flat R^2 prior** on the amplitudes for the marginal map (above).
* **Shared velocity grid** for :func:`combine_stats` (below).
"""

import numpy as np

from sparta.spectral_ops import parseval_weights, rfft_positive, shift_kms


# =============================================================================
# =============================================================================
def window_stats(data_g, g1, g2, dv, v1_grid, v2_grid, sigma=1.0, zeta_max=None):
    """Weighted least-squares sufficient statistics for one spectral window.

    The 2x2 amplitude problem at every ``(v1, v2)`` node is fully described by
    six weighted inner products -- ``S11``, ``S22``, ``SD``, ``b1(v1)``,
    ``b2(v2)``, ``A12(v1, v2)`` -- defined in the module docstring.  Crucially
    they are indexed only by the velocity grid (through the shifts ``s_i``), not
    by Fourier mode, so windows -- even from different instruments with
    different grids -- combine by simple addition (:func:`combine_stats`);
    amplitudes and velocities are then shared across all combined windows.

    Parameters
    ----------
    data_g : ndarray
        Observed residual flux (continuum removed), on a uniform log-lambda
        grid of step ``dv``, already windowed/apodised.
    g1, g2 : ndarray
        The two template residuals, already broadened, apodised, and resampled
        onto the *same* grid as ``data_g``.
    dv : float
        Velocity step of the grid (km/s).
    v1_grid, v2_grid : ndarray
        Trial radial velocities of each component (km/s).
    sigma : float
        Per-pixel noise of ``data_g`` (same flux units), assumed white and
        homoscedastic.
    zeta_max : float or None
        Optional low-pass: drop Fourier modes above ``zeta_max`` (s/km), e.g.
        to exclude scales finer than the instrumental resolution.

    Returns
    -------
    dict
        ``S11``, ``S22``, ``SD`` : scalar template/data weighted powers.
        ``b1`` (n1,), ``b2`` (n2,) : data-template cross-terms vs ``v1``/``v2``.
        ``A12`` (n1, n2) : template-template cross-term vs ``(v1, v2)``.
    """
    data_g = np.asarray(data_g, float)
    n = len(data_g)
    if len(g1) != n or len(g2) != n:
        raise ValueError("data_g, g1 and g2 must share one grid "
                         f"(got {n}, {len(g1)}, {len(g2)})")

    D, zeta = rfft_positive(data_g, dv)
    G1 = np.fft.rfft(np.asarray(g1, float))[1:]
    G2 = np.fft.rfft(np.asarray(g2, float))[1:]
    w = parseval_weights(n, sigma=sigma)

    # For even N the Nyquist coefficient of a real signal must be real; the
    # phase-shift model (via irfft in `model_flux`) enforces that, but the
    # complex phase factors used for b1/b2/A12 do not. Dropping the single
    # Nyquist mode keeps the fit self-consistent and preserves the exact
    # Parseval identity for the retained modes (its power is negligible).
    if n % 2 == 0:
        D, G1, G2, zeta, w = D[:-1], G1[:-1], G2[:-1], zeta[:-1], w[:-1]

    if zeta_max is not None:
        keep = zeta <= zeta_max
        D, G1, G2, zeta, w = D[keep], G1[keep], G2[keep], zeta[keep], w[keep]

    s1 = shift_kms(np.asarray(v1_grid, float))  # (n1,)
    s2 = shift_kms(np.asarray(v2_grid, float))  # (n2,)

    S11 = np.sum(w * np.abs(G1) ** 2)
    S22 = np.sum(w * np.abs(G2) ** 2)
    SD = np.sum(w * np.abs(D) ** 2)

    # The phase factors are unit-modulus and the weighted spectra are finite, so
    # the sums below cannot legitimately overflow or divide by zero -- but
    # Apple's Accelerate BLAS raises spurious FP-exception flags during complex
    # matmul under numpy>=2 (same issue as USURPER_functions._gram). Silence them.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        # Data-template cross terms:
        #   b_i(v_i) = Re sum_k w conj(G_i) D exp(+2j pi zeta s_i)
        wG1D = w * np.conj(G1) * D
        wG2D = w * np.conj(G2) * D
        phase1 = np.exp(2j * np.pi * np.outer(s1, zeta))  # (n1, K)
        b1 = (phase1 @ wG1D).real
        b2 = (np.exp(2j * np.pi * np.outer(s2, zeta)) @ wG2D).real

        # Template-template cross term:
        #   A12(v1, v2) = Re sum_k w conj(G1) G2 exp(-2j pi zeta (s2 - s1))
        wG1G2 = w * np.conj(G1) * G2
        em2 = np.exp(-2j * np.pi * np.outer(s2, zeta))  # (n2, K)
        tmp = wG1G2[None, :] * phase1  # (n1, K): carries exp(+2j pi zeta s1)
        A12 = (tmp @ em2.T).real  # (n1, n2)

    return {"S11": S11, "S22": S22, "SD": SD, "b1": b1, "b2": b2, "A12": A12}


# =============================================================================
# =============================================================================
def combine_stats(stats_list):
    """Sum sufficient statistics across windows -> one joint fit.

    Because each window's ``chi^2`` is linear in its six statistics and those
    are indexed by velocity (not Fourier mode), summing them pools the windows
    into a single fit with **shared ``(v1, v2, a1, a2)``**.  The windows may
    differ in ``dv`` / resolution / instrument.

    Two usage constraints -- neither enforced at runtime beyond a shape check:

    * **Same velocity grid.** The ``b1``/``b2``/``A12`` arrays are summed
      element-wise, so every window must have been evaluated on the *same*
      ``v1_grid``/``v2_grid`` (identical velocity nodes).
    * **Same epoch.** All pooled windows must share one true ``(v1, v2)``.  Do
      NOT combine windows from different epochs -- orbital motion changes the
      velocities, so a shared-``v`` fit is then meaningless.

    Note also that a *single* ``(a1, a2)`` is shared across all pooled windows.
    For chemically-distinct lines (e.g. an absorption primary and an emission
    secondary) the light/line-strength ratio is line-dependent, so one global
    amplitude pair is a modelling approximation, not an exact description.
    """
    stats_list = list(stats_list)
    if not stats_list:
        raise ValueError("combine_stats needs at least one window")

    shape = stats_list[0]["A12"].shape
    for I, st in enumerate(stats_list):
        if st["A12"].shape != shape:
            raise ValueError(
                "all windows must be evaluated on the same velocity grid "
                f"(window 0 has A12 {shape}, window {I} has {st['A12'].shape})")

    keys = ("S11", "S22", "SD", "b1", "b2", "A12")
    return {k: sum(st[k] for st in stats_list) for k in keys}


# =============================================================================
# =============================================================================
def profile_from_stats(stats, v1_grid, v2_grid, nonneg=False):
    """Profile/marginalise the amplitudes from (combined) statistics -> logL maps.

    Parameters
    ----------
    stats : dict
        Output of :func:`window_stats` or :func:`combine_stats`.
    v1_grid, v2_grid : ndarray
        The trial-velocity grids (km/s; echoed out and used for the peak).
    nonneg : bool
        If True, constrain both amplitudes to be non-negative (physical light
        fractions) via a per-node 2-variable NNLS.  This suppresses the sign
        flips that let an under-constrained component chase noise.  In this mode
        ``logL_marginal`` is set equal to ``logL_profile``: the exact marginal
        would be an orthant-truncated Gaussian integral, so this drops *both*
        the ``-1/2 ln|det|`` width term and the truncation term ``ln Phi_N``
        (not attempted).  Velocities come from the profile peak regardless.

    Returns
    -------
    dict
        ``logL_profile``, ``logL_marginal``, ``a1``, ``a2``, ``chi2`` : (n1, n2)
        maps; ``v1_grid``, ``v2_grid``; and ``best``, the ``logL_profile``
        argmax as ``{'v1', 'v2', 'a1', 'a2', 'chi2'}`` (km/s / dimensionless).
        The reported peak is always the *profile* argmax -- the marginal peak
        can differ by the ``det``-gradient.
    """
    S11, S22, SD = stats["S11"], stats["S22"], stats["SD"]
    b1g = stats["b1"][:, None]  # (n1, 1)
    b2g = stats["b2"][None, :]  # (1, n2)
    A12 = stats["A12"]  # (n1, n2)

    # det >= 0 by Cauchy-Schwarz on the weighted inner product; floor it to a
    # small *positive* value so a near-degenerate node gives large-but-finite
    # amplitudes rather than sign-flipped ones from a tiny negative det.
    det = np.maximum(S11 * S22 - A12 ** 2, 1e-30)
    a1 = (S22 * b1g - A12 * b2g) / det
    a2 = (S11 * b2g - A12 * b1g) / det

    if nonneg:
        a1, a2 = a1 + 0.0 * b2g, a2 + 0.0 * b1g  # broadcast to (n1, n2)
        infeasible = (a1 < 0) | (a2 < 0)
        # Edge solutions (one amplitude clamped to zero).
        a2_e1 = np.clip(b2g / S22, 0.0, None) + 0.0 * b1g  # a1 = 0
        chi_e1 = SD - 2.0 * a2_e1 * b2g + a2_e1 ** 2 * S22
        a1_e2 = np.clip(b1g / S11, 0.0, None) + 0.0 * b2g  # a2 = 0
        chi_e2 = SD - 2.0 * a1_e2 * b1g + a1_e2 ** 2 * S11
        take_e1 = chi_e1 <= chi_e2
        a1_edge = np.where(take_e1, 0.0, a1_e2)
        a2_edge = np.where(take_e1, a2_e1, 0.0)
        a1 = np.where(infeasible, a1_edge, a1)
        a2 = np.where(infeasible, a2_edge, a2)

    chi2 = SD - 2.0 * (b1g * a1 + b2g * a2) + (
        a1 ** 2 * S11 + 2.0 * a1 * a2 * A12 + a2 ** 2 * S22
    )
    logL_profile = -0.5 * chi2
    if nonneg:
        logL_marginal = logL_profile.copy()
    else:
        logL_marginal = logL_profile - 0.5 * np.log(np.abs(det))

    i, j = np.unravel_index(np.argmax(logL_profile), logL_profile.shape)
    best = {
        "v1": float(v1_grid[i]),
        "v2": float(v2_grid[j]),
        "a1": float(a1[i, j]),
        "a2": float(a2[i, j]),
        "chi2": float(chi2[i, j]),
    }
    return {
        "logL_profile": logL_profile,
        "logL_marginal": logL_marginal,
        "a1": a1,
        "a2": a2,
        "chi2": chi2,
        "v1_grid": np.asarray(v1_grid, float),
        "v2_grid": np.asarray(v2_grid, float),
        "best": best,
    }


# =============================================================================
# =============================================================================
def crlb_from_map(logL, v1_grid, v2_grid, half=3):
    """Cramer-Rao velocity errors from the curvature of the logL surface.

    Fits a local 2-D quadratic to ``logL`` in a ``(2*half+1)`` neighbourhood of
    its peak.  For a Gaussian likelihood the negative Hessian of the log-
    likelihood is the Fisher information, and its inverse is the parameter
    covariance.  Because ``logL`` is the *profile* over the amplitudes, the
    resulting ``(v1, v2)`` covariance already accounts for the nuisance
    amplitudes (profile = full CRLB for Gaussian-linear nuisances).  The
    amplitudes are linear only at fixed velocity, so this is a *local* CRLB,
    valid near an isolated, well-resolved peak; it degrades for blended or
    near-degenerate peaks (``A12^2 -> S11*S22``, ``det -> 0``) and returns
    ``nan`` when the local curvature is not positive definite.

    The absolute scale is only as good as the noise ``sigma`` fed to the fit; if
    the reduced chi^2 differs from 1, multiply the returned sigmas by
    ``sqrt(chi2_red)``.

    Parameters
    ----------
    logL : ndarray (n1, n2)
        Profile log-likelihood surface (``res['logL_profile']``).
    v1_grid, v2_grid : ndarray
        The trial-velocity grids (km/s).
    half : int
        Half-width (in grid nodes) of the local quadratic patch.

    Returns
    -------
    dict
        ``sigma_v1``, ``sigma_v2`` (km/s), ``rho`` (v1-v2 correlation), ``cov``
        (2x2 covariance, km^2/s^2), ``peak`` (the argmax grid node, km/s) and
        ``peak_refined`` (the stationary point of the same local quadratic --
        a sub-grid peak that removes the +/- step/2 quantisation of the argmax,
        clipped to one grid step, and equal to ``peak`` when the curvature is
        not positive definite).  The sigmas are ``nan`` if the local curvature
        is not positive definite (e.g. a peak on a grid edge or an amplitude
        boundary).
    """
    i, j = np.unravel_index(np.argmax(logL), logL.shape)
    i0, i1 = max(0, i - half), min(logL.shape[0], i + half + 1)
    j0, j1 = max(0, j - half), min(logL.shape[1], j + half + 1)

    dx = v1_grid[i0:i1] - v1_grid[i]
    dy = v2_grid[j0:j1] - v2_grid[j]
    X, Y = np.meshgrid(dx, dy, indexing="ij")
    x, y, z = X.ravel(), Y.ravel(), logL[i0:i1, j0:j1].ravel()

    design = np.column_stack([np.ones_like(x), x, y, x ** 2, x * y, y ** 2])
    coef, *_ = np.linalg.lstsq(design, z, rcond=None)
    _, p1, p2, p3, p4, p5 = coef

    fisher = -np.array([[2.0 * p3, p4], [p4, 2.0 * p5]])  # = -Hessian(logL)
    peak = (float(v1_grid[i]), float(v2_grid[j]))
    if fisher[0, 0] <= 0 or np.linalg.det(fisher) <= 0:
        return dict(sigma_v1=np.nan, sigma_v2=np.nan, rho=np.nan,
                    cov=np.full((2, 2), np.nan), peak=peak, peak_refined=peak)

    cov = np.linalg.inv(fisher)
    sv1, sv2 = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])

    # Stationary point of the quadratic: grad = (p1, p2) + H.d = 0 with
    # H = -fisher, so d = fisher^{-1} (p1, p2). Clip to one grid step --
    # a well-resolved peak refines within its own cell; anything larger
    # means the quadratic is extrapolating and the node is more honest.
    delta = np.linalg.solve(fisher, np.array([p1, p2]))
    step1 = float(np.median(np.abs(np.diff(v1_grid[i0:i1])))) if i1 - i0 > 1 else 0.0
    step2 = float(np.median(np.abs(np.diff(v2_grid[j0:j1])))) if j1 - j0 > 1 else 0.0
    d1 = float(np.clip(delta[0], -step1, step1))
    d2 = float(np.clip(delta[1], -step2, step2))
    peak_refined = (peak[0] + d1, peak[1] + d2)

    return dict(sigma_v1=float(sv1), sigma_v2=float(sv2),
                rho=float(cov[0, 1] / (sv1 * sv2)), cov=cov, peak=peak,
                peak_refined=peak_refined)


# =============================================================================
# =============================================================================
def todcor_map(data_g, g1, g2, dv, v1_grid, v2_grid, sigma=1.0, zeta_max=None,
               nonneg=False):
    """Single-window TODCOR log-likelihood map (convenience wrapper).

    Equivalent to ``profile_from_stats(window_stats(...), v1_grid, v2_grid)``.
    See :func:`window_stats` and :func:`profile_from_stats`.  For several
    line/order/instrument windows of *one epoch*, accumulate with
    :func:`combine_stats` before profiling (never across epochs).
    """
    stats = window_stats(
        data_g, g1, g2, dv, v1_grid, v2_grid, sigma=sigma, zeta_max=zeta_max
    )
    return profile_from_stats(stats, v1_grid, v2_grid, nonneg=nonneg)
