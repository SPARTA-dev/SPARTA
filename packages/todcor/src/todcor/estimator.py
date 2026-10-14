#                   ------------------------------------
#                     todcor/estimator.py (TODCOR core)
#                   ------------------------------------
# The Fourier-domain TODCOR estimator: pure functions on plain numpy arrays,
# no astropy Quantity, no I/O. The forward model lives in simulate.py and the
# shared grid/FFT/broadening primitives in unicor.dsp; SPARTA's Spectrum/Template
# orchestration (sparta.ccf2d.CCF2d) is a layer on top.
#
# Conventions: velocities and the log-lambda shift s in km/s; zeta (the Fourier
# conjugate) in s/km; spectra on a uniform log-lambda grid of step dv (km/s);
# residual flux g = flux - continuum. Velocities are 'cz' (lambda_obs/lambda_rest
# = 1 + v/c) unless velocity_convention='relativistic' (unicor.dsp.doppler_factor).
#
# Reference: Zucker, S. & Mazeh, T. 1994, ApJ 420, 806 -- "Study of
# spectroscopic binaries with TODCOR. I. A new two-dimensional correlation
# algorithm."
#
# Dependencies: numpy, unicor.dsp.
"""The Fourier-domain TODCOR estimator (the derivation).

Method
------
A composite, continuum-normalised spectrum of a double-lined system is
modelled, in the residual flux ``g = flux - 1``, as the sum of two
Doppler-shifted, scaled template spectra::

    g(u) = a1 * g1(u - s1) + a2 * g2(u - s2)

where ``u`` is a velocity coordinate (km/s) on a uniform log-lambda grid and
``s_i = c * ln(lambda_obs / lambda_rest)`` is the shift of component *i*:
``c * ln(1 + v_i / c)`` in the default 'cz' convention, ``c * artanh(v_i / c)``
in the 'relativistic' one (:func:`unicor.dsp.shift_kms`).  On the log-lambda
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
:func:`unicor.dsp.parseval_weights`) and shifts ``s_i`` (above)::

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

Oversampled grids and the noise budget
--------------------------------------
Resampling a spectrum onto a grid ``M = dv_native / dv`` times finer adds
samples but no information: ~M neighbouring analysis pixels share each native
pixel's noise.  White Parseval weights would count the Fisher information M
times and shrink the CRLB by ``sqrt(M)``.  ``oversampling=M`` divides the
weights by M (equivalently ``sigma -> sigma * sqrt(M)``), which restores the
information of the native pixels for any line resolved by the native sampling.

Each window also reports ``nu``, the expected chi^2 of its pure-noise residual
(``sum_k w_k E|N_k|^2``), i.e. its effective number of noise degrees of
freedom.  For white noise on the analysis grid ``nu`` is the number of retained
Fourier modes; on an oversampled grid it is ``~ kappa * n / M``, where
``kappa < 1`` is the interpolator's noise-variance factor (2/3 for linear,
~0.87 for a cubic spline) -- the resampled noise is smoother, not just
correlated.  On a *coarser* grid the resampled noise is coloured too, and
its variance depends on the sub-pixel phases at which the analysis nodes
sample the native pixels.  Pass the resampled-noise spectrum
(:func:`resampled_noise_psd`, times ``mean(taper**2)`` for an apodised window)
as ``noise_psd`` and ``chi2_red = chi2 / (sum nu - 4)`` is ~1 for honest
noise; then ``sigma *= sqrt(chi2_red)`` is a calibrated noise estimate.  For
windows of different noise, each window's share of the joint ``chi^2``
(:func:`chi2_from_stats` at the joint amplitudes) over its own ``nu`` estimates
its own noise, and :func:`combine_stats` re-weights the windows by it.

Assumptions (exact only under these)
------------------------------------
* **White native noise** ``sigma`` per *native* pixel, resampled by an
  interpolator (any scheme linear in the flux) whose impulse responses are
  known (``noise_psd``).  The Tukey
  taper (:func:`unicor.dsp.tukey_window`) makes the noise heteroscedastic
  (``sigma_n = w_n sigma``); only its mean effect enters ``nu``.  The weights
  stay white, so an interpolator that attenuates in-band noise (linear
  interpolation's passband droop) leaves the CRLB mildly conservative.
* **Flat R^2 prior** on the amplitudes for the marginal map (above).
* **Shared velocity grid** for :func:`combine_stats` (below).
"""

import numpy as np

from unicor.dsp import parseval_weights, rfft_positive, shift_kms


# =============================================================================
# =============================================================================
def window_stats(data_g, g1, g2, dv, v1_grid, v2_grid, sigma=1.0, zeta_max=None,
                 oversampling=1.0, noise_psd=None, velocity_convention='cz'):
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
        Velocity step of the grid (km/s): ``c`` times its *actual* ln(lambda)
        step (:func:`unicor.dsp.log_grid_step`), which is what a shift ``s``
        is measured in -- a nominal ``dv`` for a grid built as
        ``ln(1 + dv/c)`` is off by ``dv/2c`` relative.
    v1_grid, v2_grid : ndarray
        Trial radial velocities of each component (km/s).
    sigma : float
        Per-pixel noise of the data (same flux units) on its *native*
        sampling, assumed white and homoscedastic there.
    zeta_max : float or None
        Optional low-pass: drop Fourier modes above ``zeta_max`` (s/km), e.g.
        to exclude scales finer than the instrumental resolution.
    oversampling : float
        Analysis pixels per independent (native) pixel, ``dv_native / dv``,
        clipped at 1 by the caller (dimensionless, >= 1). The weights are
        divided by it, so each native pixel's information is counted once.
        Default 1: the analysis grid is the native sampling (no change).
    noise_psd : ndarray or None
        Expected noise power of ``data_g`` per Fourier mode, relative to white
        noise of variance ``sigma**2`` on the analysis grid, aligned with
        :func:`unicor.dsp.rfft_positive` (length ``n // 2``; see
        :func:`resampled_noise_psd`). Used only for the noise budget ``nu``.
        Default None: white noise on the analysis grid (all ones).
    velocity_convention : {'cz', 'relativistic'}
        Doppler convention of ``v1_grid``/``v2_grid``
        (:func:`unicor.dsp.doppler_factor`). Default 'cz'.

    Returns
    -------
    dict
        ``S11``, ``S22``, ``SD`` : scalar template/data weighted powers.
        ``b1`` (n1,), ``b2`` (n2,) : data-template cross-terms vs ``v1``/``v2``.
        ``A12`` (n1, n2) : template-template cross-term vs ``(v1, v2)``.
        ``nu`` : expected chi^2 of the window's pure-noise residual -- its
        effective noise degrees of freedom (dimensionless).
    """
    data_g = np.asarray(data_g, float)
    n = len(data_g)
    if len(g1) != n or len(g2) != n:
        raise ValueError("data_g, g1 and g2 must share one grid "
                         f"(got {n}, {len(g1)}, {len(g2)})")
    if not oversampling >= 1.0:
        raise ValueError(f"oversampling must be >= 1, got {oversampling}")
    if noise_psd is None:
        noise_psd = np.ones(n // 2)
    noise_psd = np.asarray(noise_psd, float)
    if noise_psd.shape != (n // 2,):
        raise ValueError(f"noise_psd must have length n // 2 = {n // 2} "
                         f"(one value per positive Fourier mode), got {noise_psd.shape}")

    D, zeta = rfft_positive(data_g, dv)
    G1 = np.fft.rfft(np.asarray(g1, float))[1:]
    G2 = np.fft.rfft(np.asarray(g2, float))[1:]
    w = parseval_weights(n, sigma=sigma) / oversampling

    # For even N the Nyquist coefficient of a real signal must be real; the
    # phase-shift model (via irfft in `model_flux`) enforces that, but the
    # complex phase factors used for b1/b2/A12 do not. Dropping the single
    # Nyquist mode keeps the fit self-consistent and preserves the exact
    # Parseval identity for the retained modes (its power is negligible).
    if n % 2 == 0:
        D, G1, G2, zeta, w = D[:-1], G1[:-1], G2[:-1], zeta[:-1], w[:-1]
        noise_psd = noise_psd[:-1]

    if zeta_max is not None:
        keep = zeta <= zeta_max
        D, G1, G2, zeta, w = D[keep], G1[keep], G2[keep], zeta[keep], w[keep]
        noise_psd = noise_psd[keep]

    # E|N_k|^2 = n sigma^2 psd_k for the retained modes; the dropped ones (DC,
    # Nyquist, above zeta_max) never enter chi^2 and so carry no dof.
    nu = float(np.sum(w * n * sigma ** 2 * noise_psd))

    s1 = shift_kms(np.asarray(v1_grid, float), velocity_convention)  # (n1,)
    s2 = shift_kms(np.asarray(v2_grid, float), velocity_convention)  # (n2,)

    S11 = np.sum(w * np.abs(G1) ** 2)
    S22 = np.sum(w * np.abs(G2) ** 2)
    SD = np.sum(w * np.abs(D) ** 2)

    # The phase factors are unit-modulus and the weighted spectra are finite, so
    # the sums below cannot legitimately overflow or divide by zero -- but
    # Apple's Accelerate BLAS raises spurious FP-exception flags during complex
    # matmul under numpy>=2 (same issue as sparta.pdc._gram). Silence them.
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

    return {"S11": S11, "S22": S22, "SD": SD, "b1": b1, "b2": b2, "A12": A12,
            "nu": nu}


# =============================================================================
# =============================================================================
def resampled_noise_psd(responses, n, ratio, weights=None):
    """Noise spectrum of white native noise after resampling.

    Resampling is linear in the flux, so white noise ``sigma`` on the native
    pixels becomes ``sum_j eps_j h_j`` on the analysis grid, where ``h_j`` is
    the response to a unit impulse in native pixel ``j``.  Its expected power
    per Fourier mode is ``sigma^2 * (n / ratio) * mean_j |H_j,k|^2``
    (``n / ratio`` native pixels in the window), which this returns relative
    to white noise ``sigma^2`` on the analysis grid -- the ``noise_psd`` of
    :func:`window_stats`.  Its mean over modes is the interpolator's
    noise-variance factor ``kappa``.  Valid for any ``ratio``: finer
    (``ratio > 1``) or coarser (``ratio < 1``) analysis grids.

    ``|H_j|^2`` depends on where native pixel ``j`` falls between two analysis
    nodes (its sub-pixel phase), so the mean over native pixels needs
    responses at representative phases: e.g. one per phase bin, weighted by
    the number of native pixels in that bin.  Equal weights over uniformly
    spaced phases model grids with no fixed phase relation; a grid commensurate
    with the native one (e.g. exactly twice as coarse) has only a few phases,
    and its weights must say so.

    Parameters
    ----------
    responses : sequence of ndarray
        Analysis-grid responses to unit impulses in single native pixels, each
        cut to its support (analysis pixels; dimensionless).
    n : int
        Length of the analysis window (pixels).
    ratio : float
        Native velocity step over analysis step, ``dv_native / dv``
        (dimensionless).
    weights : array_like or None
        Non-negative relative frequency of each response among the window's
        native pixels (dimensionless; normalised here).  Default None: equal
        weights.

    Returns
    -------
    ndarray (n // 2,)
        Relative noise power per positive Fourier mode, aligned with
        :func:`unicor.dsp.rfft_positive` (white noise = 1).
    """
    responses = [np.asarray(h, float) for h in responses]
    if not responses:
        raise ValueError("resampled_noise_psd needs at least one impulse response")
    if not ratio > 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    weights = np.ones(len(responses)) if weights is None else np.asarray(weights, float)
    if weights.shape != (len(responses),) or not (np.all(weights >= 0) and weights.sum() > 0):
        raise ValueError("weights must be one non-negative value per response, "
                         f"not all zero; got {weights}")
    responses = [h for h, wt in zip(responses, weights) if wt > 0]
    weights = weights[weights > 0]
    longest = max(len(h) for h in responses)
    if longest > n:
        raise ValueError(f"an impulse response ({longest} pixels) is longer than "
                         f"the window ({n} pixels)")
    # |rfft(h, n)|^2 is the transform of h's autocorrelation wrapped modulo n,
    # so average the (short) autocorrelations and transform once instead of
    # paying one length-n FFT per impulse. Padding to 2 * longest keeps the
    # short autocorrelation free of wrap-around.
    padded = np.array([np.pad(h, (0, longest - len(h))) for h in responses])
    m = 2 * longest
    power_short = np.average(np.abs(np.fft.rfft(padded, n=m, axis=1)) ** 2,
                             axis=0, weights=weights)
    acf = np.fft.irfft(power_short, n=m)
    lags = np.concatenate([np.arange(longest), np.arange(-longest, 0)])
    circ = np.zeros(n)
    np.add.at(circ, lags % n, acf)
    power = np.fft.rfft(circ).real[1:]
    return power / ratio


# =============================================================================
# =============================================================================
def combine_stats(stats_list, weights=None):
    """Sum sufficient statistics across windows -> one joint fit.

    Because each window's ``chi^2`` is linear in its six statistics and those
    are indexed by velocity (not Fourier mode), summing them pools the windows
    into a single fit with **shared ``(v1, v2, a1, a2)``**.  The windows may
    differ in ``dv`` / resolution / instrument.

    The six statistics are linear in the Parseval weights ``1/sigma^2``, so
    scaling a window's statistics by ``(sigma / sigma_new)^2`` is the same as
    recomputing them at noise ``sigma_new`` -- a window can be re-weighted
    without touching its spectra again.  ``nu`` is not scaled: it is the
    expected ``chi^2`` of the window's noise *at the noise it is weighted
    with*, whatever that noise is.

    Parameters
    ----------
    stats_list : iterable of dict
        :func:`window_stats` outputs.
    weights : sequence of float or None
        One finite, positive factor per window on its six statistics
        (dimensionless), e.g. ``1 / sigma_i**2`` for windows computed at unit
        noise.  Default None: all ones (a plain sum).

    Returns
    -------
    dict
        The pooled ``S11``, ``S22``, ``SD``, ``b1``, ``b2``, ``A12`` and ``nu``.

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

    if weights is None:
        weights = np.ones(len(stats_list))
    weights = np.asarray(weights, float)
    if weights.shape != (len(stats_list),) or not np.all(np.isfinite(weights) & (weights > 0)):
        raise ValueError(f"weights must be one finite, positive value per window "
                         f"({len(stats_list)}), got {weights}")

    combined = {k: sum(wt * st[k] for wt, st in zip(weights, stats_list))
                for k in ("S11", "S22", "SD", "b1", "b2", "A12")}
    combined["nu"] = sum(st["nu"] for st in stats_list)
    return combined


# =============================================================================
# =============================================================================
def chi2_from_stats(stats, a1, a2):
    """The weighted ``chi^2`` of (one window's or combined) statistics at given
    amplitudes, on every ``(v1, v2)`` node.

    Evaluated at the *joint* best-fit amplitudes, one window's value is its
    share of the joint ``chi^2`` (the shares sum to it) -- its own residual
    power, from which its noise can be estimated.

    Parameters
    ----------
    stats : dict
        Output of :func:`window_stats` or :func:`combine_stats`.
    a1, a2 : float or ndarray (n1, n2)
        Amplitudes of the two components (dimensionless).

    Returns
    -------
    ndarray (n1, n2)
        ``SD - 2(a1 b1 + a2 b2) + a1^2 S11 + 2 a1 a2 A12 + a2^2 S22``
        (dimensionless; ``chi^2`` at the noise the statistics carry).
    """
    b1 = stats["b1"][:, None]
    b2 = stats["b2"][None, :]
    return stats["SD"] - 2.0 * (b1 * a1 + b2 * a2) + (
        a1 ** 2 * stats["S11"] + 2.0 * a1 * a2 * stats["A12"] + a2 ** 2 * stats["S22"]
    )


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
        argmax as ``{'v1', 'v2', 'a1', 'a2', 'chi2', 'chi2_red'}`` (km/s /
        dimensionless), where ``chi2_red = chi2 / (nu - 4)`` (four fitted
        parameters; ``nan`` if ``nu <= 4``).  The reported peak is always the
        *profile* argmax -- the marginal peak can differ by the
        ``det``-gradient.
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

    chi2 = chi2_from_stats(stats, a1, a2)
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
        "chi2_red": float(chi2[i, j] / (stats["nu"] - 4.0))
        if stats["nu"] > 4.0 else np.nan,
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

    The patch must be small compared with the width of the peak.  ``logL``
    follows the *squared* correlation, so its peak is ~``sigma_ccf / sqrt(2)``
    wide; a patch spanning a sizeable fraction of that picks up the quartic
    fall-off, which biases the curvature and -- with the argmax off-centre in
    the patch -- the refined peak (several CRLB sigma for 8 km/s lines on a
    +/-3 km/s patch).  Evaluate a coarse map, then re-evaluate the statistics
    on a fine local grid around its argmax (as ``sparta.ccf2d.CCF2d`` does)
    and call this on the fine map.

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
        not positive definite), plus ``on_edge`` (True when the argmax sits on
        the boundary of the velocity grid).  The sigmas are ``nan`` if the local
        curvature is not positive definite (e.g. an amplitude boundary) or the
        peak is on a grid edge, where the true maximum may lie outside the grid
        and the one-sided patch would extrapolate.
    """
    i, j = np.unravel_index(np.argmax(logL), logL.shape)
    peak = (float(v1_grid[i]), float(v2_grid[j]))
    on_edge = i in (0, logL.shape[0] - 1) or j in (0, logL.shape[1] - 1)
    if on_edge:
        return dict(sigma_v1=np.nan, sigma_v2=np.nan, rho=np.nan,
                    cov=np.full((2, 2), np.nan), peak=peak, peak_refined=peak,
                    on_edge=True)
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
    if fisher[0, 0] <= 0 or np.linalg.det(fisher) <= 0:
        return dict(sigma_v1=np.nan, sigma_v2=np.nan, rho=np.nan,
                    cov=np.full((2, 2), np.nan), peak=peak, peak_refined=peak,
                    on_edge=False)

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
                peak_refined=peak_refined, on_edge=False)


# =============================================================================
# =============================================================================
def todcor_map(data_g, g1, g2, dv, v1_grid, v2_grid, sigma=1.0, zeta_max=None,
               nonneg=False, oversampling=1.0, noise_psd=None, velocity_convention='cz'):
    """Single-window TODCOR log-likelihood map (convenience wrapper).

    Equivalent to ``profile_from_stats(window_stats(...), v1_grid, v2_grid)``.
    See :func:`window_stats` and :func:`profile_from_stats`.  For several
    line/order/instrument windows of *one epoch*, accumulate with
    :func:`combine_stats` before profiling (never across epochs).
    """
    stats = window_stats(
        data_g, g1, g2, dv, v1_grid, v2_grid, sigma=sigma, zeta_max=zeta_max,
        oversampling=oversampling, noise_psd=noise_psd,
        velocity_convention=velocity_convention,
    )
    return profile_from_stats(stats, v1_grid, v2_grid, nonneg=nonneg)
