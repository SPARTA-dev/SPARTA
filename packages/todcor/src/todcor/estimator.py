#                   ------------------------------------
#                     todcor/estimator.py (TODCOR core)
#                   ------------------------------------
# The TODCOR estimator: pure functions on plain numpy arrays,
# no astropy Quantity, no I/O. The forward model lives in simulate.py and the
# shared grid/FFT/broadening primitives in unicor.dsp; SPARTA's Spectrum/Template
# orchestration (sparta.ccf2d.CCF2d) is a layer on top.
#
# Conventions: velocities in km/s; spectra on a uniform log-lambda grid of step
# dv (km/s); flux minus its median, a free offset per window profiled out. Velocities are 'cz' (lambda_obs/lambda_rest
# = 1 + v/c) unless velocity_convention='relativistic' (unicor.dsp.doppler_factor).
#
# Reference: Zucker, S. & Mazeh, T. 1994, ApJ 420, 806 -- "Study of
# spectroscopic binaries with TODCOR. I. A new two-dimensional correlation
# algorithm."
#
# Dependencies: numpy, unicor (dsp, and ccf.check_output).
#
# The per-window atoms at the end (prepare_window, refine_grid, joint_fit,
# window_noise) are the steps of SPARTA's CCF2d on plain arrays: chaining them
# by hand reproduces its fit bit for bit.
"""The TODCOR estimator (the derivation).

Method
------
A composite, continuum-normalised spectrum of a double-lined system is
modelled, on the observed pixels of one window (an echelle order), as two
Doppler-shifted, scaled templates plus a free continuum offset::

    d(u) = a1 * g1(u - s1) + a2 * g2(u - s2) + c

where ``u`` is a velocity coordinate (km/s) on a uniform log-lambda grid and
``s_i = c * ln(lambda_obs / lambda_rest)`` is the shift of component *i*:
``c * ln(1 + v_i / c)`` in the default 'cz' convention, ``c * artanh(v_i / c)``
in the 'relativistic' one (:func:`unicor.dsp.shift_kms`).  On the log-lambda
grid a Doppler shift is a translation; the templates are shifted by a Fourier
phase factor (the shift theorem) on the window's grid *extended* by the
largest trial shift on each side, so every shift reads a template's real
neighbouring spectrum and nothing wraps around into the observed pixels
(:func:`prepare_window`). No taper is applied.  The amplitudes ``(a1, a2)``
and the offset ``c`` enter linearly, so for every trial ``(v1, v2)`` they are
solved in closed form by weighted least squares; profiling them out gives two
equivalent surfaces over ``(v1, v2)``: the Gaussian log-likelihood
``-chi^2/2`` and the classic TODCOR two-dimensional correlation ``R``
(Zucker & Mazeh 1994), the fraction of the data's power the best-fit pair of
templates explains (below).

Sufficient statistics
---------------------
The chi^2 is a sum over the observed pixels with weight ``w``.  Projecting out
the offset (each sum centred, as a Pearson correlation is) leaves, at any
trial ``(v1, v2, a1, a2)``, a quadratic form in six statistics
(:func:`window_stats`)::

    SD                                                          (scalar)
    S11(v1), S22(v2)       template powers over the window       (n1,), (n2,)
    b1(v1),  b2(v2)        data-template cross terms             (n1,), (n2,)
    A12(v1, v2)            template-template cross term          (n1, n2)

    chi^2 = SD - 2(a1 b1 + a2 b2) + a1^2 S11 + 2 a1 a2 A12 + a2^2 S22

which :func:`profile_from_stats` minimises.  The template powers depend on the
velocity because the part of the template inside the window moves; ``A12``
depends on both velocities, not only on their difference.

Profiling and marginalising the amplitudes
------------------------------------------
The 2x2 normal equations ``N a = b`` with ``N = [[S11, A12], [A12, S22]]`` and
``b = (b1, b2)`` solve in closed form (``det = S11*S22 - A12^2``)::

    a1 = (S22 b1 - A12 b2) / det        a2 = (S11 b2 - A12 b1) / det

Substituting the profile amplitudes gives ``logL_profile = -chi^2/2``.  A
component whose template has no lines in the window (zeroed by
:func:`prepare_window`) gets amplitude 0; there, and where the templates are
degenerate (``det`` below 1e-10 of ``S11 S22``), the better single-component
fit is used.

The two-dimensional correlation
-------------------------------
At the profile amplitudes ``chi^2_min = SD - b^T N^-1 b``, so::

    R^2 = 1 - chi^2_min / SD = b^T N^-1 b / SD
        = (c1^2 - 2 c1 c2 c12 + c2^2) / (1 - c12^2)

with the normalised correlations ``c1 = b1 / sqrt(S11 SD)``,
``c2 = b2 / sqrt(S22 SD)`` and ``c12 = A12 / sqrt(S11 S22)`` -- exactly
Zucker & Mazeh's (1994) ``R`` with the light ratio fitted at every node; with
one template, ``c1`` is UNICOR's per-lag Pearson CCF.  ``R`` is in [0, 1]
(1 for a noiseless, perfectly matched composite).  At fixed ``SD``,
``logL_profile = -(SD/2) (1 - R^2)``: the two surfaces share their argmax.

*Marginalising* the amplitudes under a flat (improper) prior on
``(a1, a2) in R^2`` adds ``-1/2 ln|det|``::

    logL_marginal = logL_profile - 1/2 ln|det|

When ``nonneg=True`` the amplitudes are constrained to ``a >= 0``
(2-variable NNLS); the exact marginal is then an orthant-truncated Gaussian
integral, which is not evaluated: ``logL_marginal`` is set equal to
``logL_profile``.

Combining windows
-----------------
The statistics are indexed by *velocity*, so windows on different grids /
instruments / resolutions evaluated on one ``(v1, v2)`` grid combine by
**summation** (:func:`combine_stats`) into one joint fit with **shared
``(v1, v2, a1, a2)``**; or each window keeps its own ``(a1, a2)`` -- its own
light ratio -- and the windows' profiled chi^2 maps are summed
(:func:`profile_per_window`, ``joint_fit(amplitudes='per_order')``).  Both are
valid only *within a single epoch* -- pooled windows must share one true
``(v1, v2)``; do NOT combine across epochs.

Oversampled grids and the noise budget
--------------------------------------
Resampling a spectrum onto a grid ``M = dv_native / dv`` times finer adds
samples but no information.  ``oversampling=M`` divides the pixel weight by M
(``w = 1 / (sigma^2 M)``), which restores the information of the native
pixels.  Each window also reports ``nu``, the expected chi^2 of its pure-noise
residual: ``n * mean(noise_psd) / M - 1`` (one for the offset), where
``noise_psd`` is the resampled noise power relative to white
(:func:`resampled_noise_psd`; ``kappa = mean(noise_psd) < 1``: the resampled
noise is smoother, not just correlated).  Then ``chi2_red = chi2 / (sum nu -
4)`` (``sum nu - 2 - 2 n_windows`` with per-window amplitudes) is ~1 for
honest noise, and each window's share of the joint chi^2 over
its own dof estimates its noise (:func:`window_noise`).

Peak and errors
---------------
A coarse map locates the peak; a fine local map (:func:`refine_grid`) is
evaluated exactly around it, and :func:`crlb_from_map` fits a degree-6
polynomial there for the maximum and its curvature (the Cramer-Rao
covariance).  A quadratic patch misread the narrow logL peak by up to several
m/s.

Assumptions (exact only under these)
------------------------------------
* **White native noise** ``sigma`` per *native* pixel, resampled by an
  interpolator (any scheme linear in the flux) whose impulse responses are
  known (``noise_psd``); only its mean enters ``nu``.  The weights stay white,
  so an interpolator that attenuates in-band noise leaves the CRLB mildly
  conservative.
* **Templates covering the window +- the largest trial shift** (checked:
  uncovered pixels are dropped by :func:`prepare_window`).
* **Flat R^2 prior** on the amplitudes for the marginal map (above).
* **Shared velocity grid** for :func:`combine_stats`.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from unicor.ccf import check_output, extend_log_grid, template_coverage
from unicor.dsp import log_grid_step, resample, shift_kms


# =============================================================================
# =============================================================================
def _shifted_rows(g_ext, n, dv, v_grid, velocity_convention, chunk=64):
    """A template shifted to each trial velocity, read on the window's pixels.

    The template lives on the window's grid extended by ``(len(g_ext) - n) // 2``
    pixels on the left (and the rest on the right), so a shift brings in its *real* neighbouring
    spectrum, not the window's other end. The shift is a Fourier phase factor
    (band-limited interpolation, exact at whole pixels); the extension's outer
    ramp keeps its wrap-around junction outside every pixel read here.

    Parameters
    ----------
    g_ext : ndarray
        Template residual on the extended grid (flux units).
    n : int
        Pixels in the window.
    dv : float
        Grid step (km/s).
    v_grid : array_like
        Trial velocities (km/s).
    velocity_convention : {'cz', 'relativistic'}
        Doppler convention of ``v_grid``.
    chunk : int
        Velocities shifted per FFT batch (memory: ``chunk * len(g_ext)``).

    Returns
    -------
    ndarray, shape (len(v_grid), n)
        Row k: the template at ``v_grid[k]`` on the window's pixels.
    """
    g_ext = np.asarray(g_ext, float)
    size = g_ext.size
    first = (size - n) // 2
    spectrum = np.fft.rfft(g_ext)
    freq = np.fft.rfftfreq(size)                                   # cycles per pixel
    pixels = shift_kms(np.asarray(v_grid, float), velocity_convention) / dv
    rows = np.empty((pixels.size, n))
    for k0 in range(0, pixels.size, chunk):
        phase = np.exp(-2j * np.pi * np.outer(pixels[k0:k0 + chunk], freq))
        rows[k0:k0 + chunk] = np.fft.irfft(spectrum[None, :] * phase, n=size,
                                           axis=1)[:, first:first + n]
    return rows


def window_stats(data_g, g1, g2, dv, v1_grid, v2_grid, sigma=1.0, oversampling=1.0,
                 noise_psd=None, velocity_convention='cz', reach=None):
    """Weighted least-squares sufficient statistics for one spectral window.

    The chi^2 is a sum over the window's *observed* pixels only. The data are
    the window's ``n`` pixels; the templates are given on the same grid
    extended by ``e`` pixels on each side (one more on the right for an odd
    length), so a template shifted by up to
    ``e`` pixels is still read from its real spectrum -- no circular wrap, no
    missing lines at the edges. A free additive offset per window (the
    continuum level) is profiled out: every statistic below is projected
    onto the complement of a constant, like a Pearson correlation. With the
    pixel weight ``w = 1 / (sigma^2 * oversampling)``, shifted templates
    ``G_i(s)`` on the window and ``m_X = w * sum(X)``, ``W0 = w * n``::

        SD        = w sum d^2 - m_d^2 / W0                       (scalar)
        S11(v1)   = w sum G1^2 - m_G1^2 / W0                     (n1,)
        S22(v2)   = w sum G2^2 - m_G2^2 / W0                     (n2,)
        b1(v1)    = w sum d G1 - m_G1 m_d / W0                   (n1,)
        b2(v2)    = w sum d G2 - m_G2 m_d / W0                   (n2,)
        A12(v1,v2)= w sum G1 G2 - m_G1 m_G2 / W0                 (n1, n2)

    and ``chi^2 = SD - 2(a1 b1 + a2 b2) + a1^2 S11 + 2 a1 a2 A12 + a2^2 S22``.
    The template powers depend on the velocity (the overlap moves); ``A12``
    depends on both velocities, not only on their difference. Windows on one
    velocity grid combine by summation (:func:`combine_stats`).

    Parameters
    ----------
    data_g : ndarray
        The window's flux (any constant level; it is profiled out), on a
        uniform log-lambda grid of step ``dv``; ``n`` pixels.
    g1, g2 : ndarray
        The two templates on that grid extended by the same ``e >= 0`` pixels
        on the left and ``e`` or ``e + 1`` on the right (length ``n + 2e`` or
        ``n + 2e + 1``), as :func:`prepare_window` makes them.
    dv : float
        Velocity step of the grid (km/s): ``c`` times its *actual* ln(lambda)
        step (:func:`unicor.dsp.log_grid_step`).
    v1_grid, v2_grid : ndarray
        Trial radial velocities of each component (km/s); their shifts must
        stay within ``reach`` pixels.
    sigma : float
        Per-pixel noise of the data (same flux units) on its *native*
        sampling, assumed white and homoscedastic there.
    oversampling : float
        Analysis pixels per independent (native) pixel, ``dv_native / dv``,
        clipped at 1 by the caller (dimensionless, >= 1). The weight is
        divided by it, so each native pixel's information is counted once.
    noise_psd : ndarray or None
        Expected noise power of ``data_g`` per Fourier mode, relative to white
        noise of variance ``sigma**2`` on the analysis grid
        (:func:`resampled_noise_psd`); only its mean enters the noise budget
        ``nu``. Default None: white noise on the analysis grid.
    velocity_convention : {'cz', 'relativistic'}
        Doppler convention of ``v1_grid``/``v2_grid``
        (:func:`unicor.dsp.doppler_factor`). Default 'cz'.
    reach : int or None
        The largest shift (pixels) the templates may be read at: a prepared
        window's ``reach`` (its extension less the ramp). Default None: the
        whole left extension ``e`` (with ``e = 0``, any shift: circular).

    Returns
    -------
    dict
        ``S11`` (n1,), ``S22`` (n2,), ``SD`` (scalar), ``b1`` (n1,), ``b2``
        (n2,), ``A12`` (n1, n2), all on the noise scale ``w``; and ``nu``,
        the expected chi^2 of the window's pure-noise residual (its noise
        degrees of freedom, dimensionless): ``n * mean(noise_psd) /
        oversampling - 1`` (one for the offset).

    Raises
    ------
    ValueError
        If a template is shorter than the window, the two templates differ
        in length, a trial shift exceeds ``reach`` (or the extension, when
        ``e > 0``), or ``oversampling < 1``.
    """
    data_g = np.asarray(data_g, float)
    n = data_g.size
    for name, g in (("g1", g1), ("g2", g2)):
        if len(g) < n:
            raise ValueError(f"{name} must be the window's grid extended on each side "
                             f"(length n + 2e or n + 2e + 1); got {len(g)} for "
                             f"{n} window pixels")
    if len(g1) != len(g2):
        raise ValueError(f"g1 and g2 must share one extended grid; got lengths "
                         f"{len(g1)} and {len(g2)}")
    if not oversampling >= 1.0:
        raise ValueError(f"oversampling must be >= 1, got {oversampling}")
    # A shift beyond the reach would read the ramp or the wrapped far end.
    limit = (len(g1) - n) // 2 if reach is None else reach
    for name, grid in (("v1_grid", v1_grid), ("v2_grid", v2_grid)):
        pixels = np.max(np.abs(shift_kms(np.asarray(grid, float), velocity_convention))) / dv
        if limit > 0 and pixels > limit:
            raise ValueError(f"{name} shifts the templates by up to {pixels:.1f} pixels, beyond "
                             f"their reach of {limit}; prepare the window with a larger "
                             "v_max")
    kappa = 1.0 if noise_psd is None else float(np.mean(noise_psd))
    w = 1.0 / (sigma ** 2 * oversampling)
    G1 = _shifted_rows(g1, n, dv, v1_grid, velocity_convention)
    G2 = _shifted_rows(g2, n, dv, v2_grid, velocity_convention)
    W0 = w * n
    m_d = w * data_g.sum()
    m1, m2 = w * G1.sum(axis=1), w * G2.sum(axis=1)
    # Apple's Accelerate BLAS raises spurious FP-exception flags in matmul
    # under numpy>=2 (as in sparta.pdc._gram); the products are finite.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        SD = w * float(data_g @ data_g) - m_d ** 2 / W0
        S11 = w * np.einsum("ij,ij->i", G1, G1) - m1 ** 2 / W0
        S22 = w * np.einsum("ij,ij->i", G2, G2) - m2 ** 2 / W0
        b1 = w * (G1 @ data_g) - m1 * m_d / W0
        b2 = w * (G2 @ data_g) - m2 * m_d / W0
        A12 = w * (G1 @ G2.T) - np.outer(m1, m2) / W0
    nu = n * kappa / oversampling - 1.0
    return {"S11": S11, "S22": S22, "SD": SD, "b1": b1, "b2": b2, "A12": A12, "nu": nu}

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
    are indexed by velocity (not pixel), summing them pools the windows
    into a single fit with **shared ``(v1, v2, a1, a2)``**.  The windows may
    differ in ``dv`` / resolution / instrument.

    The six statistics are linear in the pixel weight ``1/sigma^2``, so
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
    amplitude pair is a modelling approximation, not an exact description;
    :func:`profile_per_window` fits one ``(a1, a2)`` per window instead.
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
def _as_maps(stats):
    """The statistics broadcast to the (n1, n2) map: S11 (n1, 1), S22 (1, n2)."""
    S11 = np.asarray(stats["S11"], float)
    S22 = np.asarray(stats["S22"], float)
    S11 = S11[:, None] if S11.ndim == 1 else S11
    S22 = S22[None, :] if S22.ndim == 1 else S22
    return S11, S22, stats["b1"][:, None], stats["b2"][None, :], stats["A12"]


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
    S11, S22, b1, b2, A12 = _as_maps(stats)
    return stats["SD"] - 2.0 * (b1 * a1 + b2 * a2) + (
        a1 ** 2 * S11 + 2.0 * a1 * a2 * A12 + a2 ** 2 * S22)


def _amplitudes(stats, nonneg):
    """The best-fit amplitudes on every node (2x2 weighted least squares).

    A component whose template power is zero on a node has no lines there
    (:func:`prepare_window` zeroes a template flat to round-off) and gets
    amplitude 0. There, and where the two templates are degenerate (``det``
    below 1e-10 of ``S11 * S22``), the better single-component fit is used;
    ``nonneg`` clamps to the physical quadrant. Both tests are invariant to
    the templates' scale.
    """
    S11, S22, b1, b2, A12 = _as_maps(stats)
    lines1, lines2 = S11 > 0, S22 > 0
    scale = S11 * S22
    det = scale - A12 ** 2
    safe = lines1 & lines2 & (det > 1e-10 * scale)
    det_safe = np.where(safe, det, 1.0)
    a1 = np.where(safe, (S22 * b1 - A12 * b2) / det_safe, 0.0)
    a2 = np.where(safe, (S11 * b2 - A12 * b1) / det_safe, 0.0)
    # Single-component solutions, used where the pair is degenerate or (with
    # nonneg) where the pair leaves the physical quadrant.
    a1_only = np.where(lines1, b1 / np.where(lines1, S11, 1.0), 0.0) + 0.0 * b2
    a2_only = np.where(lines2, b2 / np.where(lines2, S22, 1.0), 0.0) + 0.0 * b1
    if nonneg:
        a1_only, a2_only = np.clip(a1_only, 0.0, None), np.clip(a2_only, 0.0, None)
    chi_1 = stats["SD"] - 2.0 * a1_only * b1 + a1_only ** 2 * S11
    chi_2 = stats["SD"] - 2.0 * a2_only * b2 + a2_only ** 2 * S22
    take_1 = chi_1 <= chi_2
    single_a1 = np.where(take_1, a1_only, 0.0)
    single_a2 = np.where(take_1, 0.0, a2_only)
    use_single = ~safe | ((a1 < 0) | (a2 < 0) if nonneg else False)
    a1 = np.where(use_single, single_a1, a1) + 0.0 * b2
    a2 = np.where(use_single, single_a2, a2) + 0.0 * b1
    return a1, a2


def profile_from_stats(stats, v1_grid, v2_grid, nonneg=False, output='ccf'):
    """Profile/marginalise the amplitudes from (combined) statistics -> R and logL maps.

    Parameters
    ----------
    stats : dict
        Output of :func:`window_stats` or :func:`combine_stats`.
    v1_grid, v2_grid : ndarray
        The trial-velocity grids (km/s; echoed out and used for the peak).
    nonneg : bool
        If True, constrain both amplitudes to be non-negative (physical light
        fractions) via a per-node 2-variable NNLS. In this mode
        ``logL_marginal`` is set equal to ``logL_profile`` (the exact marginal
        would be an orthant-truncated Gaussian integral). Velocities come from
        the profile peak regardless.
    output : {'ccf', 'loglike'}
        Which surface is the primary result, ``'map'``: the two-dimensional
        correlation ``R`` (default) or ``logL_profile``. Both are always
        returned; they share their argmax, so ``best`` does not depend on it.

    Returns
    -------
    dict
        ``R``, ``logL_profile``, ``logL_marginal``, ``a1``, ``a2``, ``chi2`` :
        (n1, n2) maps, where ``R = sqrt(1 - chi2 / SD)`` is the
        two-dimensional correlation (dimensionless, in [0, 1]); ``map``, the
        one of ``R`` / ``logL_profile`` that ``output`` selects, and
        ``output``; ``v1_grid``, ``v2_grid``; and ``best``, the
        ``logL_profile`` argmax as ``{'v1', 'v2', 'a1', 'a2', 'chi2',
        'chi2_red', 'R'}`` (km/s / dimensionless), with ``chi2_red = chi2 /
        (nu - 4)`` (``nan`` if ``nu <= 4``).

    Raises
    ------
    ValueError
        If ``output`` is unknown, or the data carry no power (``SD <= 0``),
        which leaves ``R`` undefined.
    """
    check_output(output)
    SD = stats["SD"]
    if not SD > 0:
        raise ValueError(f"the data have no power in the window (SD = {SD}); "
                         "there is nothing to correlate")
    a1, a2 = _amplitudes(stats, nonneg)
    chi2 = chi2_from_stats(stats, a1, a2)
    logL_profile = -0.5 * chi2
    # chi2 lies in [0, SD] at the (constrained) optimum; clip its round-off.
    R = np.sqrt(np.clip(1.0 - chi2 / SD, 0.0, 1.0))
    if nonneg:
        logL_marginal = logL_profile.copy()
    else:
        S11, S22, _, _, A12 = _as_maps(stats)
        logL_marginal = logL_profile - 0.5 * np.log(np.maximum(S11 * S22 - A12 ** 2,
                                                               1e-300))
    i, j = np.unravel_index(np.argmax(logL_profile), logL_profile.shape)
    best = {
        "v1": float(v1_grid[i]), "v2": float(v2_grid[j]),
        "a1": float(a1[i, j]), "a2": float(a2[i, j]),
        "chi2": float(chi2[i, j]),
        "chi2_red": float(chi2[i, j] / (stats["nu"] - 4.0)) if stats["nu"] > 4.0 else np.nan,
        "R": float(R[i, j]),
    }
    return {"R": R, "map": R if output == "ccf" else logL_profile, "output": output,
            "logL_profile": logL_profile, "logL_marginal": logL_marginal,
            "a1": a1, "a2": a2, "chi2": chi2,
            "v1_grid": np.asarray(v1_grid, float), "v2_grid": np.asarray(v2_grid, float),
            "best": best}


def profile_per_window(stats_list, weights, v1_grid, v2_grid, nonneg=False, output='ccf'):
    """Profile each window's own amplitudes, then pool the windows' chi^2 maps.

    Each window (order) gets its own ``(a1_i, a2_i)`` -- its own light ratio
    -- at every ``(v1, v2)``; the velocities stay shared. The joint profile
    log-likelihood is the sum of the windows' profiled ones,
    ``logL = -1/2 sum_i min_{a_i} chi2_i``, and ``R^2 = 1 - sum chi2_i /
    sum SD_i``. Use it when the light ratio of the two stars changes across
    the spectrum (different temperatures); with a constant ratio, the shared
    fit (:func:`combine_stats` then :func:`profile_from_stats`) is tighter.

    Parameters
    ----------
    stats_list : sequence of dict
        Each window's :func:`window_stats` on ``v1_grid`` x ``v2_grid``.
    weights : array_like or None
        One positive factor per window on its statistics (e.g. ``1 /
        sigma_i**2``); None for all ones.
    v1_grid, v2_grid : ndarray
        The trial-velocity grids (km/s).
    nonneg : bool
        Constrain every window's amplitudes to be non-negative.
    output : {'ccf', 'loglike'}
        The primary map, ``'map'``: ``R`` (default) or ``logL_profile``.

    Returns
    -------
    dict
        As :func:`profile_from_stats`, except that ``a1`` and ``a2`` are
        (n_windows, n1, n2) and ``best['a1']``, ``best['a2']`` one value per
        window; ``best['chi2_red'] = chi2 / (sum nu - 2 - 2 n_windows)``; and
        ``logL_marginal`` equals ``logL_profile`` (the per-window marginal is
        not evaluated).
    """
    check_output(output)
    stats_list = list(stats_list)
    weights = np.ones(len(stats_list)) if weights is None else np.asarray(weights, float)
    if weights.shape != (len(stats_list),):
        raise ValueError(f"weights must hold one factor per window ({len(stats_list)}), "
                         f"got shape {weights.shape}")
    scaled = [combine_stats([st], [wt]) for st, wt in zip(stats_list, weights)]
    amps = [_amplitudes(st, nonneg) for st in scaled]
    chi2 = sum(chi2_from_stats(st, a1, a2) for st, (a1, a2) in zip(scaled, amps))
    SD = sum(st["SD"] for st in scaled)
    if not SD > 0:
        raise ValueError(f"the data have no power in the windows (SD = {SD}); "
                         "there is nothing to correlate")
    logL_profile = -0.5 * chi2
    R = np.sqrt(np.clip(1.0 - chi2 / SD, 0.0, 1.0))
    a1 = np.array([a[0] for a in amps])
    a2 = np.array([a[1] for a in amps])
    i, j = np.unravel_index(np.argmax(logL_profile), logL_profile.shape)
    dof = sum(st["nu"] for st in scaled) - 2.0 - 2.0 * len(scaled)
    best = {"v1": float(v1_grid[i]), "v2": float(v2_grid[j]),
            "a1": a1[:, i, j].copy(), "a2": a2[:, i, j].copy(),
            "chi2": float(chi2[i, j]),
            "chi2_red": float(chi2[i, j] / dof) if dof > 0 else np.nan,
            "R": float(R[i, j])}
    return {"R": R, "map": R if output == "ccf" else logL_profile, "output": output,
            "logL_profile": logL_profile, "logL_marginal": logL_profile.copy(),
            "a1": a1, "a2": a2, "chi2": chi2,
            "v1_grid": np.asarray(v1_grid, float), "v2_grid": np.asarray(v2_grid, float),
            "best": best}

# =============================================================================
# =============================================================================
def _poly_terms(deg):
    """The (p, q) exponents of a 2-D polynomial of total degree ``deg``."""
    return [(p, q) for p in range(deg + 1) for q in range(deg + 1 - p)]


def crlb_from_map(logL, v1_grid, v2_grid, half=4):
    """Velocities and their Cramer-Rao errors from a log-likelihood map.

    Fits a 2-D polynomial of total degree 6 (by least squares, in
    coordinates scaled to the patch) to ``logL`` over the ``(2*half+1)^2``
    nodes around its argmax, and takes its maximum (Newton steps) and its
    curvature there. For a Gaussian likelihood the negative Hessian of the
    log-likelihood is the Fisher information, and its inverse the
    covariance; ``logL`` being the *profile* over the amplitudes, the
    covariance accounts for them. A quadratic patch misreads a peak that is
    not quadratic over the patch (quartic terms: up to ~6 m/s for lines of
    ~2 km/s FWHM on a 0.2 km/s grid); degree 6 reads the same maps to
    ~0.02 m/s.
    Near the edge of the map the patch is clipped and the degree drops to fit
    it. This is a *local* CRLB, valid near an isolated, well-resolved peak;
    it returns ``nan`` errors (and the unrefined ``peak``) when the curvature
    is not negative definite or the polynomial's maximum lies more than one
    grid step from the argmax (the fit would be extrapolating).

    Evaluate a coarse map, then re-evaluate the statistics on a fine local
    grid around its argmax (:func:`refine_grid`, as ``sparta.ccf2d.CCF2d``
    does) and call this on the fine map.

    Parameters
    ----------
    logL : ndarray (n1, n2)
        Profile log-likelihood surface (``res['logL_profile']``).
    v1_grid, v2_grid : ndarray
        The trial-velocity grids (km/s).
    half : int
        Half-width (in grid nodes) of the patch (>= 1).

    Returns
    -------
    dict
        ``sigma_v1``, ``sigma_v2`` (km/s), ``rho`` (v1-v2 correlation), ``cov``
        (2x2 covariance, km^2/s^2), ``peak`` (the argmax grid node, km/s) and
        ``peak_refined`` (the polynomial's maximum, within one grid step of
        ``peak``; ``peak`` itself, with ``nan`` sigmas, when the curvature is
        not negative definite or the maximum lies further), plus ``on_edge`` (True when the argmax
        sits on the boundary of the velocity grid, where the true maximum may
        lie outside it: then the sigmas are ``nan`` and nothing is refined).

    Raises
    ------
    ValueError
        If ``half`` is not a positive integer.
    """
    if not (isinstance(half, (int, np.integer)) and half >= 1):
        raise ValueError(f"half must be a positive integer (nodes), got {half!r}")
    v1_grid, v2_grid = np.asarray(v1_grid, float), np.asarray(v2_grid, float)
    i, j = np.unravel_index(np.argmax(logL), logL.shape)
    peak = (float(v1_grid[i]), float(v2_grid[j]))
    nan = dict(sigma_v1=np.nan, sigma_v2=np.nan, rho=np.nan, cov=np.full((2, 2), np.nan),
               peak=peak, peak_refined=peak)
    if i in (0, logL.shape[0] - 1) or j in (0, logL.shape[1] - 1):
        return dict(nan, on_edge=True)
    i0, i1 = max(0, i - half), min(logL.shape[0], i + half + 1)
    j0, j1 = max(0, j - half), min(logL.shape[1], j + half + 1)
    deg = max(2, min(6, i1 - i0 - 1, j1 - j0 - 1))
    dx, dy = v1_grid[i0:i1] - v1_grid[i], v2_grid[j0:j1] - v2_grid[j]
    s1, s2 = max(np.abs(dx).max(), 1e-300), max(np.abs(dy).max(), 1e-300)
    X, Y = np.meshgrid(dx / s1, dy / s2, indexing="ij")
    terms = _poly_terms(deg)
    design = np.column_stack([X.ravel() ** p * Y.ravel() ** q for p, q in terms])
    # Relative to the peak value: logL itself can be ~-1e5 (conditioning).
    patch = logL[i0:i1, j0:j1] - logL[i, j]
    coef, *_ = np.linalg.lstsq(design, patch.ravel(), rcond=None)

    def grad_hess(u, v):
        g = np.zeros(2)
        H = np.zeros((2, 2))
        for c, (p, q) in zip(coef, terms):
            if p:
                g[0] += c * p * u ** (p - 1) * v ** q
            if q:
                g[1] += c * q * u ** p * v ** (q - 1)
            if p > 1:
                H[0, 0] += c * p * (p - 1) * u ** (p - 2) * v ** q
            if q > 1:
                H[1, 1] += c * q * (q - 1) * u ** p * v ** (q - 2)
            if p and q:
                H[0, 1] += c * p * q * u ** (p - 1) * v ** (q - 1)
        H[1, 0] = H[0, 1]
        return g, H

    # Newton from the argmax node, in patch units; a maximum more than one
    # grid step away means the polynomial is extrapolating.
    u = np.zeros(2)
    for _ in range(50):
        g, H = grad_hess(*u)
        if not (H[0, 0] < 0 and np.linalg.det(H) > 0):
            return dict(nan, on_edge=False)
        step = np.linalg.solve(H, g)
        u = u - step
        if np.max(np.abs(step)) < 1e-12:
            break
    g, H = grad_hess(*u)
    step1 = 0.5 * (v1_grid[min(i + 1, len(v1_grid) - 1)] - v1_grid[max(i - 1, 0)])
    step2 = 0.5 * (v2_grid[min(j + 1, len(v2_grid) - 1)] - v2_grid[max(j - 1, 0)])
    d1, d2 = u[0] * s1, u[1] * s2
    if not (H[0, 0] < 0 and np.linalg.det(H) > 0) or abs(d1) > step1 or abs(d2) > step2:
        return dict(nan, on_edge=False)
    hessian = H / np.array([[s1 * s1, s1 * s2], [s1 * s2, s2 * s2]])   # in (km/s)^-2
    cov = np.linalg.inv(-hessian)
    sv1, sv2 = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    return dict(sigma_v1=float(sv1), sigma_v2=float(sv2), rho=float(cov[0, 1] / (sv1 * sv2)),
                cov=cov, peak=peak, peak_refined=(peak[0] + float(d1), peak[1] + float(d2)),
                on_edge=False)


# =============================================================================
# =============================================================================
def todcor_map(data_g, g1, g2, dv, v1_grid, v2_grid, sigma=1.0, nonneg=False,
               oversampling=1.0, noise_psd=None, velocity_convention='cz', output='ccf'):
    """Single-window TODCOR correlation and log-likelihood maps (convenience wrapper).

    Equivalent to ``profile_from_stats(window_stats(...), v1_grid, v2_grid)``.
    For several line/order/instrument windows of *one epoch*, accumulate with
    :func:`combine_stats` before profiling (never across epochs).

    Parameters
    ----------
    data_g, g1, g2, dv, v1_grid, v2_grid, sigma, oversampling, noise_psd, velocity_convention
        As :func:`window_stats`: the window's flux and the two templates on
        its grid extended by whole pixels on each side (step ``dv``, km/s),
        trial velocities (km/s), native per-pixel noise (flux units),
        oversampling factor, noise spectrum and Doppler convention.
    nonneg : bool
        As :func:`profile_from_stats`: constrain both amplitudes to be
        non-negative.
    output : {'ccf', 'loglike'}
        As :func:`profile_from_stats`: the primary surface ``'map'`` is the
        two-dimensional correlation ``R`` (default) or ``logL_profile``.

    Returns
    -------
    dict
        The :func:`profile_from_stats` result: the (n1, n2) maps 'R',
        'logL_profile', 'logL_marginal', 'a1', 'a2', 'chi2', and 'map' (the
        one ``output`` selects) with 'output'; 'v1_grid', 'v2_grid'; and
        'best' (v1, v2, a1, a2, chi2, chi2_red, R at the profile argmax).
    """
    stats = window_stats(data_g, g1, g2, dv, v1_grid, v2_grid, sigma=sigma,
                         oversampling=oversampling, noise_psd=noise_psd,
                         velocity_convention=velocity_convention)
    return profile_from_stats(stats, v1_grid, v2_grid, nonneg=nonneg, output=output)


# =============================================================================
# Per-window atoms: the steps of a multi-window fit (sparta.ccf2d.CCF2d)
# =============================================================================
@dataclass(frozen=True, eq=False)
class TodcorWindow:
    """One prepared TODCOR window: the observed pixels and the extended templates.

    Made by :func:`prepare_window`; its arrays are the inputs of
    :func:`window_stats` and :func:`joint_fit`.

    Parameters
    ----------
    wave : ndarray
        The window's observed pixels: a uniform log-lambda grid (Angstrom),
        possibly narrower than the order (see :func:`prepare_window`).
    dv : float
        Its actual velocity step, ``c`` times its ln(lambda) step (km/s;
        :func:`unicor.dsp.log_grid_step`).
    data_g : ndarray
        The observed flux on ``wave``, minus its median (flux units).
    g1, g2 : ndarray
        The two templates minus their median on ``wave`` extended by
        ``extension`` pixels on the left and ``extension`` (odd ``wave``) or
        ``extension + 1`` (even ``wave``) on the right, so their length is
        odd, ramped to zero over the outermost ``ramp`` pixels (flux units).
    extension : int
        Pixels added on the left of ``wave``: ``reach`` plus the ramp.
    reach : int
        The largest shift (pixels) the window may be evaluated at without
        reading the ramp: pass it to :func:`window_stats` (``reach=``).
    """
    wave: np.ndarray
    dv: float
    data_g: np.ndarray
    g1: np.ndarray
    g2: np.ndarray
    extension: int
    reach: int


def prepare_window(wave, flux, t1_wave, t1_flux, t2_wave, t2_flux, *, v_max, ramp=20):
    """Prepare one window for TODCOR: observed pixels, templates extended past them.

    The templates are resampled (linearly) onto the window's grid extended by
    ``v_max`` plus ``ramp`` pixels on each side, so every trial shift up to
    ``v_max`` reads them from their real spectrum: nothing wraps around and no
    line is missing at the edges, so no taper is needed. Observed pixels whose
    templates do not reach ``v_max`` beyond them are dropped (as UNICOR's
    :func:`unicor.ccf.template_coverage` does), never filled with a continuum.
    The outermost ``ramp`` pixels of each extended template fall smoothly to
    zero, keeping the Fourier shift's wrap-around junction away from every
    pixel the fit reads at shifts up to ``reach`` (pass ``reach=win.reach``
    to :func:`window_stats`, which then refuses larger ones). Constant levels
    do not matter: :func:`window_stats` profiles out a free offset, so the
    medians are removed only to keep the numbers small. A template flat to
    1e-9 of its own scale over the pixels a shift can read has no lines in
    this window and is set to exactly zero (amplitude 0 in the fit).

    Parameters
    ----------
    wave : array_like
        The observed window's wavelengths (Angstrom), uniform in ln(lambda).
    flux : array_like
        Its flux, one value per wavelength.
    t1_wave, t1_flux, t2_wave, t2_flux : array_like
        The primary's and the secondary's rest-frame templates: strictly
        increasing wavelengths (Angstrom) and flux.
    v_max : float
        The largest |trial velocity| the window will be evaluated at,
        refinement included (km/s, > 0).
    ramp : int
        Pixels over which each extension falls to zero (>= 2).

    Returns
    -------
    TodcorWindow
        The observed grid, its step ``dv`` (km/s), the data residual, the two
        extended templates, the extension and the reach (pixels).

    Raises
    ------
    ValueError
        If ``wave`` is not uniform in ln(lambda), ``flux`` does not match it,
        a flux is not finite, ``v_max`` or ``ramp`` is invalid, or the
        templates cover no observed pixel across ``+-v_max``.
    """
    wave = np.asarray(wave, float)
    flux = np.asarray(flux, float)
    dv = log_grid_step(wave)
    if flux.shape != wave.shape:
        raise ValueError(f"one flux per wavelength needed: wave has {wave.shape}, "
                         f"flux {flux.shape}")
    if not (np.isfinite(v_max) and v_max > 0):
        raise ValueError(f"v_max must be a positive velocity (km/s), got {v_max}")
    if not (isinstance(ramp, (int, np.integer)) and ramp >= 2):
        raise ValueError(f"ramp must be an integer of at least 2 pixels, got {ramp!r}")
    for name, arr in (("spectrum", flux), ("template1", t1_flux), ("template2", t2_flux)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} flux contains NaN/inf; "
                             "mask or interpolate those pixels first")
    # Pixels a shift of +-v_max moves (either sign and convention: under 'cz'
    # -v_max moves furthest, by ~v^2/2c more), plus one for safety.
    k = int(np.ceil(max(abs(shift_kms(sign * v_max, c)) for sign in (1.0, -1.0)
                        for c in ("cz", "relativistic")) / dv)) + 1
    probe = extend_log_grid(wave, k)
    lo1, hi1 = template_coverage(probe, np.asarray(t1_wave, float), k)
    lo2, hi2 = template_coverage(probe, np.asarray(t2_wave, float), k)
    lo, hi = max(lo1, lo2), min(hi1, hi2)
    if hi - lo < 2:
        raise ValueError(f"the two templates cover fewer than 2 pixels of the window "
                         f"({wave[0]:.2f}-{wave[-1]:.2f} A) across +-{v_max:g} km/s; "
                         "use wider templates or a smaller velocity range")
    wave, flux = wave[lo:hi], flux[lo:hi]
    extension = k + ramp
    # Odd total length, so the Fourier shift has no Nyquist mode: an even
    # window gets one more pixel on the right (n + 2e alone keeps n's parity).
    ext = extend_log_grid(wave, extension + 1)
    ext = ext[1:] if wave.size % 2 == 0 else ext[1:-1]
    edge = 0.5 * (1.0 - np.cos(np.pi * np.arange(ramp) / ramp))   # 0 -> 1
    templates = []
    for t_wave, t_flux in ((t1_wave, t1_flux), (t2_wave, t2_flux)):
        g = resample(t_wave, t_flux, ext)
        reachable = g[extension - k: extension + wave.size + k]
        level = np.max(np.abs(reachable))
        g = g - np.median(reachable)
        # A template flat to round-off over every pixel a shift can read has
        # no lines here: make it exactly zero, so the fit gives it amplitude 0
        # (judged against the template's own scale, whatever its units).
        if np.max(np.abs(g[extension - k: extension + wave.size + k])) <= 1e-9 * level:
            g = np.zeros_like(g)
        g[:ramp] *= edge
        g[-ramp:] *= edge[::-1]
        templates.append(g)
    return TodcorWindow(wave=wave, dv=dv, data_g=flux - np.median(flux),
                        g1=templates[0], g2=templates[1], extension=extension, reach=k)



def refine_grid(v1_grid, v2_grid, i, j, span=2, factor=10):
    """The fine local (v1, v2) grid around a coarse-map node.

    ``span`` local steps either side of node ``(i, j)``, ``factor`` times
    finer. The local step is the mean of the two neighbouring intervals, so
    explicit non-uniform grids refine sensibly too.

    Parameters
    ----------
    v1_grid, v2_grid : array_like
        The coarse trial velocities (km/s).
    i, j : int
        The node to refine around (e.g. the coarse argmax); interior.
    span : int
        Half-width of the fine grid, in local coarse steps.
    factor : int
        Coarse step over fine step.

    Returns
    -------
    f1, f2 : ndarray
        The fine trial velocities (km/s), ``2 * span * factor + 1`` each,
        centred on ``v1_grid[i]`` and ``v2_grid[j]``.

    Raises
    ------
    ValueError
        If ``(i, j)`` is on the edge of the grid (no local step there).
    """
    v1, v2 = np.asarray(v1_grid, float), np.asarray(v2_grid, float)
    if not (0 < i < len(v1) - 1 and 0 < j < len(v2) - 1):
        raise ValueError(f"node ({i}, {j}) is not interior to the ({len(v1)}, {len(v2)}) "
                         "grid; the fine map needs a neighbour on each side")
    step1 = 0.5 * (v1[i + 1] - v1[i - 1])
    step2 = 0.5 * (v2[j + 1] - v2[j - 1])
    nodes = np.arange(-span * factor, span * factor + 1) / factor
    return v1[i] + step1 * nodes, v2[j] + step2 * nodes


@dataclass(frozen=True, eq=False)
class TodcorFit:
    """One joint TODCOR fit of pooled windows, as made by :func:`joint_fit`.

    The maps are on the noise scale of the pooled (weighted) statistics.

    Parameters
    ----------
    pooled : dict
        The weighted coarse-map statistics (:func:`combine_stats`).
    profile : dict
        The coarse map (:func:`profile_from_stats`): 'R', 'logL_profile', ...,
        'best' at its argmax node.
    coarse : dict
        :func:`crlb_from_map` on the coarse map; ``coarse['on_edge']`` says
        the coarse argmax is on the grid edge.
    fine : dict or None
        The fine local map (:func:`profile_from_stats` on
        :func:`refine_grid`); None if the coarse argmax is on the grid edge,
        or if the fine argmax is on the fine map's edge (a ridge or a tie:
        the coarse map is then the more honest answer).
    crlb : dict
        :func:`crlb_from_map` on the map behind the result: the fine one, or
        ``coarse`` itself if ``fine`` is None. ``crlb['peak_refined']`` are
        the velocities (km/s), with ``sigma_v1``, ``sigma_v2``, ``rho``,
        ``cov``.
    at_peak : list of dict
        Each window's :func:`window_stats` at the refined velocities (1 x 1),
        unweighted.
    best : dict
        The joint profile at the refined velocities (``profile_from_stats``'s
        'best': 'v1', 'v2' (km/s), 'a1', 'a2', 'chi2', 'chi2_red', 'R').
    window_chi2 : ndarray
        Each window's share of the joint chi2 at ``best``'s amplitudes, on its
        own (unweighted) noise scale: :func:`chi2_from_stats` of ``at_peak``
        -- the input of :func:`window_noise`.
    """
    pooled: dict
    profile: dict
    coarse: dict
    fine: Optional[dict]
    crlb: dict
    at_peak: list
    best: dict
    window_chi2: np.ndarray


def joint_fit(stats_list, weights, prepared, v1_grid, v2_grid, *, nonneg, output,
              amplitudes='shared', half=4, span=2, factor=10, fine_cache=None):
    """One joint fit of pooled windows: coarse map, fine map, CRLB, peak values.

    Pools the per-window coarse statistics with ``weights`` and profiles them;
    unless its argmax is on the grid edge, re-evaluates every window exactly
    on the fine grid of :func:`refine_grid` around it and profiles that; takes
    the velocities and their Cramer-Rao errors from the fine map
    (:func:`crlb_from_map`, ``half`` nodes); and evaluates every window and
    the joint profile exactly at the refined velocities, so the velocities
    and amplitudes belong to one point.

    The fine statistics are the expensive part, and they are linear in the
    weights: ``fine_cache`` keeps them by coarse node, so a second call with
    other weights (e.g. per-window noise estimates, :func:`window_noise`)
    whose coarse argmax has not moved only re-pools them.

    Parameters
    ----------
    stats_list : sequence of dict
        Each window's :func:`window_stats` on ``v1_grid`` x ``v2_grid``.
    weights : array_like or None
        One positive factor per window on its statistics (dimensionless,
        e.g. ``1 / sigma_i**2`` for unit-noise windows); None for all ones.
    prepared : sequence of (TodcorWindow, dict)
        Each window and the :func:`window_stats` keyword arguments behind its
        statistics (``sigma``, ``oversampling``, ``noise_psd``,
        ``velocity_convention``), in the order of ``stats_list``.
    v1_grid, v2_grid : ndarray
        The coarse trial velocities (km/s).
    nonneg : bool
        Constrain the amplitudes to be non-negative.
    output : {'ccf', 'loglike'}
        The primary map, ``'map'``, of ``profile`` and ``fine``.
    amplitudes : {'shared', 'per_order'}
        One ``(a1, a2)`` for all windows (default: :func:`combine_stats`
        then :func:`profile_from_stats`), or one per window
        (:func:`profile_per_window`), for a light ratio that changes across
        the spectrum.
    half : int
        Half-width (nodes) of the polynomial patch of :func:`crlb_from_map`.
    span, factor : int
        The fine grid (:func:`refine_grid`).
    fine_cache : dict, optional
        Per-window fine statistics by coarse node ``(i, j)``, read and filled
        here; share it only between calls with the same ``prepared`` windows,
        grids, ``span`` and ``factor`` (e.g. the two noise passes of one
        epoch). Default None: a fresh one.

    Returns
    -------
    TodcorFit
        The pooled statistics, the coarse and fine maps, the CRLB behind the
        velocities and every value at the refined peak; with
        ``amplitudes='per_order'``, ``best['a1']`` and ``best['a2']`` hold one
        value per window.

    Raises
    ------
    ValueError
        If ``prepared`` and ``stats_list`` differ in length, ``amplitudes``
        is unknown, or as :func:`combine_stats` and :func:`profile_from_stats`.
    """
    stats_list, prepared = list(stats_list), list(prepared)
    if len(prepared) != len(stats_list):
        raise ValueError(f"one prepared window per statistics needed: {len(prepared)} "
                         f"windows, {len(stats_list)} statistics")
    if amplitudes not in ("shared", "per_order"):
        raise ValueError(f"amplitudes must be 'shared' or 'per_order', not {amplitudes!r}")
    if fine_cache is None:
        fine_cache = {}

    def profile(stats, g1, g2, out=output):
        if amplitudes == "shared":
            return profile_from_stats(combine_stats(stats, weights), g1, g2,
                                      nonneg=nonneg, output=out)
        return profile_per_window(stats, weights, g1, g2, nonneg=nonneg, output=out)

    pooled = combine_stats(stats_list, weights)
    coarse_map = profile(stats_list, v1_grid, v2_grid)
    coarse = crlb_from_map(coarse_map['logL_profile'], v1_grid, v2_grid, half=half)
    fine = None
    if not coarse['on_edge']:
        logl = coarse_map['logL_profile']
        i, j = np.unravel_index(np.argmax(logl), logl.shape)
        f1, f2 = refine_grid(coarse_map['v1_grid'], coarse_map['v2_grid'], i, j, span, factor)
        if (i, j) not in fine_cache:
            fine_cache[(i, j)] = [window_stats(win.data_g, win.g1, win.g2, win.dv, f1, f2, **kw)
                                  for win, kw in prepared]
        fine = profile(fine_cache[(i, j)], f1, f2)
        fi, fj = np.unravel_index(np.argmax(fine['logL_profile']), fine['logL_profile'].shape)
        if fi in (0, len(f1) - 1) or fj in (0, len(f2) - 1):
            fine = None
    crlb = coarse if fine is None else crlb_from_map(
        fine['logL_profile'], fine['v1_grid'], fine['v2_grid'], half=half)
    p1, p2 = np.array([crlb['peak_refined'][0]]), np.array([crlb['peak_refined'][1]])
    at_peak = [window_stats(win.data_g, win.g1, win.g2, win.dv, p1, p2, **kw)
               for win, kw in prepared]
    best = profile(at_peak, p1, p2, out='ccf')['best']
    if amplitudes == "shared":
        window_chi2 = np.array([chi2_from_stats(st, best['a1'], best['a2']).item()
                                for st in at_peak])
    else:
        window_chi2 = np.array([chi2_from_stats(st, a1, a2).item()
                                for st, a1, a2 in zip(at_peak, best['a1'], best['a2'])])
    return TodcorFit(pooled=pooled, profile=coarse_map, coarse=coarse, fine=fine, crlb=crlb,
                     at_peak=at_peak, best=best, window_chi2=window_chi2)


def window_noise(chi2, nu, min_dof=50, amplitudes='shared'):
    """Each window's noise from its own share of a unit-noise joint fit's chi2.

    Window i's share chi2_i (``TodcorFit.window_chi2``) is set against its
    noise degrees of freedom ``dof_i``: with shared amplitudes the four
    shared parameters (v1, v2, a1, a2) are charged to the windows in
    proportion to ``nu_i``, ``dof_i = nu_i (1 - 4 / sum(nu))``; with
    per-window amplitudes the two shared velocities are, and each window pays
    its own two amplitudes, ``dof_i = nu_i (1 - 2 / sum(nu)) - 2``. Then
    ``sigma_i^2 = chi2_i / dof_i``; with one shared window this is
    ``sqrt(chi2_red)``.

    A window with fewer than ``min_dof`` dof, or no residual to measure
    (chi2_i <= 0 or not finite), falls back to the common estimate
    ``sum(chi2_i) / sum(dof_i)`` over the windows that did measure their own,
    or over all windows if none did. No warning is issued here; the caller
    decides how to report ``fallback``.

    Parameters
    ----------
    chi2 : array_like
        Each window's share of the joint chi2 at unit noise (dimensionless).
    nu : array_like
        Each window's noise degrees of freedom (``window_stats``'s ``nu``).
    min_dof : float
        The fewest dof for a window's own estimate.
    amplitudes : {'shared', 'per_order'}
        How the fit treated the amplitudes (:func:`joint_fit`).

    Returns
    -------
    sigma : ndarray
        Each window's noise, per native pixel (flux units: the unit noise
        the statistics were computed at, times the estimate).
    fallback : ndarray of bool
        The windows that got the common estimate.

    Raises
    ------
    ValueError
        If ``chi2`` and ``nu`` are not one value per window each,
        ``amplitudes`` is unknown, or no positive, finite noise can be
        estimated.
    """
    chi2, nu = np.asarray(chi2, float), np.asarray(nu, float)
    if chi2.ndim != 1 or chi2.shape != nu.shape:
        raise ValueError(f"one chi2 per window and one nu per window needed, got "
                         f"{chi2.shape} and {nu.shape}")
    if amplitudes == "shared":
        dof = nu * (1.0 - 4.0 / nu.sum())
    elif amplitudes == "per_order":
        dof = nu * (1.0 - 2.0 / nu.sum()) - 2.0
    else:
        raise ValueError(f"amplitudes must be 'shared' or 'per_order', not {amplitudes!r}")
    own = (dof >= min_dof) & np.isfinite(chi2) & (chi2 > 0)
    pool = own if own.any() else np.ones_like(own)
    common = chi2[pool].sum() / dof[pool].sum()
    if not (np.isfinite(common) and common > 0):
        raise ValueError(
            f"cannot estimate the noise from the fit (chi2 per dof = {common}); "
            "pass an explicit sigma")
    fallback = ~own
    var = np.where(fallback, common, chi2 / np.where(fallback, 1.0, dof))
    return np.sqrt(var), fallback
