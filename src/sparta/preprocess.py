"""Per-order preprocessing atoms: the steps of ``Spectrum`` on plain arrays.

Each function takes one order's wavelength and flux arrays and returns plain
arrays (or a tuple of them); no ``Spectrum`` objects. ``Spectrum.trim``,
``interpolate``, ``remove_cosmics``, ``filter_spectrum`` and ``preprocess``
are loops over these, so chaining them by hand reproduces the object's
result exactly. The taper of ``apply_cosine_bell`` is
``flux * unicor.tukey_window(len(flux), alpha)``.

``resampling_noise_psd`` and ``native_phases`` model what that resampling
does to white native noise -- the noise budget of ``CCF2d``, which feeds
them to ``todcor.window_stats`` as ``noise_psd``. They probe
``resample_flux`` itself, so they live here rather than in ``todcor``.

Conventions: wavelengths in Angstrom, velocities in km/s. The log-lambda
step is ``ln(1 + dv/c)``, c from ``astropy.constants``.
"""

from functools import lru_cache

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter
from astropy.stats import sigma_clip, mad_std
from astropy import constants as consts, units as u
from specutils import Spectrum as Spec1D
from specutils.manipulation import SplineInterpolatedResampler
from todcor.estimator import resampled_noise_psd
from unicor import dsp

__all__ = ["resample_flux", "trim_margins", "resample_to_linear_grid",
           "resample_to_log_grid", "coarsest_native_step", "remove_cosmics", "bandpass",
           "resampling_noise_psd", "native_phases"]


def resample_flux(wave, flux, new_wave, unit=u.Angstrom):
    '''
    Resample one order's flux with the cubic spline of ``Spectrum.interpolate``.

    The spline is specutils' ``SplineInterpolatedResampler``. Other code that
    must model this resampling exactly (the noise model of
    ``resampling_noise_psd``, the simulator's archive rebinning) calls this
    function too.

    Parameters
    ----------
    wave : array_like
        1-D increasing wavelengths, in ``unit``.
    flux : array_like
        1-D flux matching ``wave``.
    new_wave : array_like
        1-D increasing wavelengths (``unit``) inside the span of ``wave``.
    unit : astropy.units.Unit
        Unit of both wavelength arrays (default Angstrom).

    Returns
    -------
    ndarray
        The flux at ``new_wave``; NaN outside the span of ``wave``.
    '''
    src = Spec1D(flux=np.asarray(flux, float) * u.dimensionless_unscaled,
                 spectral_axis=np.asarray(wave, float) * unit)
    return SplineInterpolatedResampler()(src, np.asarray(new_wave, float) * unit).flux.value


def trim_margins(wave, flux, n_left=0, n_right=0, clean_margins=False, remove_nans=True,
                 margin_fraction=0.1, margin_window=15):
    '''
    Trim the ends of one order: NaN pixels, padded margins, then fixed counts.

    Echelle orders are often padded at their ends: either with exact zeros,
    or -- as in the NRES pipeline -- with unexposed, background-subtracted
    detector that reads zero plus noise. Those margins must go *before* the
    continuum is removed: filtering a jump from ~0 to the continuum level
    turns it into a step at the order edge.

    Parameters
    ----------
    wave : array_like
        1-D wavelengths (Angstrom).
    flux : array_like
        1-D flux matching ``wave``.
    n_left, n_right : int
        Pixels to remove from the blue / red end, counted inside the margins
        when ``clean_margins`` is set.
    clean_margins : bool
        Drop the padded margins: the leading and trailing runs whose running
        median (``margin_window`` pixels) is below ``margin_fraction`` of the
        order's median flux.
    remove_nans : bool
        Drop NaN pixels first.
    margin_fraction : float
        Flux level, as a fraction of the order's median, below which an edge
        pixel counts as padding.
    margin_window : int
        Width (pixels) of the running median that smooths the noise before
        the threshold is applied.

    Returns
    -------
    wave, flux : ndarray
        The trimmed order (Angstrom, flux).

    Raises
    ------
    ValueError
        If ``clean_margins`` finds no pixel above the margin level.
    '''
    w = np.asarray(wave, float)
    s = np.asarray(flux, float)

    if remove_nans:
        keep = ~np.isnan(s)
        w, s = w[keep], s[keep]

    # Inclusive bounds of the exposed part, on the arrays as they now are
    # (after NaN removal).
    first, last = 0, len(s) - 1
    if clean_margins and len(s):
        level = margin_fraction * np.nanmedian(s)
        smooth = median_filter(np.nan_to_num(s), size=margin_window, mode='nearest')
        exposed = np.flatnonzero(smooth > level)
        if exposed.size == 0:
            raise ValueError("an order has no flux above the margin level; "
                             "drop it (select_orders) or lower margin_fraction")
        first, last = exposed[0], exposed[-1]

    lo = first + n_left
    hi = last - n_right + 1
    return w[lo:hi], s[lo:hi]


def resample_to_linear_grid(wave, flux, delta):
    '''
    Resample one order onto an evenly spaced wavelength grid.

    The step is ``|delta|`` times the order's median wavelength step. The
    grid starts one new step above the first wavelength and stops before one
    new step below the last, so the cubic spline (``resample_flux``) never
    extrapolates.

    Parameters
    ----------
    wave : array_like
        1-D increasing wavelengths (Angstrom), at least 2.
    flux : array_like
        1-D flux matching ``wave``.
    delta : float
        The step, in units of the order's median wavelength step
        (dimensionless; ``Spectrum.interpolate``'s default is 0.5).

    Returns
    -------
    wave, flux : ndarray
        The new grid (Angstrom) and the resampled flux.

    Raises
    ------
    ValueError
        For an order of fewer than 2 pixels.
    '''
    _require_two_pixels(wave)
    dlam = delta * np.median(np.diff(wave))
    new_wave = np.arange(np.min(wave) + abs(dlam), np.max(wave) - abs(dlam), abs(dlam))
    return new_wave, resample_flux(wave, flux, new_wave)


def resample_to_log_grid(wave, flux, dv_kms):
    '''
    Resample one order onto a log-lambda (velocity-even) grid.

    The ln(lambda) step is ``ln(1 + dv/c)``, c from ``astropy.constants``
    (the step CCF1d and CCF2d check against). The grid starts one median
    input step above the first wavelength and stops before one median step
    below the last, so the cubic spline (``resample_flux``) never
    extrapolates.

    Parameters
    ----------
    wave : array_like
        1-D increasing wavelengths (Angstrom), at least 2.
    flux : array_like
        1-D flux matching ``wave``.
    dv_kms : float
        The velocity step (km/s).

    Returns
    -------
    wave, flux : ndarray
        The new grid (Angstrom) and the resampled flux.

    Raises
    ------
    ValueError
        For an order of fewer than 2 pixels.
    '''
    _require_two_pixels(wave)
    beta = ((dv_kms * u.kilometer / u.second / consts.c.to('km/s')).decompose()).value
    median_step = np.median(np.diff(wave))
    new_wave = np.exp(np.arange(np.log(np.min(wave) + median_step),
                                np.log(np.max(wave) - median_step),
                                np.log(1 + beta)))
    return new_wave, resample_flux(wave, flux, new_wave)


def coarsest_native_step(previous_kms, wave):
    '''
    The native velocity step of an order after it is put on a new grid.

    Resampling onto a finer grid adds no independent pixels; resampling onto
    a coarser one leaves fewer. So the coarsest sampling seen so far is the
    native one: ``max(previous, step of the new grid)``. The new grid's step
    is c * median(d(lambda)/lambda) (``unicor.dv_from_wave``): on a linear
    grid, whose velocity step falls from the blue to the red end, its
    mid-order value.

    Parameters
    ----------
    previous_kms : float
        The native step before (km/s).
    wave : array_like
        The new wavelength grid (Angstrom), at least 2 points.

    Returns
    -------
    float
        The native step (km/s).
    '''
    return max(previous_kms, dsp.dv_from_wave(wave))


def remove_cosmics(wave, flux, sigma_upper=3):
    '''
    Replace the upward outliers (cosmic-ray hits) of one order.

    An outlier lies more than ``sigma_upper`` x mad_std above the median of
    the order (iterated, as in astropy's sigma_clip); points below are never
    touched. A clipped pixel is replaced by linear interpolation in
    wavelength between its nearest unclipped neighbours, so the grid stays
    uniform. NaN pixels stay NaN and are never interpolation anchors.

    Parameters
    ----------
    wave : array_like
        1-D wavelengths (Angstrom).
    flux : array_like
        1-D flux matching ``wave``; not modified.
    sigma_upper : float
        Clipping threshold, in units of mad_std.

    Returns
    -------
    ndarray
        The cleaned flux, on ``wave``.
    '''
    w = np.asarray(wave, float)
    s = np.array(flux, float)
    clipped = np.ma.getmaskarray(sigma_clip(s, sigma_lower=np.inf, sigma_upper=sigma_upper,
                                            masked=True, cenfunc='median', stdfunc=mad_std))
    finite = np.isfinite(s)
    cosmic = clipped & finite
    anchors = ~clipped & finite
    s[cosmic] = np.interp(w[cosmic], w[anchors], s[anchors])
    return s


def bandpass(wave, flux, lowcut, highcut, order=1):
    '''
    Band-pass filter one evenly sampled order with a Butterworth filter.

    Removes the slowly varying instrumental response (below ``lowcut``) and
    the pixel-scale noise (above ``highcut``). The filter runs forward and
    backward (zero phase) on the flux minus its 80th percentile, and the
    result is standardized to zero mean and unit variance. The Nyquist
    frequency is taken from the first wavelength step.

    Parameters
    ----------
    wave : array_like
        1-D evenly spaced wavelengths (Angstrom).
    flux : array_like
        1-D flux matching ``wave``.
    lowcut : float
        Lower band edge, in units of the lowest frequency of the order,
        1 / (max(wave) - min(wave)).
    highcut : float
        Upper band edge, in units of the Nyquist frequency (0-1).
    order : int
        Order of the Butterworth filter.

    Returns
    -------
    ndarray
        The filtered, standardized flux (dimensionless).
    '''
    w = np.asarray(wave, float)
    s = np.asarray(flux, float)
    nyq = 0.5/np.abs(w[1]-w[0])

    # The lowest frequency the order holds, in units of the Nyquist frequency.
    df = (np.max(w)-np.min(w))**(-1)/nyq
    low = lowcut*df

    b, a = signal.butter(order, [low, highcut], btype='band')

    # Subtract a continuum-like level (the 80th percentile) so the filter's
    # edge transients start near zero.
    s = signal.filtfilt(b, a, s-np.percentile(s, 80), padlen=10*max(len(b), len(a)))
    return (s-np.mean(s))/np.std(s)


def resampling_noise_psd(n, dv_native, dv, phases=None, n_phase=32):
    '''
    Noise spectrum of one order's white native noise after ``resample_flux``.

    The ``noise_psd`` of ``todcor.window_stats`` (see
    ``todcor.resampled_noise_psd``) for an order resampled once, by the
    cubic spline of ``Spectrum.interpolate``, from a uniform native grid onto
    a log grid of step ``dv``, finer or coarser than native. Each native
    pixel's impulse response depends on its sub-pixel phase relative to the
    analysis nodes, so the responses at ``n_phase`` phases are averaged with
    weights counting the order's native pixels at each phase
    (``native_phases``). With unknown native positions the phases are taken
    as uniformly spread -- true for a linear-wavelength native grid, whose
    velocity step drifts across the order, but not for a commensurate log
    grid (e.g. one exactly half as fine), whose few fixed phases can change
    the noise power by ~15 per cent.

    The responses depend on the two steps only (a log grid is the same at
    every wavelength), so they are cached on them, rounded to 1e-9 km/s:
    the same grid, epoch after epoch, shares one probe. Earlier filtering or
    repeated resampling is not represented.

    Parameters
    ----------
    n : int
        Length of the analysis window (pixels).
    dv_native : float
        The order's native velocity step (km/s).
    dv : float
        The analysis step (km/s).
    phases : array_like, optional
        Sub-pixel phases in [0, 1) of the order's native pixels on the
        analysis grid (analysis pixels); None for uniformly spread.
    n_phase : int
        Number of phase bins probed.

    Returns
    -------
    ndarray
        Relative noise power per positive Fourier mode (white noise of the
        native variance on the analysis grid = 1), length ``n // 2``.
    '''
    responses = _impulse_responses(round(float(dv_native), 9), round(float(dv), 9), n_phase)
    weights = None
    if phases is not None:
        bins = np.floor(np.asarray(phases) * n_phase).astype(int) % n_phase
        weights = np.bincount(bins, minlength=n_phase)
    return resampled_noise_psd(responses, n, dv_native / dv, weights=weights)


@lru_cache(maxsize=256)
def _impulse_responses(dv_native, dv, n_phase=32, half=20):
    '''
    Analysis-grid responses of ``resample_flux`` to a unit impulse in one
    native pixel, at ``n_phase`` sub-pixel phases.

    The native grid is synthetic: log-uniform with step ``dv_native``, with
    impulses ``2 * half`` pixels apart. Impulse k is evaluated at analysis
    nodes placed so that it sits (k + 0.5) / n_phase analysis pixels above the
    node below it, out to ``half`` native pixels either side -- far beyond the
    cubic spline's support (its cardinal function decays as 0.27 per pixel).
    One spline evaluation serves every phase: the nodes of all impulses are
    concatenated.

    Parameters
    ----------
    dv_native : float
        Native velocity step (km/s).
    dv : float
        Analysis velocity step (km/s).
    n_phase : int
        Number of phases.
    half : int
        Native pixels kept either side of each impulse.

    Returns
    -------
    tuple of ndarray
        n_phase read-only responses (analysis pixels).
    '''
    lam0 = 5000.0                            # Angstrom; the response is scale-free
    step_native, step = dv_native / dsp.C_KMS, dv / dsp.C_KMS
    hits = half + 2 * half * np.arange(n_phase)
    ln_native = np.log(lam0) + step_native * np.arange(2 * half * n_phase + 1)
    flux = np.zeros(ln_native.size)
    flux[hits] = 1.0

    reach = half * dv_native / dv            # analysis pixels within `half` native ones
    nodes = []
    for k, j in enumerate(hits):
        phase = (k + 0.5) / n_phase
        m = np.arange(np.floor(phase - reach) + 1, np.ceil(phase + reach))
        nodes.append(ln_native[j] + (m - phase) * step)
    response = resample_flux(np.exp(ln_native), flux, np.exp(np.concatenate(nodes)))
    out = np.split(response, np.cumsum([len(x) for x in nodes])[:-1])
    for h in out:
        h.flags.writeable = False
    return tuple(out)


def native_phases(native_wave, wave, dv):
    '''
    Sub-pixel phase of each native pixel on a log analysis grid.

    The ``phases`` of ``resampling_noise_psd`` for an order whose native
    wavelengths are known (i.e. it has not been resampled before).

    Parameters
    ----------
    native_wave : array_like
        The order's native wavelengths (Angstrom).
    wave : ndarray
        The log analysis grid (Angstrom), step ``dv``.
    dv : float
        Analysis velocity step (km/s): ``c`` times the actual ln(lambda)
        step of ``wave``.

    Returns
    -------
    ndarray
        Phases in [0, 1) (analysis pixels) of the native pixels inside the
        analysis grid.
    '''
    x = dsp.C_KMS * np.log(np.asarray(native_wave, float) / wave[0]) / dv
    return np.mod(x[(x >= 0) & (x <= len(wave) - 1)], 1.0)

def _require_two_pixels(wave):
    if len(wave) < 2:
        raise ValueError(f"an order of {len(wave)} pixels cannot be resampled; "
                         "it needs at least 2 pixels")
