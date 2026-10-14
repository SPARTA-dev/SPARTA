"""Wavelength-axis helpers shared by the instrument loaders (Angstrom throughout)."""
import numpy as np


def _ascending(w, *arrays):
    '''
    Order a wavelength axis (and the arrays sampled on it) by increasing
    wavelength.

    Parameters
    ----------
    w : array_like
        Wavelengths (Angstrom), strictly monotonic.
    *arrays : ndarray
        Arrays of the same length as ``w`` (flux, mask, ...).

    Returns
    -------
    tuple of ndarray
        ``(w, *arrays)`` as float64 ``w`` reversed together with ``arrays`` if
        ``w`` was decreasing.

    Raises
    ------
    ValueError
        If ``w`` is not strictly monotonic (or contains NaN).
    '''
    w = np.asarray(w, dtype=float)
    step = np.diff(w)
    if np.all(step > 0):
        return (w, *arrays)
    if np.all(step < 0):
        return (w[::-1], *(a[::-1] for a in arrays))
    raise ValueError("wavelength axis is not strictly monotonic (or contains NaN)")


def _wavelength_window(w, min_wv, max_wv):
    '''
    Slice selecting the pixels of an ascending wavelength axis inside a window.

    Parameters
    ----------
    w : ndarray
        Ascending wavelengths (Angstrom).
    min_wv, max_wv : float
        Window bounds (Angstrom); pixels with ``min_wv < w <= max_wv`` are kept.
        A window that only partly overlaps the coverage is clipped to it.

    Returns
    -------
    slice

    Raises
    ------
    ValueError
        If the window is empty/inverted, ``w`` is not ascending, or the window
        does not overlap the coverage of ``w``.
    '''
    if not min_wv < max_wv:
        raise ValueError(f"empty wavelength window: min_wv={min_wv} >= max_wv={max_wv} (Angstrom)")
    if np.isnan(w).any() or np.any(np.diff(w) < 0):
        raise ValueError("wavelength axis must be ascending and NaN-free to apply a window")
    lo = np.searchsorted(w, min_wv, side='right')
    hi = np.searchsorted(w, max_wv, side='right')
    if hi <= lo:
        raise ValueError(f"wavelength window ({min_wv}, {max_wv}] Angstrom does not overlap "
                         f"the spectrum coverage [{w[0]:.3f}, {w[-1]:.3f}] Angstrom")
    return slice(lo, hi)


def _best_order(waves, fluxes, min_wv, max_wv):
    '''
    Pick the echelle order that covers the wavelength window best.

    Parameters
    ----------
    waves, fluxes : sequence of 1-D arrays
        Per-order wavelengths (Angstrom; either direction) and fluxes.
    min_wv, max_wv : float
        Window (Angstrom), as in `_wavelength_window`.

    Returns
    -------
    (w, s) : tuple of float64 ndarray
        Ascending wavelengths and flux of the order with the most pixels in
        the window (the first such order on ties), trimmed to the window.

    Raises
    ------
    ValueError
        If no order overlaps the window.
    '''
    best = None
    for wave, flux in zip(waves, fluxes):
        w, s = _ascending(wave, np.asarray(flux, dtype=float))
        n_in = (np.searchsorted(w, max_wv, side='right')
                - np.searchsorted(w, min_wv, side='right'))
        if n_in > 0 and (best is None or n_in > best[0]):
            best = (n_in, w, s)
    if best is None:
        raise ValueError(f"no echelle order overlaps the wavelength window "
                         f"({min_wv}, {max_wv}] Angstrom")
    _, w, s = best
    window = _wavelength_window(w, min_wv, max_wv)
    return w[window], s[window]
