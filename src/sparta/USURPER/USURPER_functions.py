#                   -----------------------------------------------
#                     USuRPer_functions.py (SPARTA USuRPer file)
#                   -----------------------------------------------
# Functions for the Zucker phase-distance-correlation (PDC) periodogram and its
# unit-sphere representation, USuRPER (arxiv.org/pdf/1711.06075.pdf). Called by
# the PeriodicityDetector class.
#
# Distance-correlation core:
#   inner_prod               - U-centered inner product, normalized by 1/(n(n-3)).
#   norm_z                   - part of one U-centered matrix orthogonal to another.
#   unbiased_u_centering     - unbiased U-centering of a distance matrix (zero diagonal).
#   calc_PDC                 - biased PDC value at one frequency.
#   calc_PDC_unbiased        - unbiased PDC value at one frequency.
#   calc_pdc_distance_matrix - build and U-center the value/phase distance matrices.
#
# Spectrum helpers (USuRPER; single-order):
#   _single_order_arrays, get_wv_grid, resample_spec_to_common_grid, _gram.
#
# Conventions: times in days, frequency in 1/day. U-centering uses 1/(n(n-3))
# and /(n-2), /(n-1) factors -- singular for n <= 3. The pairwise distance
# matrices are vectorized (broadcasting / a BLAS Gram product).
#
# Dependencies: numpy, unicor.dsp.
# Last update: Sahar Shahaf, 20220915

import numpy as np
from unicor import dsp


# =============================================================================
# =============================================================================
def inner_prod(a, b):
    '''
    Unbiased U-centered inner product of two distance matrices.

    :param a: U-centered distance matrix (n x n numpy array).
    :param b: U-centered distance matrix (n x n numpy array).
    :return: float, sum(a*b) / (n*(n-3)).
    '''
    n = len(a)
    return np.sum(a * b) / (n * (n - 3))


def norm_z(a, b):
    '''
    The component of ``a`` orthogonal to ``b`` under the U-centered inner
    product (used to build the partial / semi-partial distance correlation).

    :param a: U-centered distance matrix (n x n numpy array).
    :param b: U-centered distance matrix (n x n numpy array).
    :return: a - (<a,b>/<b,b>) * b.
    '''
    factor = inner_prod(a, b) / inner_prod(b, b)
    return a - factor * b


def unbiased_u_centering(a, n):
    '''
    :param a: distance matrix (numpy nXn array)
    :param n: integer (length of a...)
    :return: U-centered distance matrix (unbiased; zero diagonal)
    :raises ValueError: for n <= 3, where the /(n-2) factor here and the
                        1/(n(n-3)) of the U-centered inner product are singular
                        or negative.
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


# =============================================================================
# =============================================================================
def calc_PDC(periodicity_detector, f):
    '''
    This function calculates the PDC value for a given frequency.
    Input:
          periodicity_detector: PeriodicityDetector class for which the PDC is calculated
          f: float, a specific frequency value for it the PDC value is calculated
    '''

    z = np.array([periodicity_detector.time_series.times])
    mat_phase_delta = (z.T - z) % (1 / f)
    b = mat_phase_delta * (1 / f - mat_phase_delta)

    b_mean_x = np.mean(b, axis=0)
    b_mean_y = np.mean(b, axis=1)
    b_mean_total = np.mean(b)

    # Double-centering of b, arranged to match the original [j][i] indexing:
    # B[j, i] = b[i, j] - b_mean_x[i] - b_mean_y[j] + b_mean_total.
    A = np.asarray(periodicity_detector.pdc_mat_A)
    B = b.T - b_mean_x[None, :] - b_mean_y[:, None] + b_mean_total

    num = np.sum(A * B)
    den = np.sqrt(np.sum(A ** 2) * np.sum(B ** 2))
    return num / den


# =============================================================================
# =============================================================================
def calc_PDC_unbiased(periodicity_detector, f):
    '''
    This function calculates the unbiased PDC value for a given frequency.
    Input:
          periodicity_detector: PeriodicityDetector class for which the PDC is calculated
          f: float, a specific frequency value for it the PDC value is calculated
    '''

    z = np.array([periodicity_detector.time_series.times])
    mat_phase_delta = z.T - z
    mat_phase_delta = mat_phase_delta % (1 / f)

    b = mat_phase_delta * (1 / f - mat_phase_delta)

    mat_A_unbiased = periodicity_detector.pdc_mat_A_unbiased
    mat_B_unbiased = unbiased_u_centering(b, periodicity_detector.time_series.size)

    if periodicity_detector.method.split('_')[0] == "shift" or periodicity_detector.method.split('_')[0] == "shape":
        semi = True
        if semi:
            mat_B_unbiased = norm_z(mat_B_unbiased, periodicity_detector.pdc_mat_C_unbiased)
        num = inner_prod(periodicity_detector.pdc_mat_A_unbiased, mat_B_unbiased)
        den = np.sqrt(inner_prod(periodicity_detector.pdc_mat_A_unbiased, periodicity_detector.pdc_mat_A_unbiased))
        den = den * np.sqrt(inner_prod(mat_B_unbiased, mat_B_unbiased))
        return num / den
    else:
        # numerator part
        A = np.array(mat_A_unbiased)
        B = np.array(mat_B_unbiased)
        num = np.sum(np.multiply(A, B))

        # denominators part
        den_A = np.sum(A ** 2)
        den_B = np.sum(B ** 2)

        den = np.sqrt(den_A * den_B)

        res = num / den

        return res


# =============================================================================
# =============================================================================
def _gram(sp_matrix):
    '''Pairwise inner products of the (L2-normalized, bounded) spectrum columns.

    Apple's Accelerate BLAS raises spurious FP-exception flags during matmul
    under numpy>=2 even for finite, small inputs; the Gram of unit vectors is
    bounded in [-1, 1] so a real exception is impossible. Silence the flags.
    '''
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return sp_matrix.T @ sp_matrix


def calc_pdc_distance_matrix(periodicity_detector, calc_biased_flag, calc_unbiased_flag, reverse_existing=False, errors=False):
    '''
    This function calculates the distance matrix used for the PDC calculation.
    Input:
          periodicity_detector: PeriodicityDetector class for which the PDC is calculated
          calc_biased_flag: bool, sets calculating the distance matrix for the biased PDC
          calc_unbiased_flag: bool, sets calculating the distance matrix for the unbiased PDC
    '''
    n = periodicity_detector.time_series.size
    if not reverse_existing:

        a = np.zeros((n,n))
        c = np.zeros((n, n))

        if periodicity_detector.method == 'USURPER':
            waves, fluxes = _single_order_arrays(periodicity_detector.time_series.vals)
            sp_matrix = resample_spec_to_common_grid(waves, fluxes, factor=1)

        if (periodicity_detector.method.split('_')[0] == "shift") or (periodicity_detector.method.split('_')[0] == "shape"):
            # Shift copies to rest: writing back into the caller's spectra made
            # every repeated call shift again and drift the periodogram.
            waves, fluxes = _single_order_arrays(periodicity_detector.time_series.vals)
            vrads = periodicity_detector.time_series.calculated_vrad_list
            if len(vrads) != len(waves):
                raise ValueError(
                    f"The partial periodogram needs one RV (km/s) per epoch in "
                    f"time_series.calculated_vrad_list; got {len(vrads)} for {len(waves)} epochs.")
            waves = [dsp.doppler_shift_wave(w, -v) for w, v in zip(waves, vrads)]
            sp_matrix = resample_spec_to_common_grid(waves, fluxes, factor=1)

        method = periodicity_detector.method
        vals = periodicity_detector.time_series.vals

        if method == 'USURPER':
            # Pairwise zero-lag correlations at once: the Gram matrix of the
            # (L2-normalized) resampled spectra. d = sqrt(2*(1-corr)).
            gram = _gram(sp_matrix)
            a = np.sqrt(np.clip(1.0 - gram, 0.0, None)) * np.sqrt(2)
            np.fill_diagonal(a, 0.0)

        elif method.split('_')[0] in ("shift", "shape"):
            vr = np.asarray(periodicity_detector.time_series.calculated_vrad_list, float)
            a = np.abs(vr[:, None] - vr[None, :])
            gram = _gram(sp_matrix)
            c = np.sqrt(np.clip(1.0 - gram, 0.0, None)) * np.sqrt(2)
            np.fill_diagonal(c, 0.0)

        elif method == 'PDC':
            if len(vals) and isinstance(vals[0], tuple):
                # Partial distance correlation on (..., phase, amp, l) tuples.
                # Kept as an explicit loop: complex, rarely used, and untested.
                for i in range(n):
                    for j in range(i + 1, n):
                        i_val, j_val = vals[i], vals[j]
                        l = j_val[3]
                        dphi = i_val[1] - j_val[1]
                        cos, sin = np.cos(dphi), np.sin(dphi)
                        base = i_val[2] ** 2 - 2 * i_val[2] * j_val[2] * cos + j_val[2] ** 2
                        temp = (np.sqrt(base + l * (i_val[2] + j_val[2]) * sin + 0.5 * l * l * (1 + cos))
                                + np.sqrt(base + l * (i_val[2] - j_val[2]) * sin + 0.5 * l * l * (1 - cos))
                                + np.sqrt(base - l * (i_val[2] - j_val[2]) * sin + 0.5 * l * l * (1 - cos))
                                + np.sqrt(base - l * (i_val[2] + j_val[2]) * sin + 0.5 * l * l * (1 + cos)))
                        a[i, j] = a[j, i] = temp / 2 - l
            elif errors:
                # Error-aware distance of Binnenfeld, Shahaf & Zucker (2024),
                # Eq. 3, as published: e^(-x^2) + x*erf(x) - y, without a
                # sqrt(pi) on the erf term. An energy-distance derivation
                # suggests one; the authors confirmed (2026-10-14) that the
                # published form is intended. Pinned by test_pdc.py.
                from scipy import special
                y = np.asarray(vals, float)
                err = np.asarray(periodicity_detector.time_series.errors, float)
                y1, y2 = y[:, None], y[None, :]
                e1, e2 = err[:, None], err[None, :]
                den = np.sqrt(2 * (e1 ** 2 + e2 ** 2))
                x = (y1 - y2) / den
                yv = (e1 + e2) / den
                ev = np.sqrt(e1 ** 2 + e2 ** 2) * (np.exp(-x ** 2) + x * special.erf(x) - yv)
                a = np.sqrt(np.clip(ev, 0.0, None))
                np.fill_diagonal(a, 0.0)
            else:
                y = np.asarray(vals, float)
                a = np.abs(y[:, None] - y[None, :])

        if method == 'shape_periodogram':
            a, c = c.copy(), a.copy()

    else:

        temp = periodicity_detector.pdc_a.copy()
        a = periodicity_detector.pdc_c.copy()
        c = temp.copy()

    periodicity_detector.pdc_a = a
    periodicity_detector.pdc_c = c

    # U - centering

    if calc_biased_flag:
        periodicity_detector.pdc_mat_A = a - np.mean(a, axis=0)[None, :] - np.mean(a, axis=1)[:, None] + np.mean(a)

    if calc_unbiased_flag:
        mat_A_unbiased = unbiased_u_centering(periodicity_detector.pdc_a, periodicity_detector.time_series.size)

        if periodicity_detector.method.split('_')[0] == "shift" or periodicity_detector.method.split('_')[0] == "shape":
            periodicity_detector.pdc_mat_C_unbiased = unbiased_u_centering(periodicity_detector.pdc_c, periodicity_detector.time_series.size)
            periodicity_detector.pdc_mat_A_unbiased = norm_z(mat_A_unbiased, periodicity_detector.pdc_mat_C_unbiased)
        else:
            periodicity_detector.pdc_mat_A_unbiased = mat_A_unbiased
    return a



# =============================================================================
# =============================================================================
def _single_order_arrays(sp_list):
    '''
    The wavelength and flux arrays of single-order spectra, validated for the
    common-grid resampling.

    :param sp_list: list of Spectrum-like objects (``.wv``/``.sp`` lists of
                    per-order arrays); wavelengths in Angstrom.
    :return: (waves, fluxes) -- lists of 1-D float arrays, one per spectrum.
    :raises ValueError: if a spectrum has more than one order, or its
                        wavelengths are not strictly increasing.
    '''
    waves, fluxes = [], []
    for i, s in enumerate(sp_list):
        if len(s.wv) != 1:
            raise ValueError(
                f"Spectrum {i} has {len(s.wv)} orders, but USuRPER and the partial "
                "periodograms use a single order. Select one first (e.g. "
                "Observations.SelectOrders).")
        w = np.asarray(s.wv[0], float)
        if not np.all(np.diff(w) > 0):
            raise ValueError(
                f"Spectrum {i}: wavelengths must be strictly increasing for the "
                "common-grid resampling.")
        waves.append(w)
        fluxes.append(s.sp[0])
    return waves, fluxes


# =============================================================================
# =============================================================================
def get_wv_grid(waves, factor):
    '''
    The uniform wavelength grid common to all spectra: their overlap, sampled at
    the finest median pixel step.

    :param waves: list of 1-D increasing wavelength arrays (Angstrom).
    :param factor: float, multiplies the grid step.
    :return: 1-D wavelength grid (Angstrom).
    :raises ValueError: if the spectra do not overlap (fewer than two common
                        grid points).
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


# =============================================================================
# =============================================================================

def resample_spec_to_common_grid(waves, fluxes, factor):
    '''
    Resample spectra onto their common grid and normalize each one.

    :param waves: list of 1-D increasing wavelength arrays (Angstrom).
    :param fluxes: list of the matching 1-D flux arrays.
    :param factor: float, multiplies the common-grid step (see get_wv_grid).
    :return: (n_grid x n_spectra) array; each column mean-subtracted and
             L2-normalized.
    '''
    wv_grid   = get_wv_grid(waves, factor)
    sp_matrix = np.zeros((len(wv_grid), len(waves)))
    for i, (w, f) in enumerate(zip(waves, fluxes)):
        sp_matrix[:, i] = dsp.normalize(dsp.resample(w, f, wv_grid))
    return sp_matrix
