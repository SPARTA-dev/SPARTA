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
#   get_wv_grid, resample_spec_to_common_grid, _gram.
#
# Conventions: times in days, frequency in 1/day. U-centering uses 1/(n(n-3))
# and /(n-2), /(n-1) factors -- singular for n <= 3. The pairwise distance
# matrices are vectorized (broadcasting / a BLAS Gram product).
#
# Dependencies: numpy, sparta.spectral_ops.
# Last update: Sahar Shahaf, 20220915

import numpy as np
from sparta import spectral_ops


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
    '''
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
            sp_matrix =  resample_spec_to_common_grid(periodicity_detector.time_series.vals, factor=1)

        if (periodicity_detector.method.split('_')[0] == "shift") or (periodicity_detector.method.split('_')[0] == "shape"):
            for i, s in enumerate(periodicity_detector.time_series.vals):
                vel_tmp = periodicity_detector.time_series.calculated_vrad_list[i]
                periodicity_detector.time_series.vals[i].wv[0] = spectral_ops.doppler_shift_wave(s.wv[0], -vel_tmp)
            sp_matrix =  resample_spec_to_common_grid(periodicity_detector.time_series.vals, factor=1)

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
def get_wv_grid(sp_list, factor):
    '''
    This is only for 1d spectra
    Does not support orders for now
    (USuRPER in general does not support multioprder).
    '''

    min_wv = np.array([-1.0])
    max_wv = np.array([1000000.0])
    delta_wv = np.median(np.diff(sp_list[0].wv))

    for s in sp_list:
      mintmp = np.min(s.wv[0])
      maxtmp = np.max(s.wv[0])
      diftmp = np.median(np.diff(s.wv[0]))

      if mintmp > min_wv:
          min_wv = mintmp
      if maxtmp < max_wv:
          max_wv = maxtmp
      if diftmp < delta_wv:
          delta_wv = diftmp

    return np.arange(start=min_wv, stop=max_wv, step=delta_wv*factor)


# =============================================================================
# =============================================================================

def resample_spec_to_common_grid(sp_list, factor):
    wv_grid   = get_wv_grid(sp_list, factor)
    sp_matrix = np.zeros((len(wv_grid), len(sp_list)))
    for i, s in enumerate(sp_list):
        sp_tmp = spectral_ops.resample(s.wv[0], s.sp[0], wv_grid)
        sp_matrix[:, i] = spectral_ops.normalize(sp_tmp)
    return sp_matrix

