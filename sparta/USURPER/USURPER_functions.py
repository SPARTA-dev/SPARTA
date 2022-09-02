#                   -----------------------------------------------
#                     USuRPer_functions.py (SPARTA USuRPer file)
#                   -----------------------------------------------
# This file defines the functions necessary for Zucker PDC / USuRPer calculation (arxiv.org/pdf/1711.06075.pdf).
# It is being used inside an overall calculation method inside the PeriodicityDetector class.
#
# This file stores the following methods:
# ---------------------------------------------
#
# 1) calc_pdc_distance_matrix - resamples the spectrum on a linear or logarithmic scale
# 2) calc_PDC_unbiased - calculates the unbiased PDC value for a given frequency.
# 3) calc_PDC - calculates the PDC value for a given frequency.
#
# Dependencies: numpy, math.
# Last update: Avraham Binnenfeld, 20200607.

import numpy as np
from sparta.UNICOR import Template, Spectrum, CCF1d
from numba import jit
import math

# =============================================================================
# =============================================================================
class USURPER:

    def inner_prod(self, a, b):
        return inner_prod(a, b)

    def calc_PDC(self, periodicity_detector, f):
        return calc_PDC(periodicity_detector, f)

    def calc_PDC_unbiased(self, periodicity_detector, f):
        return calc_PDC_unbiased(periodicity_detector, f)

    def calc_pdc_distance_matrix(self, periodicity_detector, calc_biased_flag, calc_unbiased_flag):
        return calc_pdc_distance_matrix(periodicity_detector, calc_biased_flag, calc_unbiased_flag)


# =============================================================================
# =============================================================================
def inner_prod(a, b):

    size = len(a)

    A = np.array(a)
    B = np.array(b)

    np.fill_diagonal(A, 0)
    np.fill_diagonal(B, 0)

    sum_bin = np.sum(np.multiply(A, B))

    res = 1 / (size * (size - 3)) * sum_bin

    return res


def norm_z(a, b):
    '''

    :param a:
    :param b:
    :return:
    '''
    return np.array(a) - (inner_prod(a, b) / inner_prod(b, b)) * np.array(b)


def unbiased_u_centering(a, n):
    mat_unbiased = [[None for _ in range(n)] for _ in range(n)]

    mat_unbiased = np.array(mat_unbiased)
    np.fill_diagonal(mat_unbiased, 0)

    for i in range(n):
        for j in range(n):
            if i != j:
                mat_unbiased[i][j] = a[i][j]
                v2 = np.sum(a[i])
                v2 = v2 / (n - 2)

                v3 = np.sum(a, axis=0)[j]
                v3 = v3 / (n - 2)

                v4 = np.sum(a)
                v4 = v4 / ((n - 1) * (n - 2))

                mat_unbiased[i][j] = mat_unbiased[i][j] - v2 - v3 + v4

    return mat_unbiased.copy()


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
    mat_phase_delta = z.T - z
    mat_phase_delta = mat_phase_delta % (1 / f)

    b = mat_phase_delta * (1 / f - mat_phase_delta)

    b_mean_x = np.mean(b, axis=0)
    b_mean_y = np.mean(b, axis=1)
    b_mean_total = np.mean(b)

    mat_B = [[b[i][j] - b_mean_x[i] - b_mean_y[j] + b_mean_total for i in range(periodicity_detector.time_series.size)] for j in
             range(periodicity_detector.time_series.size)]

    # numerator part
    A = np.array(periodicity_detector.pdc_mat_A)
    B = np.array(mat_B)
    num = np.sum(np.multiply(A, B))

    # denominators part
    den_A = np.sum(A ** 2)
    den_B = np.sum(B ** 2)

    den = np.sqrt(den_A * den_B)

    res = num / den

    return res


# =============================================================================
# =============================================================================
def calc_PDC_unbiased(periodicity_detector, f, fast=False):
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

    if fast and periodicity_detector.method.split('_')[0] != "shift" and periodicity_detector.method.split('_')[0] != "shape":
        distx = np.array(periodicity_detector.pdc_mat_A_unbiased)
        disty = np.array(b)

        covar = calc_fast_dcov(distx, disty)
        varx = calc_fast_dcov(distx, distx)
        vary = calc_fast_dcov(disty, disty)

        return covar / np.real(np.sqrt(varx * vary))


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
def calc_pdc_distance_matrix(periodicity_detector, calc_biased_flag, calc_unbiased_flag, reverse_existing=False, fast=False):
    '''
    This function calculates the distance matrix used for the PDC calculation.
    Input:
          periodicity_detector: PeriodicityDetector class for which the PDC is calculated
          calc_biased_flag: bool, sets calculating the distance matrix for the biased PDC
          calc_unbiased_flag: bool, sets calculating the distance matrix for the unbiased PDC
    '''

    if reverse_existing == False:

        a = [[None for _ in range(periodicity_detector.time_series.size)] for _ in
             range(periodicity_detector.time_series.size)]

        c = [[None for _ in range(periodicity_detector.time_series.size)] for _ in
             range(periodicity_detector.time_series.size)]

        for i, i_val in enumerate(periodicity_detector.time_series.vals):
            for j, j_val in enumerate(periodicity_detector.time_series.vals):
                if i > j:
                    a[i][j] = a[j][i]
                    c[i][j] = c[j][i]
                elif i == j:
                    a[i][j] = 0
                    c[i][j] = 0
                else:
                    if periodicity_detector.method == 'PDC':
                        a[i][j] = abs(i_val - j_val)

                    elif periodicity_detector.method == 'USURPER':
                        ccf = CCF1d().CrossCorrelateSpec(spec_in=j_val, template_in=Template(template=i_val), dv=0.5,
                                                         VelBound=[-1, 1], fastccf=False)
                        ccf_val = ccf.subpixel_CCF(ccf.Corr['vel'], ccf.Corr['corr'][0], 0)
                        a[i][j] = np.sqrt(1 - abs(min(ccf_val, 1))) * np.sqrt(2)

                    elif periodicity_detector.method.split('_')[0] == "shift" or periodicity_detector.method.split('_')[0] == "shape":
                        a[i][j] = abs(periodicity_detector.time_series.calculated_vrad_list[i] -
                                            periodicity_detector.time_series.calculated_vrad_list[j])

                        s = Spectrum(wv=[Template().doppler(-periodicity_detector.time_series.calculated_vrad_list[j], j_val.wv[0])],
                                     sp=j_val.sp) # .SpecPreProccess()
                        t = Template(spectrum=i_val.sp[0], wavelengths=i_val.wv[0])
                        t.model.wv = t.doppler(-periodicity_detector.time_series.calculated_vrad_list[i])

                        ccf = CCF1d().CrossCorrelateSpec(spec_in=s, template_in=t, dv=0.5,
                                                         VelBound=[-1, 1], fastccf=False)
                        ccf_val = ccf.subpixel_CCF(ccf.Corr['vel'], ccf.Corr['corr'][0], v=0)
                        c[i][j] = np.sqrt(1 - abs(min(ccf_val, 1))) * np.sqrt(2)


        if periodicity_detector.method == 'shape_periodogram':
            temp = a.copy()
            a = c.copy()
            c = temp.copy()

    else:

        temp = periodicity_detector.pdc_a.copy()
        a = periodicity_detector.pdc_c.copy()
        c = temp.copy()

    periodicity_detector.pdc_a = a
    periodicity_detector.pdc_c = c

    if fast:
        periodicity_detector.pdc_mat_A_unbiased = a
        return a


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


@jit(nopython=True, cache=True)
def cpu_cumsum_cpy(data):  # pragma: no cover
    """Create cumulative sum since numba doesn't sum over axes."""
    cumsum = data.copy()
    for i in range(1, data.shape[0]):
        cumsum[i, :] = data[i, :] + cumsum[i - 1, :]
    return cumsum


@jit(nopython=True, cache=True)
def fast_1d_dcov_cpy(x, y, bias=False):
    """
    This function is taken from the Hyppo package https://github.com/neurodata/hyppo

    Calculate the Dcorr test statistic.

    See: https://www.sciencedirect.com/science/article/abs/pii/S0167947319300313
    """

    n = x.shape[0] ** 2

    # sort inputs
    x_orig = x.ravel()
    x = np.sort(x_orig)
    y = y.reshape(-1, 1)
    y = y[np.argsort(x_orig)]
    x = x.reshape(-1, 1)  # for numba

    # cumulative sum
    si = cpu_cumsum_cpy(x)

    ax = (np.arange(-(n - 2), n + 1, 2) * x.ravel()).reshape(-1, 1) + (si[-1] - 2 * si)

    v = np.hstack((x, y, x * y))
    nw = v.shape[1]

    idx = np.vstack((np.arange(n), np.zeros(n))).astype(np.int64).T
    iv1 = np.zeros((n, 1))
    iv2 = np.zeros((n, 1))
    iv3 = np.zeros((n, 1))
    iv4 = np.zeros((n, 1))

    i = 1
    r = 0
    s = 1
    while i < n:
        gap = 2 * i
        k = 0
        idx_r = idx[:, r]
        csumv = np.vstack((np.zeros((1, nw)), cpu_cumsum_cpy(v[idx_r, :])))

        for j in range(1, n + 1, gap):
            st1 = j - 1
            e1 = min(st1 + i - 1, n - 1)
            st2 = j + i - 1
            e2 = min(st2 + i - 1, n - 1)

            while (st1 <= e1) and (st2 <= e2):
                idx1 = idx_r[st1]
                idx2 = idx_r[st2]

                if y[idx1] >= y[idx2]:
                    idx[k, s] = idx1
                    st1 += 1
                else:
                    idx[k, s] = idx2
                    st2 += 1
                    iv1[idx2] += e1 - st1 + 1
                    iv2[idx2] += csumv[e1 + 1, 0] - csumv[st1, 0]
                    iv3[idx2] += csumv[e1 + 1, 1] - csumv[st1, 1]
                    iv4[idx2] += csumv[e1 + 1, 2] - csumv[st1, 2]
                k += 1

            if st1 <= e1:
                kf = k + e1 - st1 + 1
                idx[k:kf, s] = idx_r[st1 : e1 + 1]
                k = kf
            elif st2 <= e2:
                kf = k + e2 - st2 + 1
                idx[k:kf, s] = idx_r[st2 : e2 + 1]
                k = kf

        i = gap
        r = 1 - r
        s = 1 - s

    covterm = np.sum(n * (x - np.mean(x)).T @ (y - np.mean(y)))
    c1 = np.sum(iv1.T @ v[:, 2].copy())
    c2 = np.sum(iv4)
    c3 = np.sum(iv2.T @ y)
    c4 = np.sum(iv3.T @ x)
    d = 4 * ((c1 + c2) - (c3 + c4)) - 2 * covterm

    y_sorted = y[idx[n::-1, r], :]
    si = cpu_cumsum_cpy(y_sorted)
    by = np.zeros((n, 1))
    by[idx[::-1, r]] = (np.arange(-(n - 2), n + 1, 2) * y_sorted.ravel()).reshape(
        -1, 1
    ) + (si[-1] - 2 * si)

    if bias:
        denom = [n**2, n**3, n**4]
    else:
        denom = [n * (n - 3), n * (n - 3) * (n - 2), n * (n - 3) * (n - 2) * (n - 1)]

    stat = np.sum(
        (d / denom[0])
        + (np.sum(ax) * np.sum(by) / denom[2])
        - (2 * (ax.T @ by) / denom[1])
    )

    return stat


# =============================================================================
# =============================================================================
def calc_fast_dcov(distx, disty):
    '''
    '''

    res = fast_1d_dcov_cpy(distx, disty, bias=False)

    return res
