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

    # for i in range(size):
    #     for j in range(size):
    #         if i != j:
    #             sum_bin = sum_bin + a[i][j] * b[i][j]

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

    if periodicity_detector.method == "Partial_USURPER":
        semi = False
        if not semi:
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
def calc_pdc_distance_matrix(periodicity_detector, calc_biased_flag, calc_unbiased_flag, reverse_existing=False):
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
                                                         VelBound=[-1, 1], fastccf=True)
                        ccf_val = ccf.subpixel_CCF(ccf.Corr['vel'], ccf.Corr['corr'][0], 0)
                        a[i][j] = np.sqrt(1 - abs(min(ccf_val, 1))) * np.sqrt(2)

                    elif periodicity_detector.method == 'Partial_USURPER':
                        a[i][j] = abs(periodicity_detector.time_series.calculated_vrad_list[i] -
                                            periodicity_detector.time_series.calculated_vrad_list[j])

                        s = Spectrum(wv=[Template().doppler(-periodicity_detector.time_series.calculated_vrad_list[j], j_val.wv[0])],
                                     sp=j_val.sp) # .SpecPreProccess()
                        t = Template(spectrum=i_val.sp[0], wavelengths=i_val.wv[0])
                        t.model.wv = t.doppler(-periodicity_detector.time_series.calculated_vrad_list[i])

                        ccf = CCF1d().CrossCorrelateSpec(spec_in=s, template_in=t, dv=0.05,
                                                         VelBound=[-100, 100], fastccf=True)
                        # ccf_val = ccf.subpixel_CCF(ccf.Corr['vel'], ccf.Corr['corr'][0], 0)
                        ccf_val = ccf.subpixel_CCF(ccf.Corr['vel'], ccf.Corr['corr'][0])[1]
                        c[i][j] = np.sqrt(1 - abs(min(ccf_val, 1))) * np.sqrt(2)


        if periodicity_detector.method == 'Partial_USURPER' and periodicity_detector.reverse_partial_flag:
            temp = a.copy()
            a = c.copy()
            c = temp.copy()

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

        if periodicity_detector.method == 'Partial_USURPER':
            periodicity_detector.pdc_mat_C_unbiased = unbiased_u_centering(periodicity_detector.pdc_c, periodicity_detector.time_series.size)
            periodicity_detector.pdc_mat_A_unbiased = norm_z(mat_A_unbiased, periodicity_detector.pdc_mat_C_unbiased)
        else:
            periodicity_detector.pdc_mat_A_unbiased = mat_A_unbiased
    return a
