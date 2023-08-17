import math
import numpy as np
from scipy import special

def calc_PDC_unbiased(f, A, vals, times, errors=[]):
    '''
    This function calculates the unbiased PDC value for a given frequency.
    Input:
          periodicity_detector: PeriodicityDetector class for which the PDC is calculated
          f: float, a specific frequency value for it the PDC value is calculated
    '''

    z = np.array([times])
    mat_phase_delta = z.T - z
    mat_phase_delta = mat_phase_delta % (1 / f)

    b = mat_phase_delta * (1 / f - mat_phase_delta)

    mat_A_unbiased = A
    mat_B_unbiased = unbiased_u_centering(b, len(vals))

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

def unbiased_u_centering(a, n):
    '''
    :param a: U-centered matrix (numpy nXn array)
    :param n: integer (length of a...)
    :return: U-center a distance matrix
    '''

    # Sum of rows
    v2_vec = np.array([np.sum(x) / (n - 2) for x in a])
    v2     = v2_vec.repeat(n).reshape((-1, n))

    # Sum of cols
    v3_vec = np.array([np.sum(x) / (n - 2) for x in a.T])
    v3 = v3_vec.repeat(n).reshape((-1, n)).T

    # Total sum
    tot_sum = np.sum(v3_vec)/(n-1)
    v4 = np.full((n, n), tot_sum)

    # Do the U-centering and make sure that the diagonal is zero.
    mat_unbiased = a - v2 - v3 + v4
    for i in np.arange(n):
        mat_unbiased[i,i] = 0

    return mat_unbiased


def calc_pdc_distance_matrix(vals, errors=[]):
    '''
    This function calculates the distance matrix used for the PDC calculation.
    Input:
          periodicity_detector: PeriodicityDetector class for which the PDC is calculated
          calc_biased_flag: bool, sets calculating the distance matrix for the biased PDC
          calc_unbiased_flag: bool, sets calculating the distance matrix for the unbiased PDC
    '''

    a = [[None for _ in range(len(vals))] for _ in
         range(len(vals))]

    for i, i_val in enumerate(vals):
        for j, j_val in enumerate(vals):
            if i > j:
                a[i][j] = a[j][i]
            elif i == j:
                a[i][j] = 0
            else:
                if len(errors)==0:
                    a[i][j] = abs(i_val - j_val)
                else:
                    y1 = i_val
                    y2 = j_val
                    e1 = errors[i]
                    e2 = errors[j]

                    den = np.sqrt(2 * (e1 ** 2 + e2 ** 2))

                    x = (y1 - y2) / den
                    y = (e1 + e2) / den

                    e = np.sqrt(e1 ** 2 + e2 ** 2) * (math.exp(-x ** 2) + x * special.erf(x) - y)

                    a[i][j] = np.sqrt(e)


    A = unbiased_u_centering(np.array(a), len(vals))

    return A, a


def calc_pdc(freqs, times, vals, errors=[]):
    
    A, a = calc_pdc_distance_matrix(vals, errors)

    pdc_res_power_unbiased = freqs.copy()
    
    for index, f in enumerate(freqs):
        pdc_res_power_unbiased[index] = calc_PDC_unbiased(f, A, vals, times)

    return A, a, pdc_res_power_unbiased
