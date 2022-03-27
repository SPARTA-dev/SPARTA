#                   ------------------------------
#                     TimeSeries.py (SPARTA class)
#                   ------------------------------
# This file defines the "TimeSeries" class. An object of this class stores
# measured time series data (spectrum or radial velocity values and their corresponding observation times).
# It is meant to be used by an Observations class, alongside a PeriodicityDetector class
# which runs GLS and Zucker PDC and USuRPer on the data (arxiv.org/pdf/1711.06075.pdf).
#
# A TimeSeries class stores the following methods:
# ---------------------------------------------
# 1) calc_rv_against_template - This function calculates radial velocity for the entire spectrum time series,
#                               using CCF against resting template
# 2) getBIS - Calculates the time series BIS, from the calculated CCFs.
# 3) plot_velocities - plots the time series velocities.
# 4) TIRAVEL - Implements Zucker & Mazeh 2006 TIRAVEL, enabling template-independent radial velocities measurement
#    given a ball-park RVs estimation.

# ---------------------------------------------
# Dependencies: os and easygui, tqdm, numpy, copy, scipy and matplotlib.
# Last update: Avraham Binnenfeld, 20210510.

from sparta.UNICOR.CCF1d import CCF1d
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sparta.UNICOR.Template import Template
from numpy import linalg as LA
from copy import deepcopy
from scipy import interpolate


class TimeSeries:

    # =============================================================================
    # =============================================================================
    def __init__(self, **kwargs):
        '''
        Input: All input is optional, and needs to be called along
               with its keyword. Below appears a list of the possible input
               variables.

              size: integer, number of data points.
              times: float list, observation times
              vals: list, observation values
        '''
        self.size = kwargs['size']
        self.times = kwargs['times']
        self.vals = kwargs['vals']

        if 'calculated_vrad_list' in kwargs:
            self.calculated_vrad_list = kwargs['calculated_vrad_list']
        else:
            self.calculated_vrad_list = []

        if 'fold_period' in kwargs:
            self.fold_period = kwargs['fold_period']
        else:
            self.fold_period = []

        self.calculated_ccf_peaks = []

    # =============================================================================
    # =============================================================================
    def calc_rv_against_template(self, spec_list, template, dv=0.1, VelBound=150,
                                 err_per_ord=False, combine_ccfs=True, fastccf=False):
        '''
        This function calculates radial velocity for the entire spectrum time series, using CCF against resting template.
        :param: spec_list - a list of Spectrum objects, with the observed data.
        :param: template  - a Template object, with which the observaations are correlated.
        :param: dv, VelBound, err_per_ord - parameters for the CCF1d CrossCorrelateSpec routine
                                            see documentation therein.

        :return: ccfs - A list of CCF1d objects, one for each observation.

        '''
        if isinstance(spec_list[0], float):
            assert 'RV calculation is available for spectrum time-series only'
            return

        calculated_vrad_list = []
        sample_size = len(spec_list)

        ccfs = []
        with tqdm(total=sample_size) as pbar:
            for spec in spec_list:
                ccf = CCF1d().CrossCorrelateSpec(template_in=template,
                                                 spec_in=spec,
                                                 dv=dv,
                                                 VelBound=VelBound,
                                                 err_per_ord=err_per_ord,
                                                 fastccf=fastccf)
                if combine_ccfs:
                    ccf.CombineCCFs()

                ccfs.append(ccf)
                pbar.update(1)

        return ccfs

    # =============================================================================
    # =============================================================================
    def getBIS(self, ccf_list, bisect_val=[0.35, 0.95], use_combined=True):
        '''
        This function calculated the BIS, from the calculated CCFs

        :param: ccf_list - list of CCF1d objects.
        :param: bisect_val - values in which to calculate the bisectors
        :param: use_combined - Boolean. if True, use the combined correlations

        :return:
        '''
        BIS = np.full(len(ccf_list), np.nan)
        eBIS = np.full(len(ccf_list), np.nan)

        if use_combined:
            for ind, c in enumerate(ccf_list):
                BIS[ind], eBIS[ind], _ = c.calcBIS(c.CorrCombined['vel'],
                                                   c.CorrCombined['corr'],
                                                   bisect_val=bisect_val,
                                                   n_ord=c.n_ord)
        else:
            for ind, c in enumerate(ccf_list):
                BISo = np.full(c.n_ord, np.nan)
                for co in c.CombinedCorr['corr']:
                    BISo[ind], _ = c.calcBIS(c.Corr['vel'],
                                             co,
                                             bisect_val=bisect_val,
                                             n_ord=1)
                BIS[ind] = np.average(BISo)
                eBIS[ind] = np.std(BISo)
        return BIS, eBIS

    # =============================================================================
    # =============================================================================
    def multiorder_systematics(self, ccf_list, v0_switch=0, plot_switch=False):
        '''
        This function calculates radial velocity for the entire spectrum time series, using CCF against resting template.
        Input:
              template_spectrum: Spectrum object, a resting template of the target for CCF peak measurement

        '''

    # =============================================================================
    # =============================================================================
    def plot_velocities(self, figsize=(7,7), sty='ok'):
        '''
        This function plots the time series.
        '''
        if not isinstance(self.vals[0], float):
            assert 'plot_velocities is available for scalar values time-series only'
            return

        fig, axs = plt.subplots(2, figsize=figsize)
        axs[0].plot(self.times, self.vals,sty)
        axs[0].set_title("Time VS. Value")
        if self.fold_period:
            times_phased = [t % self.fold_period for t in self.times]
            axs[1].plot(times_phased, self.vals,sty)
            axs[1].set_title("Phase Folded Values")
        else:
            axs[1] = None
        plt.tight_layout()

        plt.show()

    # =============================================================================
    # =============================================================================
    def TIRAVEL_vel_test(self, velocities=[]):
        '''
        This function is used by TIRAVEL() and should not be called independently.
        '''

        a = [[None for _ in range(self.size)] for _ in
             range(self.size)]

        vals_cpy = deepcopy(self.vals)

        for i, s in enumerate(vals_cpy):
            vals_cpy[i].wv[0] = Template().doppler(-velocities[i], deepcopy(s.wv[0]))

        grid_0 = deepcopy(vals_cpy[0])

        # interpolate to the same grid
        for i, s in enumerate(vals_cpy):
            if i == 0:
                new_spec = s.sp[0][100:-100]

                new_wl = s.wv[0][100:-100]

                vals_cpy[i].sp = [deepcopy(new_spec)]
                vals_cpy[i].wv = [deepcopy(new_wl)]

            else:
                z1 = interpolate.interp1d(s.wv[0], s.sp[0], kind='quadratic')

                new_spec = z1(grid_0.wv[0][100:-100])

                new_wl = grid_0.wv[0][100:-100]

                vals_cpy[i].sp = [deepcopy(new_spec)]
                vals_cpy[i].wv = [deepcopy(new_wl)]


        for i, i_val in enumerate(vals_cpy):
            for j, j_val in enumerate(vals_cpy):
                ccf = CCF1d()

                ccf.CrossCorrelateSpec(spec_in=deepcopy(j_val).TrimSpec(Ntrim=50), template_in=Template(template=i_val), dv=0.01,
                                                 VelBound=[-1, 1], fastccf=True)

                ccf_val = ccf.subpixel_CCF(ccf.Corr['vel'], ccf.Corr['corr'][0], 0)

                a[i][j] = ccf_val

        # max eigen value, eigen vector
        e_w, e_v = LA.eig(a)

        lambda_m = max(e_w)

        rho = (lambda_m - 1) / (self.size - 1)

        return e_w, e_v, rho, vals_cpy

    # =============================================================================
    # =============================================================================
    def TIRAVEL(self, velocities=[[]]):
        '''
        This function implements Zucker & Mazeh 2006 TIRAVEL, enabling template-independent
        radial velocities measurement given a ball-park RVs estimation. See https://arxiv.org/abs/astro-ph/0607293

        This function calculates radial velocity for the entire spectrum time series, using CCF against resting template.
        :param: velocities - a list of floats, a ball-park RVs estimation.
        :return: tiravel_res - A generated "template" that can be used to measure RV against.

        '''

        e_w_list = []
        e_v_list = []
        rho_list = []
        vals_cpy_list = []

        for v_list in velocities:
            e_w, e_v, rho, vals_cpy = self.TIRAVEL_vel_test(v_list)
            e_w_list.append(e_w)
            e_v_list.append(e_v)
            rho_list.append(rho)
            vals_cpy_list.append(vals_cpy)

        index_max = rho_list.index(max(rho_list))

        template_wv = vals_cpy_list[index_max][0].wv[0]
        template_sp = vals_cpy_list[index_max][0].sp[0]

        for i, w in enumerate(e_v_list[index_max].transpose()[0]):
            if i == 0:
                template_sp = template_sp * w.real ** 2
            else:
                template_sp = template_sp + vals_cpy_list[index_max][i].sp[0] * w.real ** 2

        tiravel_res = Template(wavelengths=[template_wv], spectrum=[template_sp])

        tiravel_res.model.InterpolateSpectrum(delta=1)

        return tiravel_res
