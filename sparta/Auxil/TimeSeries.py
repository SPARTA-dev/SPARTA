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
# 2) plot_velocities - plots the time series velocities.
# TBD

# ---------------------------------------------
# Last update: Sahar Shahaf, 20200527.

from sparta.UNICOR.CCF1d import CCF1d
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


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

        if 'period' in kwargs:
            self.period = kwargs['period']
        else:
            self.period = []

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
    def plot_velocities(self):
        '''
        This function plots the time series.
        '''
        if not isinstance(self.vals[0], float):
            assert 'plot_velocities is available for scalar values time-series only'
            return

        fig, axs = plt.subplots(2)
        axs[0].scatter(self.times, self.vals)
        axs[0].set_title("Time VS. Value")
        if self.period:
            times_phased = [t % self.period for t in self.times]
            axs[1].scatter(times_phased, self.vals)
            axs[1].set_title("Phase Folded Values")
        else:
            axs[1] = None
        plt.tight_layout()

        plt.show()

