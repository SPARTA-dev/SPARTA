#                   ---------------------------------------------
#                     PeriodicityDetector.py (SPARTA USuRPer file)
#                   ---------------------------------------------
# This file defines the "PeriodicityDetector" class. An object of this class handles the "TimeSeries" class
# and enables running GLS and Zucker PDC and USuRPer on the data (arxiv.org/pdf/1711.06075.pdf).
#
# A PeriodicityDetector class stores the following methods:
# ---------------------------------------------
# 1) run_PDC_process - an 'overall' procedure runs the entire Zucker PDC calculation process
#                      using USuRPer_functions.py functions.
# 2) run_USURPER_process - an 'overall' procedure runs the entire Zucker USuRPer calculation process
#                      using USuRPer_functions.py functions.
# 3) calc_partial_periodogram - an 'overall' procedure runs the partial PDC.
# 4) run_GLS_process - runs the GLS calculation process.
# 5) plot_periodograms - plots the calculated periodograms.
#
# Dependencies: numpy, astropy, matplotlib, copy and random.
# Last update: Sahar Shahaf, 20220327.

import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from sparta.USURPER.USURPER_functions import calc_PDC, calc_PDC_unbiased, calc_pdc_distance_matrix
from scipy.stats.distributions import chi2

class PeriodicityDetector:

    # =============================================================================
    # =============================================================================
    def __init__(self, **kwargs):
        '''
        Input: All input is optional, and needs to be called along
               with its keyword. Below appears a list of the possible input
               variables.

              time_series: TimeSeries object, the time series for periodicity detection.

              freq_range: tuple, frequency grid boundaries
              periodogram_grid_resolution: int, number of frequency grid ticks
        '''
        if 'time_series' in kwargs:
            self.time_series = kwargs['time_series']
        if 'freq_range' and 'periodogram_grid_resolution' in kwargs:
            self.periodogram_grid_resolution = kwargs['periodogram_grid_resolution']
            self.freq_range = kwargs['freq_range']
            self.period_truth = None
            self.GLS_flag = False
            self.biased_PDC_flag = False
            self.unbiased_PDC_flag = False
            self.partial_flag = False
            self.partial_type = "shift"

        self.method = []

        self.results_frequency = {}
        self.results_power = {}

        # Initialize pdc process variables
        self.pdc_a = None
        self.pdc_c = None
        self.pdc_mat_A_biased = None
        self.pdc_mat_A_unbiased = None
        self.pdc_mat_C_unbiased = None

        # initialize PDC and GLS grids
        grid_template = np.arange(self.freq_range[0], self.freq_range[1], 1/self.periodogram_grid_resolution)

        self.pdc_res_freqs = grid_template.copy()
        self.pdc_res_power_biased = grid_template.copy()
        self.pdc_res_power_unbiased = grid_template.copy()
        self.GLS_power = grid_template.copy()
        self.GLS_frequency = grid_template.copy()

        self.fap_dict = {}

    # =============================================================================
    # =============================================================================
    def calc_PDC(self, calc_biased_flag=False, calc_unbiased_flag=True, errors=False):
        '''
        This function runs the entire PDC calculation process.
        Input:
              calc_biased_flag: bool, sets running the biased PDC
              calc_unbiased_flag: bool, sets running the unbiased PDC
        '''

        temp = []

        if isinstance(self.time_series.vals[0], float) or isinstance(self.time_series.vals[0], tuple) or isinstance(self.time_series.vals[0], list):
            self.method = "PDC"
        elif self.time_series.calculated_vrad_list != []:
            self.method = "PDC"
            temp = self.time_series.vals.copy()
            self.time_series.vals = self.time_series.calculated_vrad_list
        else:
            assert 'PDC is available for scalar time-series only'
            return


        self.biased_PDC_flag = calc_biased_flag
        self.unbiased_PDC_flag = calc_unbiased_flag

        calc_pdc_distance_matrix(self, calc_biased_flag, calc_unbiased_flag, errors=errors)

        for index, f in enumerate(self.pdc_res_freqs):
            if calc_biased_flag:
                self.pdc_res_power_biased[index] = calc_PDC(self, f)
            if calc_unbiased_flag:
                self.pdc_res_power_unbiased[index] = calc_PDC_unbiased(self, f)


        if calc_biased_flag:
            self.results_power.update({self.method + "_biased": self.pdc_res_power_biased.copy()})
            self.results_frequency.update({self.method + "_biased": self.pdc_res_freqs.copy()})
        if calc_unbiased_flag:
            self.results_power.update({self.method + "_unbiased": self.pdc_res_power_unbiased.copy()})
            self.results_frequency.update({self.method + "_unbiased": self.pdc_res_freqs.copy()})

        if temp:
            self.time_series.vals = temp


    # =============================================================================
    # =============================================================================
    def calc_USURPER(self, calc_biased_flag=False, calc_unbiased_flag=True):
        '''
        This function runs the entire USURPER calculation process.
        Input:
              calc_biased_flag: bool, sets running the biased PDC
              calc_unbiased_flag: bool, sets running the unbiased PDC
        '''

        if not isinstance(self.time_series.vals[0], float):
            self.method = "USURPER"
        else:
            assert 'USURPER is available for spectrum time-series only'
            return

        self.biased_PDC_flag = calc_biased_flag
        self.unbiased_PDC_flag = calc_unbiased_flag

        calc_pdc_distance_matrix(self, calc_biased_flag, calc_unbiased_flag)

        for index, f in enumerate(self.pdc_res_freqs):
            if calc_biased_flag:
                self.pdc_res_power_biased[index] = calc_PDC(self, f)
            if calc_unbiased_flag:
                self.pdc_res_power_unbiased[index] = calc_PDC_unbiased(self, f)

        if calc_biased_flag:
            self.results_power.update({self.method + "_biased": self.pdc_res_power_biased.copy()})
            self.results_frequency.update({self.method + "_biased": self.pdc_res_freqs.copy()})

        elif calc_unbiased_flag:
            self.results_power.update({self.method: self.pdc_res_power_unbiased.copy()})
            self.results_frequency.update({self.method: self.pdc_res_freqs.copy()})

    # =============================================================================
    # =============================================================================
    def calc_partial_periodogram(self, partial_type="shift", reverse_existing=False):
        '''
        Currently under development.
        Input:
              calc_biased_flag: bool, sets running the biased PDC
              calc_unbiased_flag: bool, sets running the unbiased PDC
        '''

        self.partial_type = partial_type

        if not isinstance(self.time_series.vals[0], float):
            if partial_type == "shift":
                self.method = "shift_periodogram"
            elif partial_type == "shape":
                self.method = "shape_periodogram"
        else:
            assert 'Partial_periodogram is available for spectrum time-series only'
            return

        calc_pdc_distance_matrix(self, calc_biased_flag=False, calc_unbiased_flag=True, reverse_existing=reverse_existing)

        for index, f in enumerate(self.pdc_res_freqs):
            self.pdc_res_power_unbiased[index] = calc_PDC_unbiased(self, f)

        self.results_frequency.update({self.method: self.pdc_res_freqs.copy()})
        self.results_power.update({self.method: self.pdc_res_power_unbiased.copy()})

    # =============================================================================
    # =============================================================================
    def calc_GLS(self):
        '''
        This function runs the GLS calculation process.
        '''

        if isinstance(self.time_series.vals[0], float):
            self.GLS_power = LombScargle(self.time_series.times, self.time_series.vals).power(self.GLS_frequency)
        else:
            if self.time_series.calculated_vrad_list != []:
                self.GLS_power = LombScargle(self.time_series.times, self.time_series.calculated_vrad_list).power(self.GLS_frequency)
            else:
                assert 'GLS is possible for scalar time-series only'
                return

        self.GLS_flag = True

        self.results_frequency.update({"GLS": self.GLS_frequency})
        self.results_power.update({"GLS": self.GLS_power})

        return self.GLS_frequency, self.GLS_power

    # =============================================================================
    # =============================================================================
    def calc_pdc_pval(self, values, inverse_sf=False):
        '''
        This function returns the p-value of a given peak value in the PDC periodogram,
        based on Shen et. al. 2019 (https://arxiv.org/abs/1912.12150).
        '''

        N = len(self.time_series.times)
        if not inverse_sf:
            output = np.array(
                [chi2.sf(val * N + 1, 1) for val in values]
            )
        else:
            output = np.array(
                [(chi2.isf(val, 1)-1)/N for val in values]
            )
        return output

    # =============================================================================
    # =============================================================================
    def plot_periodograms(self, plot_vals=False, annotate_pval=None, figsize=(11, 4)):
        '''
        This function plots the calculated periodograms.
        '''

        index = 0
        count = len(self.results_frequency) + plot_vals
        colors = ['red', 'orange', 'blue', 'green', 'purple', 'dodgerblue']
        if annotate_pval is not None:
            annotate_levels = self.calc_pdc_pval(annotate_pval, inverse_sf=True)

        fig, axs = plt.subplots(count, squeeze=False,
                                figsize=(figsize[0], count * figsize[1]))

        for method in self.results_frequency:
            axs[index, 0].plot(self.results_frequency[method],
                               self.results_power[method], 'k', linewidth=1.0)
            l = [method]
            if (annotate_pval is not None) and (method != 'GLS') and \
                    (method != "USURPER_biased") and (method != "PDC_biased"):
                for anlev_ind, anlev in enumerate(annotate_levels):
                    axs[index, 0].hlines(y=anlev, xmin=self.freq_range[0], xmax=self.freq_range[1],
                                         linewidth=0.5, alpha=0.5, ls='--', color=colors[anlev_ind % 6])
                    l.append(f"p-value: {annotate_pval[anlev_ind]:.2g}")
            axs[index, 0].legend(l)

            if self.period_truth is not None:
                for p in self.period_truth:
                    axs[index, 0].axvline(x=1/p, alpha=0.5, ls='--')
            index = index + 1



        if plot_vals:
            if self.period_truth == None:
                self.period_truth = [999_999]

            times_folded = [t % self.period_truth[0] for t in self.time_series.times]

            if isinstance(self.time_series.vals[0], float):
                v_list = self.time_series.vals
            else:
                if self.time_series.calculated_vrad_list != []:
                    v_list = self.time_series.calculated_vrad_list

            axs[index, 0].scatter(times_folded, v_list, alpha=0.6)
            axs[index, 0].set(xlabel="time (days)", ylabel="Vrad (km/s)")
            axs[index, 0].set_ylim(min(v_list), max(v_list))

        if index == 0:
            axs[index, 0].set(xlabel="Frequency [1/day]")
        else:
            axs[count - 1 - plot_vals, 0].set(xlabel="Frequency [1/day]")
