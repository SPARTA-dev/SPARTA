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
# 3) run_Partial_USURPER_process - an 'overall' procedure runs the partial PDC. Currently under development.
# 4) run_GLS_process - runs the GLS calculation process.
# 5) periodogram_plots - plots the calculated periodograms.
#
# Dependencies: numpy, astropy and matplotlib.
# Last update: Avraham Binnenfeld, 20200316.

import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from sparta.USURPER.USURPER_functions import calc_PDC, calc_PDC_unbiased, calc_pdc_distance_matrix


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
            self.period = None
            self.GLS_flag = False
            self.biased_PDC_flag = False
            self.unbiased_PDC_flag = False
            self.partial_flag = False
            self.reverse_partial_flag = False

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

    # =============================================================================
    # =============================================================================
    def run_PDC_process(self, calc_biased_flag=True, calc_unbiased_flag=False):
        '''
        This function runs the entire PDC calculation process.
        Input:
              calc_biased_flag: bool, sets running the biased PDC
              calc_unbiased_flag: bool, sets running the unbiased PDC
        '''

        temp = []

        if isinstance(self.time_series.vals[0], float):
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

        calc_pdc_distance_matrix(self, calc_biased_flag, calc_unbiased_flag)

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
    def run_USURPER_process(self, calc_biased_flag=False, calc_unbiased_flag=True):
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
    def run_Partial_USURPER_process(self, reversed_flag=False):
        '''
        Currently under development.
        Input:
              calc_biased_flag: bool, sets running the biased PDC
              calc_unbiased_flag: bool, sets running the unbiased PDC
        '''

        self.reverse_partial_flag = reversed_flag

        if not isinstance(self.time_series.vals[0], float):
            self.method = "Partial_USURPER"
        else:
            assert 'Partial_USURPER is available for spectrum time-series only'
            return

        calc_pdc_distance_matrix(self, calc_biased_flag=False, calc_unbiased_flag=True)

        for index, f in enumerate(self.pdc_res_freqs):
            self.pdc_res_power_unbiased[index] = calc_PDC_unbiased(self, f)

        if reversed_flag:
            self.results_frequency.update({self.method + "_reversed": self.pdc_res_freqs.copy()})
            self.results_power.update({self.method + "_reversed": self.pdc_res_power_unbiased.copy()})
        else:
            self.results_frequency.update({self.method: self.pdc_res_freqs.copy()})
            self.results_power.update({self.method: self.pdc_res_power_unbiased.copy()})

    # =============================================================================
    # =============================================================================
    def run_GLS_process(self):
        '''
        This function runs the GLS calculation process.
        '''

        # , normalization='psd'

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
    def periodogram_plots(self, velocities_flag=False):
        '''
        This function plots the calculated periodograms.
        '''

        periodograms_count = len(self.results_frequency) + velocities_flag

        index = 0

        fig, axs = plt.subplots(periodograms_count, squeeze=False, figsize=(11, 16))

        for method in self.results_frequency:
            axs[index, 0].plot(self.results_frequency[method], self.results_power[method], 'k')
            axs[index, 0].set_title(method)
            if self.period != None:
                for p in self.period:
                    axs[index, 0].axvline(x=1/p, alpha=0.5, ls='--') # , c='r'

                pass
            index = index + 1

        axs[index-1, 0].set(xlabel="Frequency [1/day]")

        if velocities_flag:
            if self.period == None:
                self.period = 9_999

            times_folded = [t % self.period[0] for t in self.time_series.times]

            if isinstance(self.time_series.vals[0], float):
                v_list = self.time_series.vals
            else:
                if self.time_series.calculated_vrad_list != []:
                    v_list = self.time_series.calculated_vrad_list
            axs[index, 0].scatter(times_folded, v_list, alpha=0.6)
            axs[index, 0].set(xlabel="time (days)", ylabel="Vrad (km/s)")
            axs[index, 0].set_title("Vrad")
            axs[index, 0].set_ylim(min(v_list), max(v_list))

        plt.tight_layout()