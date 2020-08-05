#                   ------------------------------------------
#                     Observations.py (SPARTA Auxil class)
#                   ------------------------------------------
# This file defines the "Observations" class. An object of this class represents an astronomical observations
# time series object.  It does that either by reading an entire object observation directory, or by receiving
# a TimeSeries object when initiate.
# This class also enables running GLS and Zucker PDC and USuRPer on the data (arxiv.org/pdf/1711.06075.pdf) using a
# PeriodicityDetector class.
# Directory fits reading is currently implemented for HARPS, APOGEE, NRES, ELODIE and LAMOST observations data.
#
# Observations class stores the following methods:
# TBD tbd finish documentation
# ---------------------------------------------
# 1) convert_times_to_relative_float_values - converts the observation dates to float values,
#         marking the days past the first visit
#
# Dependencies: os and easygui, tqdm, numpy.
# Last update: Avraham Binnenfeld, 20200611.

import os, sys
import numpy as np
from sparta.UNICOR.Spectrum import Spectrum
from sparta.Auxil.ReadSpec import ReadSpec
from sparta.Auxil.TimeSeries import TimeSeries
from sparta.Auxil.PeriodicityDetector import PeriodicityDetector
import matplotlib.pyplot as plt
import easygui
from tqdm import tqdm


class Observations:
    # =============================================================================
    # =============================================================================
    def __init__(self, read_function=None, survey=None, min_snr=-1, target_visits_lib=None, time_series=[]):
        '''
        Input: All input is optional, and needs to be called along
               with its keyword. Below appears a list of the possible input
               variables. In case no input was provided, the function loads
               the directory path using easygui.

              survey: str, source name.
              min_snr: float, can be used to sort out noisy observations
              target_visits_lib: str, observation visit directory path
              time_series: TimeSeries object, in case a direct load of TimeSeries (instead of file reading)
        '''

        assert ((read_function is not None) or (survey is not None)), "Provide read_function or survey"

        if time_series == []:
            self.file_list = []
            self.spec_list = []
            self.time_list = []
            self.first_time = []
            self.vrad_list = []
            self.bcv = []
            self.snr = []

            if target_visits_lib:
                path = target_visits_lib
            else:
                path = easygui.diropenbox(msg=None, title='Select ' + survey + ' dir:',)

            # for filename in os.listdir(path):
            #     if filename.endswith(".fits") or filename.endswith(".fits.gz"):
            #         self.visits_list.append(os.path.join(path, filename))
            #     else:
            #         pass

            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".fits") or file.endswith(".fits.gz"):
                        self.file_list.append(os.path.join(root, file))
                    else:
                        pass

            # self.sample_size = len(self.file_list)

            for visit_path in self.file_list:

                if read_function is None:
                    visit = ReadSpec(survey=survey)
                else:
                    visit = ReadSpec(read_function=read_function)

                visit.load_spectrum_from_fits(path=visit_path)
                w, s, bool_mask_s, date_obs, vrad, bcv, snr = visit.retrieve_all_spectrum_parameters()

                if snr >= min_snr or snr == -1 or min_snr == -1:
                    if visit.bcv==[]:
                        bcv = 0.0
                    sp = Spectrum(wv=w, sp=s, bjd=date_obs, bcv=bcv, name=survey)
                    sp.BarycentricCorrection()

                    self.spec_list.append(sp)
                    self.time_list.append(date_obs)
                    self.bcv.append(bcv)
                    self.snr.append(snr)

                    if vrad != []:
                        self.vrad_list.append(vrad)

            self.sample_size = len(self.bcv)

            self.convert_times_to_relative_float_values()

            self.observation_TimeSeries = TimeSeries(size=self.sample_size, times=self.time_list,
                                                     vals=self.spec_list)
        else:
            self.observation_TimeSeries = time_series
            self.spec_list = time_series.vals
            self.time_list = time_series.times

            self.sample_size = self.observation_TimeSeries.size

        self.periodicity_detector = []

# =============================================================================
# =============================================================================
    def PreProccessSpectra(self, Ntrim=10, CleanMargins=True, RemoveNaNs=True,
                           delta=0.5, RemCosmicNum=3, FilterLC=3, FilterHC=0.15,
                           alpha=0.3, verbose=True):
        '''
        TBD tbd finish documentation
        :param Ntrim:
        :param CleanMargins:
        :param RemoveNaNs:
        :param delta:
        :param RemCosmicNum:
        :param FilterLC:
        :param FilterHC:
        :param alpha:
        :param verbose:
        :return:
        '''
        if not verbose:
            # Disable output
            sys.stdout = open(os.devnull, 'w')

        with tqdm(total=self.sample_size) as pbar:
            for s in self.spec_list:

                s.SpecPreProccess(Ntrim=Ntrim, CleanMargins=CleanMargins, RemoveNaNs=RemoveNaNs,
                                delta=delta, RemCosmicNum=RemCosmicNum, FilterLC=FilterLC,
                                FilterHC=FilterHC, alpha=alpha)
                pbar.update(1)

        if not verbose:
            # Enable output:
            sys.stdout = sys.__stdout__

        return self

# =============================================================================
# =============================================================================
    def SelectOrders(self, orders, remove=True):
        '''
        TBD tbd finish documentation
        :param orders:
        :param remove:
        :return:
        '''
        for s in self.spec_list:
            s.SelectOrders(orders, remove=remove)

# =============================================================================
# =============================================================================
    def calc_rv_against_template(self, template, dv=0.1, VelBound=150,
                                 err_per_ord=False, combine_ccfs=True, fastccf=False):
        '''
        TBD tbd finish documentation
        :param template:
        :param dv:
        :param VelBound:
        :param err_per_ord:
        :param combine_ccfs:
        :return:
        '''

        ccfs = (self.observation_TimeSeries.
                calc_rv_against_template(self.spec_list,
                                         template,
                                         dv=dv,
                                         VelBound=VelBound,
                                         err_per_ord=err_per_ord,
                                         combine_ccfs=combine_ccfs,
                                         fastccf=fastccf)
                )

        vels = np.zeros(self.sample_size)
        evels = np.zeros_like(vels)
        ccf_peaks = np.zeros_like(vels)

        try:
            for I, ccf in enumerate(ccfs):
                vels[I] = ccf.CorrCombined['RV']
                evels[I] = ccf.CorrCombined['eRV']
                ccf_peaks[I] = ccf.CorrCombined['peakCorr']

        except AttributeError:
            for I, ccf in enumerate(ccfs):
                vels[I] = np.average(ccf.Corr['RV'])
                evels[I] = np.std(ccf.Corr['RV'])
                ccf_peaks[I] = ccf.Corr['peakCorr']

        self.ccf_list = ccfs
        self.vels = vels
        self.evels = evels
        self.ccf_peaks = ccf_peaks

        return self

# =============================================================================
# =============================================================================
    def retrieve_BIS(self, bisect_val=[0.35, 0.95], use_combined=True):
        '''
        After the cross correlations were calculated, the BIS can be computed.

        :param: bisect_val - the values at which to calculate the
        :param: use_combined - Boolean. Use the combined CCFs or multiorder.
        :return: self - added the BIS values.
        '''
        BIS, eBIS = (self.observation_TimeSeries.
                     getBIS(self.ccf_list,
                            bisect_val=bisect_val,
                            use_combined=use_combined)
                     )

        self.BIS = BIS
        self.eBIS = eBIS
        return self

# =============================================================================
# =============================================================================
    def initialize_periodicity_detector(self, freq_range=(1/1000, 1), periodogram_grid_resolution=1000):
        """
        This function initiate the object's PeriodicityDetector instance before using it to create periodograms.
        :param
        periodogram_grid_resolution: float, the periodogram grid resolution
        freq_range: float tuple, the periodogram frequency range
        """
        self.periodicity_detector = PeriodicityDetector(time_series=self.observation_TimeSeries, freq_range=freq_range,
                                                 periodogram_grid_resolution=periodogram_grid_resolution)

# =============================================================================
# =============================================================================
    def convert_times_to_relative_float_values(self):
        """
        This function converts the observation dates to float values,
        marking the days past the first visit.
        """
        float_times = [0 for _ in range(self.sample_size)]
        start_time = int(min(self.time_list))
        self.first_time = start_time
        for i, t in enumerate(self.time_list):
            float_times[i] = (t - start_time)
        self.time_list = float_times


if __name__ == '__main__':
    '''
    An examploratory code showing reading of an HARPS or an APOGEE observation directory,
    creating GLS and PDC periodograms for it and plot them.
    '''
    # obs_data = Observations(survey="LAMOST")

    # Alternatively:
    # obs_data = Observations(survey="HARPS")

    # import scipy.io
    #
    # mat = scipy.io.loadmat(r"C:\Users\AbrahamBini\Downloads\specAll_17obs.mat")
    # len(mat["HD235679_20180531"][0][0])
    obs_data = Observations(survey="UVES")

    obs_data.initialize_periodicity_detector(freq_range=(0, 10), periodogram_grid_resolution=500)
    # obs_data.initialize_periodicity_detector(freq_range=(0, 10), periodogram_grid_resolution=100)

    obs_data.periodicity_detector.run_USURPER_process(calc_biased_flag=False, calc_unbiased_flag=True)

    obs_data.periodicity_detector.periodogram_plots()

    plt.show()