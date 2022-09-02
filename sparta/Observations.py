#                   ------------------------------------------
#                     Observations.py (SPARTA Auxil class)
#                   ------------------------------------------
# This file defines the "Observations" class. An object of this class represents an astronomical observations
# time series object.  It does that either by reading an entire object observation directory, or by receiving
# a TimeSeries object when initiate.
# This class also enables running GLS and Zucker PDC and USuRPer on the data (arxiv.org/pdf/1711.06075.pdf) using a
# PeriodicityDetector class.
# Directory fits reading is currently implemented for observations from HARPS, APOGEE, NRES, ELODIE and LAMOST and more.
#
# Observations class stores the following methods:
# ---------------------------------------------
# 1) PreProccessSpectra - contains different pre-processing procedures applied on an observed spectrum.
# 2) SelectOrders - enables the user to keep or discard orders from the observed data.
# 3) calc_rv_against_template - calculates radial velocity for the entire spectrum time series,
#                               using CCF against resting template.
# 4) retrieve_BIS - computes BIS from the calculated CCF
# 5) initialize_periodicity_detector -
#                    initiate the object's PeriodicityDetector instance before using it to create periodograms.
# 6) convert_times_to_relative_float_values -
#                            converts the observation dates to float values, marking the days past the first visit.
# 7) create_avg_template - This function creates a naive toy resting template, by averaging all spectrum observations.
# 8) clean_time_series - enables systematically removing observation data.
#
#
# Dependencies: os, sys, easygui, tqdm, numpy and matplotlib.
# Last update: Sahar Shahaf, 20220327.

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
    def __init__(self, survey="APOGEE", min_snr=-1, target_visits_lib=None, time_series=[],
                 min_avg_flux_val=-1, sample_rate=1, output=False, min_wv=5150, max_wv=5300):
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
        if time_series == []:
            self.file_list = []
            self.spec_list = []
            self.time_list = []
            self.first_time = []
            self.vrad_list = []
            self.bcv = []
            self.snr = []
            self.air_mass = []

            if target_visits_lib:
                path = target_visits_lib
            else:
                path = easygui.diropenbox(msg=None, title='Select ' + survey + ' dir:',)

            count = 0

            for root, dirs, files in os.walk(path):
                for file in files:
                    if (file.endswith(".fits") or file.endswith(".fits.gz") or file.endswith(".fts.gz") or file.endswith(".fits.Z")) and not file.startswith('~'):
                        if count % sample_rate == 0:
                            self.file_list.append(os.path.join(root, file))
                        count = count + 1
                    else:
                        pass

            for visit_path in self.file_list:

                visit = ReadSpec(survey)
                visit.load_spectrum_from_fits(path=visit_path, output=output, min_wv=min_wv, max_wv=max_wv)
                w, s, bool_mask_s, date_obs, vrad, bcv, snr, air_mass = visit.retrieve_all_spectrum_parameters()

                if (snr >= min_snr or snr == -1 or min_snr == -1) and (min_avg_flux_val == -1 or min_avg_flux_val <= np.mean(s)):
                    if visit.bcv==[]:
                        bcv = 0.0
                    sp = Spectrum(wv=w, sp=s, bjd=date_obs, bcv=bcv, name=survey)
                    sp.BarycentricCorrection()

                    self.spec_list.append(sp)
                    self.time_list.append(date_obs)
                    self.bcv.append(bcv)
                    self.snr.append(snr)
                    self.air_mass.append(air_mass)

                    if vrad != []:
                        self.vrad_list.append(vrad)

            self.sample_size = len(self.bcv)

            self.convert_times_to_relative_float_values()

            self.time_series = TimeSeries(size=self.sample_size, times=self.time_list,
                                          vals=self.spec_list)
        else:
            self.time_series = time_series
            self.spec_list = time_series.vals
            self.time_list = time_series.times

            self.sample_size = self.time_series.size

        self.periodicity_detector = []

# =============================================================================
# =============================================================================
    def PreProccessSpectra(self, Ntrim=10, CleanMargins=True, RemoveNaNs=True,
                           delta=0.5, RemCosmicNum=3, FilterLC=3, FilterHC=0.15,
                           alpha=0.3, verbose=True):
        '''
        This function conatains different pre-processing procedures applied on an observed spectrum.
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
        This routine enables the user to keep or discard orders from the observed
        data. Orders are numbered according to their index (0-> #ord-1),

        :param orders: Indices of the required orders
        :param include: Boolean. If True, these orders will be removed.
                        otherwise, these will be the orders that will
                        be kept.
        :return: Same structure, with the orders arranged.
        '''
        for s in self.spec_list:
            s.SelectOrders(orders, remove=remove)

# =============================================================================
# =============================================================================
    def calc_rv_against_template(self, template, dv=0.1, VelBound=150,
                                 err_per_ord=False, combine_ccfs=True, fastccf=False):
        '''
        This function calculates radial velocity for the entire spectrum time series, using CCF against resting template.
        :param: spec_list - a list of Spectrum objects, with the observed data.
        :param: template  - a Template object, with which the observaations are correlated.
        :param: dv, VelBound, err_per_ord - parameters for the CCF1d CrossCorrelateSpec routine
                                            see documentation therein.

        :return: ccfs - A list of CCF1d objects, one for each observation.

        '''

        ccfs = (self.time_series.
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
        This function computes BIS from the calculated CCF.

        :param: bisect_val - the values at which to calculate the
        :param: use_combined - Boolean. Use the combined CCFs or multiorder.
        :return: self - added the BIS values.
        '''
        BIS, eBIS = (self.time_series.
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
        self.periodicity_detector = PeriodicityDetector(time_series=self.time_series, freq_range=freq_range,
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

# =============================================================================
# =============================================================================
    def create_avg_template(self):
        """
        This function creates a naive toy resting template, by averaging all spectrum observations.
        """
        float_times = [0 for _ in range(self.sample_size)]
        start_time = int(min(self.time_list))
        self.first_time = start_time
        for i, t in enumerate(self.time_list):
            float_times[i] = (t - start_time)
        self.time_list = float_times

# =============================================================================
# =============================================================================
    def clean_time_series(self, max_vel=[], min_vel=[], max_time=[], nan_flag=True, sample_rate=1):
        """
        This functions enables systematically removing observations, according to either min or max values, max time,
        nan values or sampling rate.
        """

        self.air_mass = self.time_list

        if sample_rate != 1:
            ok_index = []
            for i in range(len(self.time_series.times)):
                if sample_rate > 0 :
                    if i % sample_rate == 0:
                        ok_index.append(True)
                    else:
                        ok_index.append(False)
                elif sample_rate < 0:
                    if i % abs(sample_rate) == 0:
                        ok_index.append(False)
                    else:
                        ok_index.append(True)
            ok_index = np.where(ok_index)[0]
            self.time_series.times = [self.time_series.times[i] for i in ok_index]
            self.time_series.vals = [self.time_series.vals[i] for i in ok_index]
            self.time_series.calculated_vrad_list = [self.time_series.calculated_vrad_list[i] for i in
                                                     ok_index]
            self.air_mass = [self.air_mass[i] for i in ok_index]


        if nan_flag:
            ok_index = np.where(~np.isnan(self.time_series.calculated_vrad_list))[0]
            self.time_series.times = [self.time_series.times[i] for i in ok_index]
            self.time_series.vals = [self.time_series.vals[i] for i in ok_index]
            self.time_series.calculated_vrad_list = [self.time_series.calculated_vrad_list[i] for i in ok_index]
            self.air_mass = [self.air_mass[i] for i in ok_index]

        if max_vel:

            ok_index = [i for i, x in enumerate(self.time_series.calculated_vrad_list) if x <= max_vel]
            self.time_series.times = [self.time_series.times[i] for i in ok_index]
            self.time_series.vals = [self.time_series.vals[i] for i in ok_index]
            self.time_series.calculated_vrad_list = [self.time_series.calculated_vrad_list[i] for i in ok_index]
            self.air_mass = [self.air_mass[i] for i in ok_index]

        if min_vel:

            ok_index = [i for i, x in enumerate(self.time_series.calculated_vrad_list) if x >= min_vel]
            self.time_series.times = [self.time_series.times[i] for i in ok_index]
            self.time_series.vals = [self.time_series.vals[i] for i in ok_index]
            self.time_series.calculated_vrad_list = [self.time_series.calculated_vrad_list[i] for i in ok_index]
            self.air_mass = [self.air_mass[i] for i in ok_index]

        if max_time:

            ok_index = [i for i, x in enumerate(self.time_series.times) if x <= max_time]
            self.time_series.times = [self.time_series.times[i] for i in ok_index]
            self.time_series.vals = [self.time_series.vals[i] for i in ok_index]
            self.time_series.calculated_vrad_list = [self.time_series.calculated_vrad_list[i] for i in ok_index]
            self.air_mass = [self.air_mass[i] for i in ok_index]


        self.sample_size = len(self.time_series.vals)
        self.time_series.size = self.sample_size
        self.spec_list = self.time_series.vals


if __name__ == '__main__':
    '''
    An examploratory code showing reading of an APOGEE observation directory,
    creating GLS and USURPER periodograms for it and plotting them.
    '''

    obs_data = Observations(survey="APOGEE")

    obs_data.initialize_periodicity_detector(freq_range=(0, 1), periodogram_grid_resolution=10_000)

    obs_data.periodicity_detector.run_USURPER_process(calc_biased_flag=False, calc_unbiased_flag=True)
    obs_data.periodicity_detector.run_GLS_process()

    obs_data.periodicity_detector.periodogram_plots()

    plt.show()