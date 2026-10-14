#                   ------------------------------
#                     TimeSeries.py (SPARTA class)
#                   ------------------------------
# This file defines the "TimeSeries" class. An object of this class stores
# measured time series data (spectrum or radial velocity values and their corresponding observation times).
# It is meant to be used by an Observations class, alongside a PeriodicityDetector class
# which runs GLS and Zucker PDC and USuRPer on the data (arxiv.org/pdf/1711.06075.pdf).
#
# Conventions: velocities in km/s, times in BJD (days).
#
# A TimeSeries class stores the following methods:
# ---------------------------------------------
# 1) calc_rv_against_template - This function calculates radial velocity for the entire spectrum time series,
#                               using CCF against resting template
# 2) calc_rv_against_two_templates - The double-lined (SB2) counterpart: TODCOR
#    against two templates, one (v1, v2) fit per epoch.
# 3) getBIS - Calculates the time series BIS, from the calculated CCFs.
# 4) plot_velocities - plots the time series velocities.
# 5) build_tiravel_template - Zucker & Mazeh 2006 TIRAVEL: builds a template out
#    of the observations themselves, given a set of trial velocities.

# ---------------------------------------------
# Dependencies: os, tqdm, numpy and matplotlib.
# Last update: Sahar Shahaf, 20220327.

from sparta.UNICOR.CCF1d import CCF1d
from sparta.TODCOR.CCF2d import CCF2d
from sparta.TIRAVEL.TIRAVEL import TIRAVEL
from sparta.Auxil.PeriodicityDetector import is_numeric_scalar
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

              size: integer, number of data points. Optional: derived from
                    len(times); a ValueError is raised if it disagrees.
              times: float list, observation times (days)
              vals: list, observation values
        '''
        self.times = kwargs['times']
        self.vals = kwargs['vals']
        self.size = kwargs.get('size', len(self.times))
        if self.size != len(self.times):
            raise ValueError(f"size={self.size} disagrees with len(times)={len(self.times)}; "
                             "omit size to derive it from times.")

        if 'calculated_vrad_list' in kwargs:
            self.calculated_vrad_list = kwargs['calculated_vrad_list']
        else:
            self.calculated_vrad_list = []

        if 'fold_period' in kwargs:
            self.fold_period = kwargs['fold_period']
        else:
            self.fold_period = []
        
        if 'errors' in kwargs:
            self.errors = kwargs['errors']
        else:
            self.errors = []


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
        if is_numeric_scalar(spec_list[0]):
            raise ValueError('RV calculation is available for spectrum time-series only')

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
    def calc_rv_against_two_templates(self, spec_list, template1, template2,
                                      dv=0.1, VelBound=150, dvel=None,
                                      sigma=1.0, nonneg=True, **kwargs):
        '''
        Double-lined counterpart of calc_rv_against_template: a TODCOR fit of
        both components against two resting templates, one per observation.

        Each epoch is fitted independently -- orbital motion changes (v1, v2),
        so epochs must never be pooled (only the orders *within* one epoch are,
        which CCF2d does internally).

        :param: spec_list - a list of Spectrum objects, with the observed data.
        :param: template1, template2 - Template objects for the primary and the
                                       secondary component.
        :param: dv, VelBound, dvel, sigma, nonneg - parameters of the CCF2d
                                       CrossCorrelateSpec routine, see the
                                       documentation therein. Remaining keyword
                                       arguments are passed through -- including
                                       keep_windows, which defaults to False
                                       here (retaining spectrum-sized arrays for
                                       every epoch is what fills memory).

        :return: ccfs - a list of CCF2d objects, one per observation.
        '''
        if is_numeric_scalar(spec_list[0]):
            raise ValueError('RV calculation is available for spectrum time-series only')

        kwargs.setdefault('keep_windows', False)

        ccfs = []
        with tqdm(total=len(spec_list)) as pbar:
            for spec in spec_list:
                ccfs.append(CCF2d().CrossCorrelateSpec(spec_in=spec,
                                                       template1_in=template1,
                                                       template2_in=template2,
                                                       dv=dv,
                                                       VelBound=VelBound,
                                                       dvel=dvel,
                                                       sigma=sigma,
                                                       nonneg=nonneg,
                                                       **kwargs))
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
    def plot_velocities(self, figsize=(7,7), sty='ok'):
        '''
        This function plots the time series.
        '''
        if not is_numeric_scalar(self.vals[0]):
            raise ValueError('plot_velocities is available for scalar values time-series only')

        fig, axs = plt.subplots(2, figsize=figsize)
        if len(self.errors) == 0:
            axs[0].plot(self.times, self.vals,sty)
        else:
            axs[0].errorbar(self.times, self.vals, self.errors, fmt='o')

        axs[0].set_title("Time VS. Value")
        if self.fold_period:
            times_phased = [t % self.fold_period for t in self.times]
            if len(self.errors) == 0:
                axs[1].plot(times_phased, self.vals, sty)
            else:
                axs[1].errorbar(times_phased, self.vals, self.errors, fmt='o')
            axs[1].set_title("Phase Folded Values")
        else:
            axs[1] = None
        plt.tight_layout()

        plt.show()

    # =============================================================================
    # =============================================================================
    def build_tiravel_template(self, spec_list, velocities, **kwargs):
        '''
        Build a template out of the observations themselves, at given velocities.

        The template-independent counterpart of calc_rv_against_template: rather
        than correlating against a library template, TIRAVEL shifts every epoch
        to rest, measures how well the set aligns, and co-adds the epochs with
        maximum-likelihood weights. The returned template is then usable by the
        UNICOR (CCF1d) or TODCOR (CCF2d) engines.

        :param: spec_list  - a list of Spectrum objects, with the observed data.
        :param: velocities - a list of radial velocities (km/s), one per epoch.
                             Only their differences matter -- see the TIRAVEL
                             class documentation on the zero-point degeneracy.
        :param: kwargs     - passed through to TIRAVEL.BuildTemplate (orders,
                             reference, margin, kind, keep_aligned).

        :return: a TIRAVEL object, carrying .template (a Template), .rho (the
                 alignment measure), .weights, .correlation and .diagnostics.
        '''
        if is_numeric_scalar(spec_list[0]):
            raise ValueError('TIRAVEL is available for spectrum time-series only')

        return TIRAVEL().BuildTemplate(spec_list, velocities, **kwargs)
