#                   ------------------------------------------
#                     Observations.py (SPARTA Auxil class)
#                   ------------------------------------------
# This file defines the "Observations" class: an astronomical-observation time
# series, built either by reading an entire observation directory or from an
# existing TimeSeries object. It also drives GLS / Zucker PDC / USuRPER on the
# data (arxiv.org/pdf/1711.06075.pdf) via a PeriodicityDetector.
#
# Per-instrument FITS reading lives in sparta.instruments (one module per
# instrument; the survey key selects the loader). Conventions: wavelengths in
# Angstrom, velocities in km/s, times in BJD (days), periodogram frequency in 1/day.
#
# Observations class stores the following methods:
# ---------------------------------------------
# 1) PreProccessSpectra - contains different pre-processing procedures applied on an observed spectrum.
# 2) SelectOrders - enables the user to keep or discard orders from the observed data.
# 3) calc_rv_against_template - calculates radial velocity for the entire spectrum time series,
#                               using CCF against resting template.
# 3b) calc_rv_against_two_templates - the double-lined (SB2) counterpart: TODCOR
#                               against two templates, per epoch.
# 3c) build_tiravel_template - the template-independent counterpart: TIRAVEL
#                               builds a template out of the observations.
# 4) retrieve_BIS - computes BIS from the calculated CCF
# 5) initialize_periodicity_detector -
#                    initiate the object's PeriodicityDetector instance before using it to create periodograms.
# 6) convert_times_to_relative_float_values -
#                            converts the observation dates to float values, marking the days past the first visit.
# 7) clean_time_series - enables systematically removing observation data.
#
#
# Dependencies: os, contextlib, tqdm and numpy.
# Last update: Sahar Shahaf, 20220327.

import os
import contextlib
import warnings
import numpy as np
from sparta.UNICOR.Spectrum import Spectrum
from sparta import instruments
from unicor import dsp
from sparta.Auxil.TimeSeries import TimeSeries
from sparta.Auxil.PeriodicityDetector import PeriodicityDetector
from tqdm import tqdm

class Observations:
    # =============================================================================
    # =============================================================================
    def __init__(self, survey=None, min_snr=-1, target_visits_lib=None, time_series=[],
                 min_avg_flux_val=-1, sample_rate=1, output=False, min_wv=5150, max_wv=5300,
                 simulation=None):
        '''
        Input: All input is optional, and needs to be called along
               with its keyword. Below appears a list of the possible input
               variables.

              survey: str, registry key of the data product (see
                      sparta.instruments.REGISTRY); required with target_visits_lib
                      or simulation.
              min_snr: float, can be used to sort out noisy observations
              target_visits_lib: str, observation visit directory path
                                 (required when reading files).
              time_series: TimeSeries object, in case a direct load of TimeSeries (instead of file reading)
              min_wv, max_wv: float, wavelength window (Angstrom) passed to
                              single-order loaders.
              simulation: sparta.instruments.Simulation, with survey='SIMULATED':
                          simulated spectra instead of files, ingested the same
                          way; what was injected is kept as ``self.truth``
                          (sparta.instruments.Truth), which is None otherwise.

        When reading files, ``time_system`` and ``wavelength_frame`` record
        what the loader returned (e.g. 'BJD-TDB', 'air'); a directory whose
        visits disagree on either raises ValueError. Both are None for a
        directly supplied TimeSeries.
        '''
        self.truth = None
        if not isinstance(time_series, TimeSeries):
            if survey is None:
                raise ValueError("survey is required to read a visit directory; known surveys: "
                                 f"{', '.join(sorted(instruments.REGISTRY))}")
            product = instruments.get(survey)
            if product.synthetic:
                if simulation is None or target_visits_lib:
                    raise ValueError(f"survey {survey!r} is simulated: pass simulation="
                                     "Simulation(...), not target_visits_lib.")
                self.file_list = []
                records, self.truth = product.load(simulation)
            else:
                if simulation is not None:
                    raise ValueError(f"simulation= needs a simulated survey ('SIMULATED'), "
                                     f"not {survey!r}.")
                if not target_visits_lib:
                    raise ValueError(
                        "target_visits_lib is required: pass the path to the "
                        f"{survey} visit directory.")
                records = self._read_directory(product, target_visits_lib, sample_rate,
                                               output, min_wv, max_wv)
            self._ingest(records, min_snr=min_snr, min_avg_flux_val=min_avg_flux_val)
        else:
            self.time_series = time_series
            self.time_system = None
            self.wavelength_frame = None
            self.spec_list = time_series.vals
            self.time_list = time_series.times

            self.sample_size = self.time_series.size

        self.periodicity_detector = []

# =============================================================================
# =============================================================================
    def _read_directory(self, product, target_visits_lib, sample_rate, output, min_wv, max_wv):
        '''
        Load every ``sample_rate``-th FITS file under ``target_visits_lib``
        with ``product.load``; the paths are kept in ``self.file_list``.

        :return: list of sparta.instruments.LoadedSpectrum.
        '''
        self.file_list = []
        count = 0
        for root, dirs, files in os.walk(target_visits_lib):
            for file in files:
                if (file.endswith(".fits") or file.endswith(".fits.gz") or file.endswith(".fts.gz") or file.endswith(".fits.Z")) and not file.startswith('~'):
                    if count % sample_rate == 0:
                        self.file_list.append(os.path.join(root, file))
                    count = count + 1

        records = []
        for visit_path in self.file_list:
            if output:
                print(visit_path)
            records.append(product.load(visit_path, min_wv=min_wv, max_wv=max_wv))
        return records

# =============================================================================
# =============================================================================
    def _ingest(self, records, min_snr=-1, min_avg_flux_val=-1):
        '''
        Build the time series from loaded spectra.

        Keeps the records that pass the SNR and mean-flux cuts, applies each
        one's barycentric correction, and fills spec_list, time_list (days
        after ``first_time``), bcv (km/s), snr, air_mass, vrad_list and
        time_series.

        A record of a resampled product carries its native (detector) step,
        ``native_step_kms``; it becomes the spectrum's ``Info['NativeStep']``
        (km/s), so UNICOR / TODCOR count native pixels rather than archive
        ones. Per order it is ``max(native, archive step)`` -- the rule of
        ``Spectrum.InterpolateSpectrum``, since an archive grid coarser than
        the detector (e.g. HARPS's 0.01 A bins blueward of ~4900 A) leaves
        fewer independent pixels than the detector had. This is approximate:
        the pipeline's own rebinning kernel is unknown, so the noise is
        modelled as white on the native pixels and resampled once, from
        uniformly spread sub-pixel phases (see CCF2d.CrossCorrelateSpec).

        :param records: iterable of sparta.instruments.LoadedSpectrum.
        :param min_snr: float, drop records with snr < min_snr; -1 disables
                        the cut, and a record of unknown (NaN) snr is kept.
        :param min_avg_flux_val: float, drop records whose mean flux is below
                                 it; -1 disables the cut.
        :raises ValueError: if the records (including dropped ones) disagree on
                            (time_system, wavelength_frame).
        '''
        self.spec_list = []
        self.time_list = []
        self.first_time = []
        self.vrad_list = []
        self.bcv = []
        self.snr = []
        self.air_mass = []

        frames = set()
        for rec in records:
            frames.add((rec.time_system, rec.wavelength_frame))
            snr_ok = min_snr == -1 or np.isnan(rec.snr) or rec.snr >= min_snr
            flux_ok = min_avg_flux_val == -1 or min_avg_flux_val <= np.mean(rec.flux)
            if not (snr_ok and flux_ok):
                continue
            sp = Spectrum(wv=rec.wave, sp=rec.flux, bjd=rec.time, bcv=rec.bcv, name=rec.survey)
            if rec.native_step_kms is not None:
                if len(rec.native_step_kms) != len(rec.wave):
                    raise ValueError(f"{rec.path or rec.survey}: native_step_kms has "
                                     f"{len(rec.native_step_kms)} values for "
                                     f"{len(rec.wave)} orders")
                sp.Info['NativeStep'] = [max(float(native), dsp.dv_from_wave(w))
                                         for native, w in zip(rec.native_step_kms, rec.wave)]
            sp.BarycentricCorrection()

            self.spec_list.append(sp)
            self.time_list.append(rec.time)
            self.bcv.append(rec.bcv)
            self.snr.append(rec.snr)
            self.air_mass.append(rec.air_mass)
            if rec.vrad is not None:
                self.vrad_list.append(rec.vrad)

        self.sample_size = len(self.bcv)

        # Relative times and a common CCF grid are meaningless across
        # time systems or air/vacuum frames, so refuse a mixed directory.
        if len(frames) > 1:
            raise ValueError("visits disagree on (time_system, wavelength_frame): "
                             f"{sorted(frames, key=str)}")
        self.time_system, self.wavelength_frame = frames.pop() if frames else (None, None)

        self.convert_times_to_relative_float_values()

        self.time_series = TimeSeries(size=self.sample_size, times=self.time_list,
                                      vals=self.spec_list)

# =============================================================================
# =============================================================================
    def PreProccessSpectra(self, Ntrim=10, CleanMargins=True, RemoveNaNs=True,
                           delta=0.5, RemCosmicNum=3, FilterLC=3, FilterHC=0.15,
                           alpha=0.3, verbose=True):
        '''
        This function conatains different pre-processing procedures applied on an observed spectrum.
        '''
        # Optionally silence per-spectrum stdout via a context manager, so the
        # real stdout is always restored (even on error) and no file handle
        # leaks -- unlike the previous global sys.stdout reassignment.
        with open(os.devnull, 'w') as devnull:
            silence = contextlib.redirect_stdout(devnull) if not verbose \
                else contextlib.nullcontext()
            with silence, tqdm(total=self.sample_size) as pbar:
                for s in self.spec_list:
                    s.SpecPreProccess(Ntrim=Ntrim, CleanMargins=CleanMargins, RemoveNaNs=RemoveNaNs,
                                      delta=delta, RemCosmicNum=RemCosmicNum, FilterLC=FilterLC,
                                      FilterHC=FilterHC, alpha=alpha)
                    pbar.update(1)

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
    def calc_rv_against_two_templates(self, template1, template2, dv=0.1,
                                      VelBound=150, dvel=None, sigma=None,
                                      nonneg=True, **kwargs):
        '''
        Double-lined (SB2) radial velocities for the whole time series, using
        TODCOR against two resting templates.

        The single-lined counterpart is calc_rv_against_template. Each epoch is
        fitted independently; the orders within an epoch are pooled into one
        joint (v1, v2) fit.

        :param: template1, template2 - Template objects for the primary and the
                                       secondary component.
        :param: dv, VelBound, dvel, sigma, nonneg - parameters for the CCF2d
                                       CrossCorrelateSpec routine, see the
                                       documentation therein.

        :return: self, with 'vels1'/'vels2' (km/s), their uncertainties
                 'evels1'/'evels2' (km/s, Cramer-Rao), the amplitude pairs
                 'amps1'/'amps2', and the list of CCF2d objects 'ccf2d_list'.
        '''
        ccfs = (self.time_series.
                calc_rv_against_two_templates(self.spec_list,
                                              template1, template2,
                                              dv=dv,
                                              VelBound=VelBound,
                                              dvel=dvel,
                                              sigma=sigma,
                                              nonneg=nonneg,
                                              **kwargs)
                )

        fields = ('RV1', 'RV2', 'eRV1', 'eRV2', 'a1_best', 'a2_best')
        vels1, vels2, evels1, evels2, amps1, amps2 = (
            np.array([c.Corr2d[f] for c in ccfs], dtype=float) for f in fields)

        self.ccf2d_list = ccfs
        self.vels1, self.vels2 = vels1, vels2
        self.evels1, self.evels2 = evels1, evels2
        self.amps1, self.amps2 = amps1, amps2

        return self

# =============================================================================
# =============================================================================
    def build_tiravel_template(self, velocities, **kwargs):
        '''
        Build a template out of the observations themselves (TIRAVEL), at a
        given set of trial velocities.

        The template-independent counterpart of calc_rv_against_template: no
        library template is needed. Typical use is to bootstrap -- measure
        ball-park velocities against any rough template, build the TIRAVEL
        template from them, then re-measure against it.

        :param: velocities - a list of radial velocities (km/s), one per epoch.
                             Only their differences matter: the template's rest
                             frame inherits this set's zero point.
        :param: kwargs     - passed through to TIRAVEL.BuildTemplate.

        :return: self, with 'tiravel' (the TIRAVEL object) and 'tiravel_template'
                 (its Template, ready for calc_rv_against_template).
        '''
        self.tiravel = self.time_series.build_tiravel_template(
            self.spec_list, velocities, **kwargs)
        self.tiravel_template = self.tiravel.template

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
    def clean_time_series(self, max_vel=None, min_vel=None, max_time=None, nan_flag=True, sample_rate=1):
        """
        Systematically remove observations by sampling rate, RV value or time.

        One boolean epoch mask is built from all the criteria and applied to
        every per-epoch array the object holds (the TimeSeries times, values,
        RVs and errors; time_list, spec_list, vrad_list, bcv, snr, air_mass;
        and any computed RV products such as vels/evels/ccf_list).

        :param max_vel: float or None, keep epochs with RV <= max_vel (km/s).
        :param min_vel: float or None, keep epochs with RV >= min_vel (km/s).
        :param max_time: float or None, keep epochs with time <= max_time (days).
        :param nan_flag: bool, drop epochs whose RV is NaN. Ignored when no RVs
                         are stored in time_series.calculated_vrad_list.
        :param sample_rate: nonzero int; k > 0 keeps every k-th epoch, k < 0
                            drops every |k|-th epoch.
        :raises ValueError: for sample_rate=0, for an RV cut with no stored RVs,
                            or when a per-epoch array has the wrong length.
        """
        ts = self.time_series
        n = len(ts.times)
        index = np.arange(n)

        if sample_rate == 0:
            raise ValueError("sample_rate must be a nonzero integer.")
        keep = (index % sample_rate == 0) if sample_rate > 0 else (index % abs(sample_rate) != 0)

        vrads = np.asarray(ts.calculated_vrad_list, float)
        has_rvs = vrads.size > 0
        if (max_vel is not None or min_vel is not None) and not has_rvs:
            raise ValueError("A velocity cut needs RVs in time_series.calculated_vrad_list, which is empty.")
        velocity_cut = max_vel is not None or min_vel is not None
        if has_rvs and vrads.shape != (n,):
            # E.g. a [min, max] pair of RV rows: not one RV per epoch, so it
            # cannot drive a cut -- but it should not block a time-only clean.
            if velocity_cut:
                raise ValueError(f"time_series.calculated_vrad_list has shape {vrads.shape}; "
                                 f"a velocity cut needs one RV per epoch ({n},).")
            if nan_flag:
                warnings.warn("calculated_vrad_list is not one RV per epoch; "
                              "skipping the NaN filter.", stacklevel=2)
        elif has_rvs and (nan_flag or velocity_cut):
            if nan_flag:
                keep &= ~np.isnan(vrads)
            # NaN RVs fail both comparisons, so they drop out of a cut regardless.
            if max_vel is not None:
                keep &= vrads <= max_vel
            if min_vel is not None:
                keep &= vrads >= min_vel

        if max_time is not None:
            keep &= np.asarray(ts.times, float) <= max_time

        ts_names = ('times', 'vals', 'errors', 'calculated_ccf_peaks')
        if not has_rvs or vrads.shape == (n,):
            ts_names += ('calculated_vrad_list',)
        for owner, names in ((ts, ts_names),
                             # vrad_list is left out: it only gets entries for
                             # headers that carry VRAD, so it is not epoch-aligned.
                             (self, ('bcv', 'snr', 'air_mass',
                                     'ccf_list', 'vels', 'evels', 'ccf_peaks',
                                     'ccf2d_list', 'vels1', 'vels2', 'evels1', 'evels2',
                                     'amps1', 'amps2', 'BIS', 'eBIS'))):
            for name in names:
                seq = getattr(owner, name, None)
                if seq is None or len(seq) == 0:
                    continue
                if len(seq) != n:
                    raise ValueError(f"{name} has {len(seq)} entries but the time series has "
                                     f"{n} epochs; cannot filter it consistently.")
                if isinstance(seq, np.ndarray):
                    setattr(owner, name, seq[keep])
                else:
                    setattr(owner, name, [x for x, k in zip(seq, keep) if k])

        self.sample_size = len(ts.times)
        ts.size = self.sample_size
        self.spec_list = ts.vals
        self.time_list = ts.times
