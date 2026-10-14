"""Observations: a spectroscopic time series and the analyses run on it.

An ``Observations`` object is built from a directory of archive files (read by
the per-instrument loaders in ``sparta.instruments``, selected by the survey
key), from a ``sparta.instruments.Simulation``, or from an existing
``TimeSeries``. It drives the RV engines -- UNICOR (``CCF1d``), TODCOR
(``CCF2d``) and TIRAVEL -- epoch by epoch, and the periodograms (GLS, PDC,
USuRPER) through a ``PeriodicityDetector``. It is the only series-level
driver: ``TimeSeries`` is a plain container.

A single-lined RV run with combined CCFs (``calc_rv_against_template``)
stores the RVs and their uncertainties in ``time_series.calculated_vrad_list``
and ``time_series.errors``, so the periodograms use them directly. A
double-lined run (``calc_rv_against_two_templates``) keeps the two components
on this object (``vels1``, ``vels2``) and leaves the time series alone.

Conventions: wavelengths in Angstrom, velocities in km/s, times in days (for
files read from disk or simulated, ``time_list`` counts days from the integer
day ``first_time``), periodogram frequencies in 1/day. RVs are in the 'cz'
Doppler convention unless ``velocity_convention='relativistic'`` is passed to
the RV drivers.

References: UNICOR -- Zucker 2003, MNRAS 342, 1291; TODCOR -- Zucker & Mazeh
1994, ApJ 420, 806; TIRAVEL -- Zucker & Mazeh 2006, MNRAS 371, 1513; PDC --
Zucker 2018, MNRAS 474, L86; USuRPER -- Binnenfeld, Shahaf & Zucker 2020,
A&A 642, A146.
"""
import os
import warnings
import numpy as np
from sparta.spectrum import Spectrum
from sparta import instruments
from unicor import dsp
from sparta.timeseries import TimeSeries, is_numeric_scalar
from sparta.periodogram import PeriodicityDetector
from sparta.ccf1d import CCF1d
from sparta.ccf2d import CCF2d
from sparta.tiravel import TIRAVEL

class Observations:
    '''
    A spectroscopic (or scalar) time series and the analyses run on it.

    Attributes
    ----------
    spec_list : list of Spectrum
        One spectrum per epoch (the values of a supplied TimeSeries).
    time_list : list of float
        Epoch times (days); for read or simulated data, days after
        ``first_time``.
    first_time : int
        The integer day subtracted from the input times (read or simulated
        data only).
    time_series : TimeSeries
        The container the periodograms run on.
    truth : sparta.instruments.Truth or None
        What a simulation injected; None for real data.
    time_system, wavelength_frame : str or None
        As the loader reported them (e.g. 'BJD-TDB', 'air'); None for a
        supplied TimeSeries.
    bcv, snr, air_mass : list of float
        Per-epoch barycentric correction (km/s; already applied to the
        wavelengths), S/N and airmass (NaN where unknown). Read or simulated
        data only.
    vrad_list : list of float
        Archive RVs (km/s) of the records whose header carries one; not
        epoch-aligned.
    periodicity_detector : PeriodicityDetector
        Set by ``init_periodicity_detector``.

    The RV methods add their results as attributes; see each method.
    '''
    # =============================================================================
    # =============================================================================
    def __init__(self, survey=None, min_snr=-1, target_visits_lib=None, time_series=None,
                 min_avg_flux_val=-1, sample_rate=1, output=False, min_wv=5150, max_wv=5300,
                 simulation=None):
        '''
        Build the time series from a visit directory, a simulation, or a TimeSeries.

        Exactly one source is used: ``time_series`` if given; else
        ``simulation`` (with ``survey='SIMULATED'``); else the FITS files
        under ``target_visits_lib`` (with the product's ``survey`` key).
        Read and simulated spectra get their barycentric correction applied,
        and their times become days after the integer day of the first visit.

        Parameters
        ----------
        survey : str, optional
            Registry key of the data product (``sparta.instruments.REGISTRY``,
            docs/instruments.md); required unless ``time_series`` is given.
        min_snr : float
            Drop records with S/N below it; -1 (default) keeps all, and a
            record of unknown S/N is always kept.
        target_visits_lib : str, optional
            Directory searched (recursively) for .fits, .fits.gz, .fts.gz and
            .fits.Z files.
        time_series : TimeSeries, optional
            Use this series directly instead of reading files.
        min_avg_flux_val : float
            Drop records whose mean flux is below it; -1 (default) keeps all.
        sample_rate : int
            Read only every ``sample_rate``-th file found.
        output : bool
            Print each file path as it is read.
        min_wv, max_wv : float
            Wavelength window (Angstrom) passed to the loaders: single-order
            products are trimmed to it (or pick the order covering it).
        simulation : sparta.instruments.Simulation, optional
            Simulated spectra, ingested like files; what was injected is kept
            as ``self.truth``.

        Raises
        ------
        ValueError
            For a missing or unknown ``survey``, a missing
            ``target_visits_lib``, a simulation with a non-simulated survey
            (or vice versa), or visits that disagree on time system or
            wavelength frame.
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

        Parameters
        ----------
        product : sparta.instruments.Product
        target_visits_lib, sample_rate, output, min_wv, max_wv
            See ``__init__``.

        Returns
        -------
        list of sparta.instruments.LoadedSpectrum
        '''
        self.file_list = []
        count = 0
        # Sorted walk: os.walk order is filesystem-dependent, and epoch order
        # (and sample_rate's selection) must not depend on the machine.
        for root, dirs, files in os.walk(target_visits_lib):
            dirs.sort()
            for file in sorted(files):
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
        ``Spectrum.interpolate``, since an archive grid coarser than
        the detector (e.g. HARPS's 0.01 A bins blueward of ~4900 A) leaves
        fewer independent pixels than the detector had. This is approximate:
        the pipeline's own rebinning kernel is unknown, so the noise is
        modelled as white on the native pixels and resampled once, from
        uniformly spread sub-pixel phases (see CCF2d.cross_correlate).

        Parameters
        ----------
        records : iterable of sparta.instruments.LoadedSpectrum
        min_snr : float
            Drop records with snr < min_snr; -1 disables the cut, and a record
            of unknown (NaN) snr is kept.
        min_avg_flux_val : float
            Drop records whose mean flux is below it; -1 disables the cut.

        Raises
        ------
        ValueError
            If the records (including dropped ones) disagree on
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
            sp.barycentric_correction()

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

        self._relative_times()

        self.time_series = TimeSeries(size=self.sample_size, times=self.time_list,
                                      vals=self.spec_list)

# =============================================================================
# =============================================================================
    def preprocess(self, n_trim=10, clean_margins=True, remove_nans=True,
                   delta=0.5, cosmic_sigma=3, filter_lowcut=3, filter_highcut=0.15,
                   alpha=0.3, progress=False):
        '''
        Apply ``Spectrum.preprocess`` to every epoch, in place.

        Parameters
        ----------
        n_trim, clean_margins, remove_nans, delta, cosmic_sigma, filter_lowcut, filter_highcut, alpha
            Passed through; see ``Spectrum.preprocess``. The per-instrument
            recommendations are in docs/instruments.md.
        progress : bool
            Show a tqdm progress bar over the epochs. Off by default; needs the
            optional ``tqdm`` package (``pip install "SPARTA[progress]"``).

        Returns
        -------
        Observations
            self.
        '''
        for s in _progress(self.spec_list, progress):
            s.preprocess(n_trim=n_trim, clean_margins=clean_margins, remove_nans=remove_nans,
                         delta=delta, cosmic_sigma=cosmic_sigma, filter_lowcut=filter_lowcut,
                         filter_highcut=filter_highcut, alpha=alpha)

        return self

# =============================================================================
# =============================================================================
    def select_orders(self, orders, remove=True):
        '''
        Keep or discard the same orders in every epoch (``Spectrum.select_orders``).

        Parameters
        ----------
        orders : sequence of int
            Order indices, 0 .. n_ord - 1 of each epoch's spectrum.
        remove : bool
            If True, drop these orders; if False, keep only these.

        Returns
        -------
        Observations
            self.

        Raises
        ------
        ValueError
            If an index is out of range (the epochs before the offending one
            have already been changed).
        '''
        for s in self.spec_list:
            s.select_orders(orders, remove=remove)
        return self

# =============================================================================
# =============================================================================
    def _spectra(self, engine):
        '''
        The spectra the RV engines run on.

        Parameters
        ----------
        engine : str
            The engine's name, for the error message.

        Returns
        -------
        list of Spectrum
            One per epoch.

        Raises
        ------
        ValueError
            For a scalar (e.g. RV) time series.
        '''
        if is_numeric_scalar(self.spec_list[0]):
            raise ValueError(f'{engine} is available for spectrum time-series only')
        return self.spec_list

# =============================================================================
# =============================================================================
    def calc_rv_against_template(self, template, dv=0.1, vel_bound=150,
                                 errors_per_order=False, combine_ccfs=True,
                                 velocity_convention='cz', progress=False):
        '''
        Single-lined (SB1) radial velocities of every epoch, by UNICOR.

        Each epoch is cross-correlated with one template at rest
        (``CCF1d.cross_correlate``); the spectra are not modified.

        Parameters
        ----------
        template : Template
            The template, at rest, in the observations' wavelength frame.
        dv : float
            Log-lambda grid step (km/s) for epochs not already on one.
        vel_bound : float or (float, float)
            CCF scan half-width (km/s).
        errors_per_order : bool
            Also compute each order's ML uncertainty.
        combine_ccfs : bool
            If True, combine every epoch's per-order CCFs by maximum
            likelihood (Zucker 2003) and read the RV and its uncertainty from
            the combined CCF. If False, make only the per-order measurements.
        velocity_convention : {'cz', 'relativistic'}
            Doppler convention of the RVs (see ``CCF1d.cross_correlate``).
        progress : bool
            Show a tqdm progress bar over the epochs. Off by default; needs the
            optional ``tqdm`` package (``pip install "SPARTA[progress]"``).

        Returns
        -------
        Observations
            self, with these attributes set:

            ``ccf_list``
                The CCF1d objects, one per epoch.
            ``vels_per_order``, ``evels_per_order``, ``ccf_peaks_per_order``
                (epochs x orders) arrays: per-order RVs (km/s), their
                uncertainties (km/s; NaN unless ``errors_per_order``) and CCF
                peaks.
            ``vels``, ``evels``, ``ccf_peaks``
                Per-epoch RV (km/s), its ML uncertainty (km/s) and the CCF
                peak, from the combined CCF. None when ``combine_ccfs`` is
                False: an average of the per-order RVs is left to the caller,
                as their scatter is not an uncertainty.

            With ``combine_ccfs``, ``time_series.calculated_vrad_list`` and
            ``time_series.errors`` are set to copies of ``vels`` and
            ``evels``, so the periodograms use the measured RVs; without,
            they are emptied rather than left holding an earlier run's RVs.
        '''
        spectra = self._spectra('RV calculation')

        ccfs = []
        for spec in _progress(spectra, progress):
            ccf = CCF1d().cross_correlate(spec, template,
                                          dv=dv,
                                          vel_bound=vel_bound,
                                          errors_per_order=errors_per_order,
                                          velocity_convention=velocity_convention)
            if combine_ccfs:
                ccf.combine()
            ccfs.append(ccf)

        self.ccf_list = ccfs
        self.vels_per_order, self.evels_per_order, self.ccf_peaks_per_order = (
            np.array([c.Corr[f][:, 0] for c in ccfs], dtype=float)
            for f in ('RV', 'eRV', 'peakCorr'))

        if combine_ccfs:
            self.vels, self.evels, self.ccf_peaks = (
                np.array([c.CorrCombined[f] for c in ccfs], dtype=float)
                for f in ('RV', 'eRV', 'peakCorr'))
            self.time_series.calculated_vrad_list = self.vels.copy()
            self.time_series.errors = self.evels.copy()
        else:
            self.vels = self.evels = self.ccf_peaks = None
            self.time_series.calculated_vrad_list = []
            self.time_series.errors = []

        return self

# =============================================================================
# =============================================================================
    def calc_rv_against_two_templates(self, template1, template2, dv=0.1,
                                      vel_bound=150, dvel=None, sigma=None,
                                      nonneg=True, velocity_convention='cz', progress=False,
                                      **kwargs):
        '''
        Double-lined (SB2) radial velocities of every epoch, by TODCOR.

        The single-lined counterpart is ``calc_rv_against_template``. Each
        epoch is fitted independently -- orbital motion changes (v1, v2), so
        epochs are never pooled; the orders within an epoch are pooled into
        one joint (v1, v2) fit by ``CCF2d.cross_correlate``.

        The two components' RVs stay on this object:
        ``time_series.calculated_vrad_list`` (one RV per epoch, read by the
        periodograms) is left untouched. To run a periodogram on one
        component, assign it there explicitly, e.g.
        ``obs.time_series.calculated_vrad_list = obs.vels1``.

        Parameters
        ----------
        template1, template2 : Template
            The primary's and the secondary's templates, at rest.
        dv : float
            Log-lambda grid step (km/s) for epochs not already on one.
        vel_bound : float or (float, float)
            Trial-velocity range (km/s) of both components.
        dvel : float, optional
            Step of the coarse trial-velocity map (km/s); default 1 km/s.
        sigma : float or array_like, optional
            Per-native-pixel noise (flux units), scalar or per order; None
            (default) estimates each order's noise from its residual.
        nonneg : bool
            Constrain both amplitudes to be non-negative.
        velocity_convention : {'cz', 'relativistic'}
            Doppler convention of the RVs (see ``CCF2d.cross_correlate``).
        progress : bool
            Show a tqdm progress bar over the epochs. Off by default; needs the
            optional ``tqdm`` package (``pip install "SPARTA[progress]"``).
        **kwargs
            Passed to ``CCF2d.cross_correlate`` (e.g. ``alpha``,
            ``zeta_max``, ``orders``); ``keep_windows`` defaults to False
            here, as keeping spectrum-sized arrays for every epoch fills
            memory.

        Returns
        -------
        Observations
            self, with ``vels1`` / ``vels2`` (km/s), their Cramer-Rao
            uncertainties ``evels1`` / ``evels2`` (km/s), the amplitudes
            ``amps1`` / ``amps2`` and the CCF2d objects ``ccf2d_list``.
        '''
        spectra = self._spectra('RV calculation')
        kwargs.setdefault('keep_windows', False)

        ccfs = [CCF2d().cross_correlate(spec, template1, template2,
                                        dv=dv,
                                        vel_bound=vel_bound,
                                        dvel=dvel,
                                        sigma=sigma,
                                        nonneg=nonneg,
                                        velocity_convention=velocity_convention,
                                        **kwargs)
                for spec in _progress(spectra, progress)]

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
        Build a template out of the observations themselves (TIRAVEL).

        The template-independent counterpart of ``calc_rv_against_template``:
        no library template is needed. TIRAVEL shifts every epoch to rest at
        the given trial velocities, measures how well the set aligns, and
        co-adds the epochs with maximum-likelihood weights. Typical use is to
        bootstrap -- measure ball-park velocities against any rough template,
        build the TIRAVEL template from them, then re-measure against it.

        Parameters
        ----------
        velocities : array_like
            One radial velocity per epoch (km/s). Only their differences
            matter: the template's rest frame inherits this set's zero point.
        **kwargs
            Passed to ``TIRAVEL.build_template`` (orders, reference, margin,
            kind, keep_aligned).

        Returns
        -------
        Observations
            self, with ``tiravel`` (the TIRAVEL object, carrying ``rho``,
            ``weights``, ``correlation`` and ``diagnostics``) and
            ``tiravel_template`` (its Template, ready for
            ``calc_rv_against_template``).
        '''
        spectra = self._spectra('TIRAVEL')
        self.tiravel = TIRAVEL().build_template(spectra, velocities, **kwargs)
        self.tiravel_template = self.tiravel.template

        return self

# =============================================================================
# =============================================================================
    def calc_bis(self, bisect_val=(0.35, 0.95), use_combined=True):
        '''
        Bisector span (BIS) of every epoch's CCF, from ``ccf_list``.

        .. warning:: The BIS formula (``unicor.bisector_span``) is known to be
           incorrect and is under review: it differences CCF values at the
           bisector levels, not velocities. It is pinned, unchanged, for a
           later fix; do not use its output scientifically. The
           ``use_combined=False`` path is broken too (it raises
           AttributeError).

        Parameters
        ----------
        bisect_val : (float, float)
            Lower and upper bisector levels, as fractions of the CCF maximum.
        use_combined : bool
            Use each epoch's combined CCF (requires
            ``calc_rv_against_template`` with ``combine_ccfs=True``).

        Returns
        -------
        Observations
            self, with ``BIS`` and ``eBIS`` arrays, one value per epoch.
        '''
        ccf_list = self.ccf_list
        BIS = np.full(len(ccf_list), np.nan)
        eBIS = np.full(len(ccf_list), np.nan)

        if use_combined:
            for ind, c in enumerate(ccf_list):
                BIS[ind], eBIS[ind], _ = c.calc_bis(c.CorrCombined['vel'],
                                                    c.CorrCombined['corr'],
                                                    bisect_val=bisect_val,
                                                    n_ord=c.n_ord)
        else:
            for ind, c in enumerate(ccf_list):
                BISo = np.full(c.n_ord, np.nan)
                for co in c.CombinedCorr['corr']:
                    BISo[ind], _ = c.calc_bis(c.Corr['vel'],
                                              co,
                                              bisect_val=bisect_val,
                                              n_ord=1)
                BIS[ind] = np.average(BISo)
                eBIS[ind] = np.std(BISo)

        self.BIS = BIS
        self.eBIS = eBIS
        return self

# =============================================================================
# =============================================================================
    def init_periodicity_detector(self, freq_range=(1/1000, 1), periodogram_grid_resolution=1000):
        """
        Create ``self.periodicity_detector`` on this object's time series.

        Parameters
        ----------
        freq_range : (float, float)
            (f_min, f_max) of the frequency grid (1/day); f_min > 0.
        periodogram_grid_resolution : float
            Grid points per 1/day (the grid step is its inverse).

        Returns
        -------
        Observations
            self.
        """
        self.periodicity_detector = PeriodicityDetector(time_series=self.time_series, freq_range=freq_range,
                                                        periodogram_grid_resolution=periodogram_grid_resolution)
        return self

# =============================================================================
# =============================================================================
    def _relative_times(self):
        """
        Make the times relative: subtract ``int(min(time_list))``.

        ``first_time`` keeps the subtracted integer day (truncated toward
        zero), so ``time_list`` holds days past it, starting within one day
        of 0. The periodograms only use time differences.
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

        Parameters
        ----------
        max_vel : float, optional
            Keep epochs with RV <= max_vel (km/s).
        min_vel : float, optional
            Keep epochs with RV >= min_vel (km/s).
        max_time : float, optional
            Keep epochs with time <= max_time (days, as in ``time_list``).
        nan_flag : bool
            Drop epochs whose RV is NaN. Ignored when no RVs are stored in
            ``time_series.calculated_vrad_list``.
        sample_rate : int
            Nonzero; k > 0 keeps every k-th epoch, k < 0 drops every |k|-th
            epoch.

        Returns
        -------
        None
            The object is filtered in place.

        Raises
        ------
        ValueError
            For sample_rate=0, for an RV cut with no stored RVs, or when a
            per-epoch array has the wrong length.
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

        ts_names = ('times', 'vals', 'errors')
        if not has_rvs or vrads.shape == (n,):
            ts_names += ('calculated_vrad_list',)
        for owner, names in ((ts, ts_names),
                             # vrad_list is left out: it only gets entries for
                             # headers that carry VRAD, so it is not epoch-aligned.
                             (self, ('bcv', 'snr', 'air_mass',
                                     'ccf_list', 'vels', 'evels', 'ccf_peaks',
                                     'vels_per_order', 'evels_per_order',
                                     'ccf_peaks_per_order',
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


def _progress(iterable, enabled):
    '''
    Wrap an iterable in a tqdm progress bar if asked; otherwise return it as is.

    tqdm is an optional dependency, imported only when a bar is requested.

    Parameters
    ----------
    iterable : iterable
        The epochs to loop over.
    enabled : bool
        Whether to show the progress bar.

    Returns
    -------
    iterable
        ``iterable`` itself, or a tqdm wrapper around it.

    Raises
    ------
    ImportError
        If ``enabled`` and tqdm is not installed.
    '''
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError as err:
        raise ImportError('progress=True needs the optional tqdm package: '
                          'pip install "SPARTA[progress]"') from err
    return tqdm(iterable)
