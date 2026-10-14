#                   -----------------------------------
#                     ReadSpec.py (SPARTA Auxil class)
#                   -----------------------------------
# This file defines the "ReadSpec" class. An object of this class reads and stores
# fits data from various types of instruments and stores it
# into a Spectrum class. Currently, the following spectrum of the following
# instruments are supported: HARPS, APOGEE, NRES, ELODIE, LAMOST ...
#
# A ReadSpec class stores the following methods:
# ---------------------------------------------
#
# 1) load_spectrum_from_fits - loads visit FITS file and parses it to fit the UNICOR Spectrum class.
# 2) retrieve_all_spectrum_parameters - returns the visit parameters read.
# 3) APOGEE_masking - removes bad pixels from an APOGEE spectra.
#
# Every loader records the frame of what it returned:
#   time_system      -- what DATE_OBS is, e.g. 'BJD-TDB', 'MJD-UTC', 'JD-UTC'.
#   wavelength_frame -- 'air', 'vacuum', or None when not documented.
#   specsys          -- FITS SPECSYS of the returned wavelengths *before* the
#                       bcv is applied: 'BARYCENT', 'HELIOCEN', 'TOPOCENT',
#                       'SOURCE' (stellar rest frame), or None when unknown.
#
# Dependencies: numpy, astropy, specutils.

import warnings

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from specutils import Spectrum as Spec1D, SpectrumList

from sparta.instruments._common import _ascending, _wavelength_window, _best_order
from sparta.instruments._eso import _eso_sdp_frames, _bcv_for_specsys
from sparta.instruments._iraf import _iraf_multispec_wavelengths


class ReadSpec:
    # =============================================================================
    # =============================================================================
    def __init__(self, survey, **kwargs):
        '''
        Input: All input is optional, and needs to be called along
               with its keyword. Below appears a list of the possible input
               variables.

              s: spectrum vector
              w: wavelength vector
              bool_mask: str, APOGEE bool mask as described here: https://www.sdss.org/dr15/algorithms/bitmasks/#APOGEE_PIXMASK
              DATE-OBS: str, observation date from FITS file
              vrad: float, APOGEE given radial velocity from FITS file
              segment: int. 0, 1 or 2 (choosing 1 APOGEE "color" data band)
        '''

        self.survey = survey

        if all(k in kwargs for k in ('s', 'w', 'bool_mask', 'DATE-OBS', 'vrad', 'segment')):
            self.s = kwargs['s']
            self.w = kwargs['w']
            self.bool_mask = kwargs['bool_mask']
            self.DATE_OBS = kwargs['DATE-OBS']
            self.vrad = kwargs['vrad']
            self.segment = kwargs['segment']
        else:
            self.s = []
            self.w = []
            self.bool_mask = []
            self.vrad = []
            self.segment = []
        self.air_mass = []

        self.snr = -1

        self.bcv = []

        self.n_orders = []

        self.time_system = None
        self.wavelength_frame = None
        self.specsys = None

# =============================================================================
# =============================================================================
    def _load_harps(self, hdul, min_wv, max_wv):
        '''
        HARPS loader -- ESO Phase-3 SDP 1-D spectrum (HDU1 binary table with
        WAVE / FLUX / ERR columns). Reads the spectrum as a specutils
        Spectrum1D (wavelengths in Angstrom, dimensionless flux), trims to
        [min_wv, max_wv], and stores the legacy single-order arrays. The
        SDP s1d product is barycentric (SPECSYS) in air (TUCD1).
        '''
        self.n_orders = 1
        wave = np.asarray(hdul[1].data['WAVE'][0], float)
        flux = np.asarray(hdul[1].data['FLUX'][0], float)
        spec = Spec1D(flux=flux * u.dimensionless_unscaled,
                      spectral_axis=wave * u.Angstrom)

        axis = spec.spectral_axis.to_value(u.Angstrom)
        window = _wavelength_window(axis, min_wv, max_wv)
        self.w = np.reshape(axis[window], (1, -1))
        self.s = np.reshape(spec.flux.value[window], (1, -1))

        self.specsys, self.wavelength_frame = _eso_sdp_frames(hdul)
        self.bcv = _bcv_for_specsys(hdul[0].header, self.specsys, 'HIERARCH ESO DRS BERV')

        self.DATE_OBS = float(hdul[0].header['HIERARCH ESO DRS BJD'])
        # UNVERIFIED: whether the HARPS DRS BJD is on the UTC or TDB scale.
        self.time_system = 'BJD (HARPS DRS; scale unverified)'
        self.snr = hdul[0].header['SNR']
        try:
            self.air_mass = (hdul[0].header['HIERARCH ESO TEL AIRM START']
                             + hdul[0].header['HIERARCH ESO TEL AIRM END']) / 2
        except KeyError:
            self.air_mass = 0

# =============================================================================
# =============================================================================
    def _load_nres(self, hdul, min_wv, max_wv):
        '''
        NRES loader (Las Cumbres, commissioning pipeline) -- 67 echelle
        orders; flux in HDU2 and wavelength (nm) in HDU6. Reads each order as
        a specutils Spectrum1D (nm converted to Angstrom) and stores the
        legacy per-order lists. Wavelengths are topocentric; the BCV is
        recovered from the barycentric-corrected (RVCC) and raw (RCC) RVs.
        '''
        self.n_orders = 67
        orders = SpectrumList([
            Spec1D(flux=np.asarray(hdul[2].data[k], float) * u.dimensionless_unscaled,
                   spectral_axis=(10.0 * np.asarray(hdul[6].data[k], float)) * u.Angstrom)
            for k in range(self.n_orders)])
        self.w = [o.spectral_axis.to_value(u.Angstrom) for o in orders]
        self.s = [o.flux.value for o in orders]

        self.bcv = hdul[0].header['RVCC'] - hdul[0].header['RCC'] * 299792.458
        self.specsys = 'TOPOCENT'
        # UNVERIFIED: air or vacuum for the commissioning (IDL) pipeline;
        # BANZAI-NRES documents vacuum, the older products do not say.
        self.wavelength_frame = None
        self.DATE_OBS = hdul[0].header['BJD']  # header: "[BJD_TDB] Exposure center"
        self.time_system = 'BJD-TDB'

# =============================================================================
# =============================================================================
    def load_spectrum_from_fits(self, path=None, APOGEE_segment=1, min_wv=4900, max_wv=5150, output=False):
        '''
        Load one visit FITS file into the legacy per-order arrays.

        Parameters
        ----------
        path : str
            Visit FITS file path (required).
        APOGEE_segment : int
            APOGEE chip row to read (0, 1 or 2).
        min_wv, max_wv : float
            Wavelength window (Angstrom) for single-order loaders; pixels with
            ``min_wv < w <= max_wv`` are kept. Ignored by the multi-order
            NRES / NRES-BANZAI loaders and by APOGEE.
        output : bool
            Print the file name being read.

        Raises
        ------
        ValueError
            If ``path`` is missing, the survey is unknown, or the window does
            not overlap the spectrum.
        '''
        if not path:
            raise ValueError("path is required: pass the FITS file path to load.")

        self.APOGEE_segment = APOGEE_segment

        if output:
            print(path)

        # One loader method per instrument, dispatched by survey name.
        loaders = {
            "HARPS": self._load_harps, "NRES": self._load_nres, "APOGEE": self._load_apogee,
            "LAMOST": self._load_lamost, "ELODIE": self._load_elodie, "NRES-BANZAI": self._load_nres_banzai,
            "TRES": self._load_tres, "UVES": self._load_uves, "FEROS": self._load_feros,
            "HARPS_15": self._load_harps_15, "FEROS_15": self._load_feros_15, "CORALIE": self._load_coralie,
            "SRG": self._load_srg, "KEK": self._load_kek, "PEPSI": self._load_pepsi,
        }
        if self.survey not in loaders:
            raise ValueError(f"Unsupported survey: {self.survey!r}")
        # memmap=False: the arrays the loaders keep must outlive the file handle.
        with fits.open(path, memmap=False) as hdul:
            loaders[self.survey](hdul, min_wv, max_wv)

    # =============================================================================
    # =============================================================================
    def _load_apogee(self, hdul_sp, min_wv, max_wv):
        '''
        APOGEE apVisit loader (SDSS DR17 data model,
        https://data.sdss.org/datamodel/files/APOGEE_REDUX/APRED_VERS/visit/TELESCOPE/FIELD/PLATE_ID/MJD5/apVisit.html):
        HDU1 flux, HDU3 APOGEE_PIXMASK, HDU4 wavelength, one row per chip.
        Wavelengths are vacuum (https://www.sdss4.org/dr17/irspec/spectra/)
        in the observer frame: the header has VHELIO = VRAD + BC, so the BC
        (km/s) is returned as the bcv. Rows are reordered to ascending
        wavelength. The window is not applied (H band only).
        '''
        self.n_orders = 1
        header = hdul_sp[0].header
        w, s, flags = _ascending(hdul_sp[4].data[self.APOGEE_segment],
                                 np.array(hdul_sp[1].data[self.APOGEE_segment], dtype=float),
                                 np.array(hdul_sp[3].data[self.APOGEE_segment]))

        self.s = np.reshape(s, (1, -1))
        self.w = np.reshape(w, (1, -1))

        self.DATE_OBS = float(header['JD-MID'])  # "JD at midpoint of visit"
        # UNVERIFIED: the time scale of JD-MID (assumed UTC).
        self.time_system = 'JD-UTC'
        self.bcv = float(header['BC'])
        self.specsys = 'TOPOCENT'
        self.wavelength_frame = 'vacuum'

        self.vrad = header['VRAD']

        # The mask HDU is int16; & 0xFFFF gives its 16-bit pattern even if bit 15 is set.
        self.bool_mask = [format(int(b) & 0xFFFF, '016b') for b in flags]

        self.APOGEE_masking()

    # =============================================================================
    # =============================================================================
    def _load_lamost(self, hdul_sp, min_wv, max_wv):
        '''
        LAMOST low-resolution loader (DR5 data model,
        https://www.lamost.org/dr5/v3/doc/data-production-description; same
        layout in DR9 v2.0): primary array row 0 is the flux; the log-linear
        dispersion is log10(lambda_i) = COEFF0 + COEFF1 * i for 0-based pixel
        i (CRPIX1 = 1, "central wavelength (log10) of first pixel").
        VACUUM=T marks vacuum wavelengths; HELIO=T marks spectra already
        shifted to the heliocentric frame (HELIO_RV is the velocity used).
        DATE-OBS is the median UTC of the exposures.
        '''
        self.n_orders = 1
        header = hdul_sp[0].header
        data = hdul_sp[0].data
        if data is None or data.ndim != 2:
            raise ValueError("LAMOST loader expects the primary data array "
                             "(flux, inverse variance, wavelength, andmask, ormask)")
        s = np.array(data[0], dtype=float)
        w = 10 ** (float(header['COEFF0']) + float(header['COEFF1']) * np.arange(s.size))

        if 'HELIO' not in header:
            raise ValueError("LAMOST header has no HELIO keyword: cannot tell whether the "
                             "spectrum is heliocentric-corrected")
        if not header['HELIO']:
            raise NotImplementedError("LAMOST spectrum is not heliocentric-corrected (HELIO=F); "
                                      "the sign convention of HELIO_RV is undocumented")
        self.specsys = 'HELIOCEN'
        self.bcv = 0.0
        self.wavelength_frame = {True: 'vacuum', False: 'air'}.get(header.get('VACUUM'))

        window = _wavelength_window(w, min_wv, max_wv)
        self.s = np.reshape(s[window], (1, -1))
        self.w = np.reshape(w[window], (1, -1))

        self.DATE_OBS = Time(header['DATE-OBS'], format='isot', scale='utc').jd
        self.time_system = 'JD-UTC'

    # =============================================================================
    # =============================================================================
    def _load_elodie(self, hdul_sp, min_wv, max_wv):
        '''
        ELODIE archive 'spec' loader: a single reconnected spectrum resampled
        to a constant step (0.05 Angstrom, 4000-6800 Angstrom) with a linear
        FITS WCS, lambda = CRVAL1 + CDELT1 * (p - CRPIX1) for 1-based pixel p.
        The archive documents these spectra as topocentric, "not corrected
        for the Earth's motion"
        (https://web.archive.org/web/2015/http://www.obs-hp.fr/archive/elodie-for-dummies.shtml);
        no BCV keyword is known, so none is applied and a warning is issued.
        '''
        self.n_orders = 1
        header = hdul_sp[0].header
        s = np.array(hdul_sp[0].data, dtype=float)
        pix = np.arange(1, s.size + 1)
        w = float(header['CRVAL1']) + float(header['CDELT1']) * (pix - float(header.get('CRPIX1', 1.0)))

        self.snr = header["SN"]

        window = _wavelength_window(w, min_wv, max_wv)
        self.s = np.reshape(s[window], (1, -1))
        self.w = np.reshape(w[window], (1, -1))

        # UNVERIFIED: DATE-OBS taken as exposure start, UTC.
        self.DATE_OBS = Time(header['DATE-OBS'], format='isot', scale='utc').jd
        self.time_system = 'JD-UTC'
        self.specsys = 'TOPOCENT'
        # UNVERIFIED: air wavelengths (the ELODIE library is in air; the archive
        # spec product is not documented either way).
        self.wavelength_frame = None
        self.bcv = 0.0
        warnings.warn("ELODIE archive spectra are topocentric and SPARTA knows no BCV keyword "
                      "for them: no barycentric correction is applied.", stacklevel=3)

    # =============================================================================
    # =============================================================================
    def _load_nres_banzai(self, hdul_sp, min_wv, max_wv):
        '''
        BANZAI-NRES loader (https://banzai-nres.readthedocs.io): per-order
        'wavelength' (Angstrom, vacuum) and 'normflux' columns of the science
        fiber; BARYCORR (m/s) is the correction still to apply; TCORR is the
        mid-exposure BJD_TDB.
        '''
        rows = hdul_sp[1].data[hdul_sp[1].data['fiber'] == hdul_sp[0].header['SCIFIBER']]
        self.n_orders = 68      # Not sure why this is 68 rather than 67
        self.s = [np.array(rows['normflux'][k], dtype=float) for k in range(self.n_orders)]
        self.w = [np.array(rows['wavelength'][k], dtype=float) for k in range(self.n_orders)]

        self.bcv = float(hdul_sp[0].header['BARYCORR']) / 1000      # m/s -> km/s
        self.specsys = 'TOPOCENT'
        self.wavelength_frame = 'vacuum'

        self.DATE_OBS = float(hdul_sp[0].header["TCORR"])
        self.time_system = 'BJD-TDB'

    # =============================================================================
    # =============================================================================
    def _load_tres(self, hdul_sp, min_wv, max_wv):
        '''
        TRES loader: IRAF echelle spectrum, one order per image line, with a
        non-linear multispec dispersion function per order in the header
        (http://tdc-www.harvard.edu/instruments/tres/reduce.html). Returns
        the order that best covers [min_wv, max_wv]. BCV (km/s) is "the
        barycentric velocity correction" added by the CfA reduction
        (http://tdc-www.harvard.edu/instruments/tres/process/process2018.html).
        '''
        self.n_orders = 1
        header = hdul_sp[0].header
        data = np.array(hdul_sp[0].data, dtype=float)
        if data.ndim != 2:
            raise ValueError(f"TRES loader expects a 2-D (order, pixel) image, got shape {data.shape}")
        waves = _iraf_multispec_wavelengths(header)
        if sorted(waves) != list(range(1, data.shape[0] + 1)):
            raise ValueError(f"multispec apertures {sorted(waves)} do not match the "
                             f"{data.shape[0]} image lines")
        w, s = _best_order([waves[n] for n in range(1, data.shape[0] + 1)],
                           [data[n - 1, :waves[n].size] for n in range(1, data.shape[0] + 1)],
                           min_wv, max_wv)
        self.w = np.reshape(w, (1, -1))
        self.s = np.reshape(s, (1, -1))

        # UNVERIFIED: WS_BJD and BCV are not in the public TRES documentation
        # (which lists HJDN/GJDN/MJD-MID); BCV assumed in km/s, WS_BJD a BJD of
        # unknown scale, and ThAr (IRAF) solutions in air.
        self.DATE_OBS = float(header["WS_BJD"])
        self.time_system = 'BJD (scale unverified)'
        self.bcv = float(header["BCV"])
        self.specsys = 'TOPOCENT'
        self.wavelength_frame = None

    # =============================================================================
    # =============================================================================
    def _load_uves(self, hdul_sp, min_wv, max_wv):
        '''
        UVES loader -- ESO Phase-3 SDP 1-D spectrum. Per the UVES echelle
        release description
        (https://www.eso.org/sci/observing/phase3/data_releases/IDP_UVES_ECH_release_description_v1.3.pdf)
        no barycentric correction is applied to the wavelengths; it is stored
        in HIERARCH ESO QC VRAD BARYCOR (km/s). Uses the flux-calibrated FLUX
        column when present, else FLUX_REDUCED. MJD-OBS is the exposure start.
        '''
        self.n_orders = 1
        table = hdul_sp[1].data
        flux_column = 'FLUX' if 'FLUX' in table.columns.names else 'FLUX_REDUCED'
        w = np.array(table['WAVE'][0], dtype=float)
        s = np.array(table[flux_column][0], dtype=float)

        self.snr = hdul_sp[0].header["SNR"]
        self.specsys, self.wavelength_frame = _eso_sdp_frames(hdul_sp)
        self.bcv = _bcv_for_specsys(hdul_sp[0].header, self.specsys, 'HIERARCH ESO QC VRAD BARYCOR')

        self.DATE_OBS = float(hdul_sp[0].header["MJD-OBS"])
        self.time_system = 'MJD-UTC'

        window = _wavelength_window(w, min_wv, max_wv)
        self.s = np.reshape(s[window], (1, -1))
        self.w = np.reshape(w[window], (1, -1))

    # =============================================================================
    # =============================================================================
    def _load_feros(self, hdul_sp, min_wv, max_wv):
        '''
        FEROS loader -- ESO Phase-3 SDP 1-D spectrum. The FEROS release
        description (http://www.eso.org/rm/api/v1/public/releaseDescriptions/94)
        states that the HIERARCH ESO DRS BARYCORR correction is already
        applied to the spectra and that wavelengths are in air, so the BCV is
        applied only if SPECSYS says the product is topocentric. Time is the
        exposure midpoint, (MJD-OBS + MJD-END) / 2.
        '''
        self.n_orders = 1
        table = hdul_sp[1].data
        w = np.array(table['WAVE'][0], dtype=float)
        s = np.array(table['FLUX'][0], dtype=float)

        self.specsys, self.wavelength_frame = _eso_sdp_frames(hdul_sp)
        self.bcv = _bcv_for_specsys(hdul_sp[0].header, self.specsys, 'HIERARCH ESO DRS BARYCORR')

        self.DATE_OBS = (hdul_sp[0].header["MJD-OBS"] + hdul_sp[0].header["MJD-END"]) / 2
        self.time_system = 'MJD-UTC'

        window = _wavelength_window(w, min_wv, max_wv)
        self.s = np.reshape(s[window], (1, -1))
        self.w = np.reshape(w[window], (1, -1))

        self.snr = hdul_sp[0].header["SNR"]

    # =============================================================================
    # =============================================================================
    def _load_ceres(self, hdul_sp, min_wv, max_wv):
        '''
        Shared reader for CERES pipeline cubes (Brahm et al. 2017,
        https://github.com/rabrahm/ceres): data[0] wavelength and data[3]
        blaze-corrected flux, per order. CERES multiplies the wavelength
        solution by lambda_bary/lambda_topo, so the spectra are barycentric;
        HIERARCH MBJD is BJD - 2400000.5 computed from the UTC MJD.
        Returns the order that best covers [min_wv, max_wv].
        '''
        self.n_orders = 1
        data = np.array(hdul_sp[0].data, dtype=float)
        w, s = _best_order(data[0], data[3], min_wv, max_wv)
        self.w = np.reshape(w, (1, -1))
        self.s = np.reshape(s, (1, -1))

        header = hdul_sp[0].header
        self.DATE_OBS = float(header['HIERARCH MBJD'])
        self.time_system = 'MBJD-UTC'
        self.bcv = 0.0
        self.specsys = 'BARYCENT'
        self.snr = header["SNR"]
        self.vrad = header["RV"]

    def _load_harps_15(self, hdul_sp, min_wv, max_wv):
        '''
        HARPS reduced with CERES (see `_load_ceres`); CERES converts the
        HARPS wavelengths to vacuum (harps/harpspipe.py, ToVacuum).
        '''
        # UNVERIFIED: identification of HARPS_15 / FEROS_15 files as CERES
        # output rests on their header keywords (MBJD, BARYCENTRIC CORRECTION).
        self._load_ceres(hdul_sp, min_wv, max_wv)
        self.wavelength_frame = 'vacuum'

    def _load_feros_15(self, hdul_sp, min_wv, max_wv):
        '''
        FEROS reduced with CERES (see `_load_ceres`).
        '''
        self._load_ceres(hdul_sp, min_wv, max_wv)
        # UNVERIFIED: CERES' FEROS solution is not passed through ToVacuum
        # (feros/ferospipe.py) but its CCF masks are, so the frame is unclear.
        self.wavelength_frame = None

    # =============================================================================
    # =============================================================================
    def _load_coralie(self, hdul_sp, min_wv, max_wv):
        '''
        CORALIE loader: Geneva DRS s1d image with a linear FITS WCS,
        lambda = CRVAL1 + CDELT1 * (p - CRPIX1) for 1-based pixel p.
        '''
        self.n_orders = 1
        header = hdul_sp[0].header
        s = np.array(hdul_sp[0].data, dtype=float)
        pix = np.arange(1, s.size + 1)
        w = float(header["CRVAL1"]) + float(header["CDELT1"]) * (pix - float(header.get('CRPIX1', 1.0)))

        self.DATE_OBS = float(header["HIERARCH ESO DRS BJD"])
        self.time_system = 'BJD (Geneva DRS; scale unverified)'
        # UNVERIFIED: barycentric, air wavelengths -- assumed from the HARPS
        # s1d product of the same DRS family (whose SDP header says so).
        self.specsys = 'BARYCENT'
        self.wavelength_frame = None
        self.bcv = 0.0

        window = _wavelength_window(w, min_wv, max_wv)
        self.s = np.reshape(s[window], (1, -1))
        self.w = np.reshape(w[window], (1, -1))

    # =============================================================================
    # =============================================================================
    def _load_srg(self, hdul_sp, min_wv, max_wv):
        '''
        SRG loader: 1-D image with a linear FITS WCS,
        lambda = CRVAL1 + CDELT1 * (p - CRPIX1) for 1-based pixel p.
        '''
        # UNVERIFIED: no documentation found for this product; frame, time
        # scale and whether the spectra are barycentric are unknown.
        self.n_orders = 1
        header = hdul_sp[0].header
        s = np.array(hdul_sp[0].data, dtype=float)
        pix = np.arange(1, s.size + 1)
        w = float(header["CRVAL1"]) + float(header["CDELT1"]) * (pix - float(header.get('CRPIX1', 1.0)))

        self.air_mass = header.get("AIRMASS")

        self.DATE_OBS = float(header["BJD"])
        self.time_system = 'BJD (scale unverified)'
        self.bcv = 0.0

        window = _wavelength_window(w, min_wv, max_wv)
        self.s = np.reshape(s[window], (1, -1))
        self.w = np.reshape(w[window], (1, -1))

    # =============================================================================
    # =============================================================================
    def _load_kek(self, hdul_sp, min_wv, max_wv):
        '''
        KEK loader placeholder: the reader was never completed (it set no
        flux, wavelength or observation date).

        :raises NotImplementedError: always.
        '''
        raise NotImplementedError(
            "The KEK FITS loader is not implemented: it never set the flux, "
            "wavelengths or observation date. Convert the data to a supported "
            "format or implement ReadSpec._load_kek.")

    # =============================================================================
    # =============================================================================
    def _load_pepsi(self, hdul_sp, min_wv, max_wv):
        '''
        PEPSI loader: binary table with wavelength (column 0) and normalized
        flux (column 1). Per the NASA Exoplanet Archive PEPSI notes
        (https://exoplanetarchive.ipac.caltech.edu/docs/PEPSIMission.html) the
        wavelengths are in air and already shifted to the stellar rest frame
        using the barycentric (OBSVEL) and catalogue stellar (RADVEL)
        velocities -- so RVs measured from them are relative to RADVEL.
        '''
        self.n_orders = 1
        table = hdul_sp[1].data
        w = np.array(table.field(0), dtype=float)
        s = np.array(table.field(1), dtype=float)

        window = _wavelength_window(w, min_wv, max_wv)
        self.s = np.reshape(s[window], (1, -1))
        self.w = np.reshape(w[window], (1, -1))

        self.bcv = 0.0
        self.specsys = 'SOURCE'
        self.wavelength_frame = 'air'
        # UNVERIFIED: the wavelength unit (Angstrom assumed; the Exoplanet
        # Archive notes say microns) and the time scale of JD-OBS (mid-exposure).
        self.DATE_OBS = float(hdul_sp[0].header["JD-OBS"])
        self.time_system = 'JD-UTC'

# =============================================================================
# =============================================================================

    def retrieve_all_spectrum_parameters(self):
        """
        This function returns the visit parameters read.
        :return: all class features:
        w: wavelength vector
        s: spectrum vector
        bool_mask: str, APOGEE bool mask as described here: https://www.sdss.org/dr15/algorithms/bitmasks/#APOGEE_PIXMASK
        DATE-OBS: float, observation time in days (see ``time_system``)
        vrad: float, radial velocity given in the fits file
        """
        return self.w, self.s, self.bool_mask, self.DATE_OBS, self.vrad, self.bcv, self.snr, self.air_mass

# =============================================================================
# =============================================================================
    def APOGEE_masking(self):
        '''
        This function removes bad pixels (marked as sky flux or telluric line) from an APOGEE spectra
        and fills them by using linear interpolation
        APOGEE bool mask as described here: https://www.sdss.org/dr15/algorithms/bitmasks/#APOGEE_PIXMASK)
        '''

        if self.survey != "APOGEE":
            raise ValueError("APOGEE_masking only takes APOGEE spectra")

        # Bits 12/13 of APOGEE_PIXMASK (sky line / telluric), read from the
        # bit strings the loader stores, plus any pixel that is already NaN.
        flux = np.asarray(self.s[0], float).copy()
        bad = np.array([flag[3] == '1' or flag[2] == '1' for flag in self.bool_mask])
        bad |= ~np.isfinite(flux)
        if bad.all():
            raise ValueError("every APOGEE pixel is masked; nothing to interpolate from")
        # Linear interpolation across masked pixels (in pixel index, as before);
        # masked runs at the edges take the nearest good value instead of the
        # former zero flux, which a CCF would read as a deep line.
        idx = np.arange(len(flux))
        flux[bad] = np.interp(idx[bad], idx[~bad], flux[~bad])
        self.s[0] = flux

        return self
