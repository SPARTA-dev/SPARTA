"""Byte-identity baseline of the object layer, pinned before its steps become atoms.

Every case is the SHA-256 of (shape, float64 bytes) of an object output on a
seeded synthetic input (plus the bundled NRES and HARPS files), with the plain
sum beside it so a failure says roughly how far it moved:

* ``spectrum.*``: the Spectrum after each preprocessing step -- trim (with and
  without margins), interpolate (linear and log, units, chained, seeded native
  step), remove_cosmics, filter_spectrum and the full
  preprocess -- with ``native_steps()`` after each;
* ``nres.*``, ``harps.*``, ``sim.*``: the bundled files and a simulated
  archive product, through the readers, ``Observations`` ingestion (the
  native-step seeding) and preprocess;
* ``template.*``: PHOENIX models (vacuum, air) and ``integrate_spec``;
* ``ccf1d.*``: ``Corr`` and ``CorrCombined`` arrays in both output modes;
* ``ccf2d.*``: coarse and fine maps, RVs, errors, amplitudes and per-window
  noise, for sigma=None and an explicit sigma, in both output modes;
* ``tiravel.*``: rho, weights, correlation matrix and template;
* ``periodogram.*``: GLS, PDC (plain and error-aware), USuRPER, the partial
  periodograms and ``calc_pdc_pval``.

Exact comparison, no tolerance, so the hashes are only meaningful on the
environment that wrote them: the JSON records the Python, numpy, scipy,
astropy and specutils versions, and the test skips elsewhere. The 1e-9
goldens (``tests/golden/pipeline.json``, ``instruments.json``) remain the
cross-platform anchor. Velocities in km/s, wavelengths in Angstrom, times in
days, frequencies in 1/day.

Re-bless only on purpose, in its own commit:
``python tests/_atoms_baseline.py --write``.
"""
import copy
import hashlib
import json
import pathlib
import platform
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if __name__ == "__main__":
    sys.path[:0] = [str(ROOT / p) for p in ("src", "packages/unicor/src", "packages/todcor/src")]

import astropy
import scipy
import specutils
from astropy import units as u
from todcor import composite_flux
from unicor import dsp

from sparta import instruments
from sparta.ccf1d import CCF1d
from sparta.ccf2d import CCF2d
from sparta.instruments import Simulation
from sparta.observations import Observations
from sparta.periodogram import PeriodicityDetector
from sparta.spectrum import Spectrum
from sparta.template import Template
from sparta.timeseries import TimeSeries
from sparta.tiravel import TIRAVEL

BASELINE_PATH = ROOT / "tests" / "golden" / "atoms_sha256.json"
NRES_FILE = ROOT / "examples" / "data" / "TOI677" / "lscnrs01-fa09-20190520-0043-e91.fits"
HARPS_FILE = ROOT / "examples" / "data" / "TOI193" / "ADP.2018-11-08T01_14_11.387.fits"

C_KMS = dsp.C_KMS
V_SB1 = 23.4                                         # km/s
ORDERS_A = ((5000.0, 5060.0, 3000), (5100.0, 5170.0, 3400), (5200.0, 5250.0, 2600))


def environment():
    """The versions the byte-identity is pinned to (dict of str)."""
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "astropy": astropy.__version__,
            "specutils": specutils.__version__}


def _digest(*arrays):
    '''SHA-256 of the shapes and float64 bytes of ``arrays``, and their nansum.'''
    h = hashlib.sha256()
    total = 0.0
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
        h.update(repr(a.shape).encode())
        h.update(a.tobytes())
        total += float(np.nansum(a))
    return {"sha256": h.hexdigest(), "sum": total}


# --- seeded synthetic inputs ------------------------------------------------

def _line_forest(seed, lo, hi, n_lines, fwhm_kms):
    """A rest-frame forest of Gaussian absorption lines, as a function of lambda."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(lo, hi, n_lines)
    depths = rng.uniform(0.1, 0.6, n_lines)
    sig = fwhm_kms / 2.3548200450309493 * centres / C_KMS

    def flux(wave):
        wave = np.asarray(wave, float)
        return np.prod(1.0 - depths * np.exp(-0.5 * ((wave[:, None] - centres) / sig) ** 2),
                       axis=1)
    return flux


SB1_LINES = _line_forest(101, 4990.0, 5260.0, 240, 9.0)


def _raw_spectrum():
    """Three linear-grid orders at V_SB1: blaze, noise, padded noisy margins
    (one with exact zeros), NaN pixels and cosmic-ray hits."""
    rng = np.random.default_rng(20261014)
    wv, sp = [], []
    for k, (lo, hi, n) in enumerate(ORDERS_A):
        w = np.linspace(lo, hi, n)
        blaze = 1000.0 * (1.0 - 0.5 * ((w - w.mean()) / (hi - lo)) ** 2)
        f = blaze * SB1_LINES(w / (1.0 + V_SB1 / C_KMS))
        f = f + np.sqrt(f) * rng.normal(size=n)
        f[:25] = rng.normal(0.0, 5.0, 25)
        f[-30:] = rng.normal(0.0, 5.0, 30)
        if k == 1:
            f[:8] = 0.0
        f[rng.choice(np.arange(100, n - 100), 4, replace=False)] += 5000.0
        wv.append(w)
        sp.append(f)
    sp[0][[400, 401, 1700]] = np.nan
    return Spectrum(wv=wv, sp=sp, bjd=2459000.5, bcv=4.2, name="raw")


def _normalized_spectrum():
    """The SB1 orders continuum-normalized, noisy, no margins (km/s, Angstrom)."""
    rng = np.random.default_rng(7)
    wv, sp = [], []
    for lo, hi, n in ORDERS_A:
        w = np.linspace(lo + 5.0, hi - 5.0, n)
        wv.append(w)
        sp.append(SB1_LINES(w / (1.0 + V_SB1 / C_KMS)) + rng.normal(0.0, 0.01, n))
    return Spectrum(wv=wv, sp=sp, name="norm")


def _sb1_template_arrays():
    wave = np.arange(4960.0, 5290.0, 0.01)
    return wave, SB1_LINES(wave)


# --- recorders ---------------------------------------------------------------

def _spectrum_cases(out, name, spec):
    out[f"{name}.wv"] = _digest(*spec.wv)
    out[f"{name}.sp"] = _digest(*spec.sp)
    out[f"{name}.native"] = _digest(spec.native_steps())


def _record_spectrum(out):
    raw = _raw_spectrum()

    def fresh(s=raw):
        return copy.deepcopy(s)

    _spectrum_cases(out, "spectrum.raw", fresh())
    _spectrum_cases(out, "spectrum.trim.default", fresh().trim())
    trimmed = fresh().trim(n_trim=10, clean_margins=True)
    _spectrum_cases(out, "spectrum.trim.n10_clean", trimmed)
    _spectrum_cases(out, "spectrum.trim.lr_clean_custom",
                    fresh().trim(n_trim_left=3, n_trim_right=7, clean_margins=True,
                                 margin_fraction=0.2, margin_window=9))
    keep_nans = fresh().trim(n_trim=5, remove_nans=False)
    _spectrum_cases(out, "spectrum.trim.keep_nans", keep_nans)
    _spectrum_cases(out, "spectrum.trim.clean_keep_nans",
                    fresh().trim(clean_margins=True, remove_nans=False))

    interps = {
        "linear_default": lambda s: s.interpolate(),
        "linear_1.7": lambda s: s.interpolate(delta=1.7),
        "linear_quantity": lambda s: s.interpolate(delta=0.25 * u.Angstrom),
        "log_default": lambda s: s.interpolate(method="log"),
        "log_2.5": lambda s: s.interpolate(delta=2.5, method="log"),
        "log_quantity": lambda s: s.interpolate(delta=800.0 * u.m / u.s, method="log"),
        "linear_then_log": lambda s: s.interpolate(delta=0.5).interpolate(delta=1.0,
                                                                            method="log"),
        "log_then_linear_coarse": lambda s: s.interpolate(delta=0.3, method="log")
                                             .interpolate(delta=2.0),
    }
    for key, step in interps.items():
        spec = step(fresh(trimmed))
        _spectrum_cases(out, f"spectrum.interp.{key}", spec)
        out[f"spectrum.interp.{key}.grid_delta"] = _digest(
            u.Quantity(spec.Info["GridDelta"]).value)

    seeded = fresh(trimmed)
    seeded.Info["NativeStep"] = [0.9, 0.2, 0.9]
    _spectrum_cases(out, "spectrum.interp.seeded_native", seeded.interpolate(delta=0.5))

    linear = fresh(trimmed).interpolate()
    _spectrum_cases(out, "spectrum.cosmics.3", fresh(linear).remove_cosmics())
    _spectrum_cases(out, "spectrum.cosmics.5", fresh(linear).remove_cosmics(sigma_upper=5))
    _spectrum_cases(out, "spectrum.cosmics.nans", fresh(keep_nans).remove_cosmics())
    clean = fresh(linear).remove_cosmics()
    filtered = fresh(clean).filter_spectrum(continuum_scale_kms=1200.0, noise_scale_kms=8.0,
                                            order=1)
    _spectrum_cases(out, "spectrum.filter.o1", filtered)
    _spectrum_cases(out, "spectrum.filter.o2",
                    fresh(clean).filter_spectrum(continuum_scale_kms=900.0, noise_scale_kms=4.0,
                                                 order=2))

    _spectrum_cases(out, "spectrum.preprocess.default", fresh().preprocess())
    _spectrum_cases(out, "spectrum.preprocess.custom",
                    fresh().preprocess(n_trim=20, clean_margins=False, delta=1.2,
                                       cosmic_sigma=4, continuum_scale_kms=700.0, noise_scale_kms=9.5))
    _spectrum_cases(out, "spectrum.preprocess.then_log",
                    fresh().preprocess().interpolate(delta=0.5, method="log"))
    _spectrum_cases(out, "spectrum.barycentric", fresh(trimmed).barycentric_correction())


def _record_bundled(out):
    nres = instruments.get("NRES")
    rec = nres.load(str(NRES_FILE), min_wv=5000, max_wv=5400)
    spec = Spectrum(wv=rec.wave, sp=rec.flux, bjd=rec.time, bcv=rec.bcv)
    spec.select_orders(list(nres.recommended.drop_orders), remove=True)
    spec.barycentric_correction()
    _spectrum_cases(out, "nres.ingest", spec)
    pre = copy.deepcopy(spec).preprocess(**nres.recommended.preprocess)
    _spectrum_cases(out, "nres.preprocess", pre)
    _spectrum_cases(out, "nres.preprocess.log", pre.interpolate(delta=0.5, method="log"))

    harps = instruments.get("HARPS")
    with tempfile.TemporaryDirectory() as tmp:
        (pathlib.Path(tmp) / HARPS_FILE.name).symlink_to(HARPS_FILE)
        obs = Observations(survey="HARPS", target_visits_lib=tmp, min_wv=4990, max_wv=5090)
    spec = obs.spec_list[0]
    _spectrum_cases(out, "harps.ingest", spec)
    pre = copy.deepcopy(spec).preprocess(**harps.recommended.preprocess)
    _spectrum_cases(out, "harps.preprocess", pre)
    _spectrum_cases(out, "harps.preprocess.log", pre.interpolate(delta=0.1, method="log"))

    sim = Simulation(like="HARPS", components=[dict(teff=5800, logg=3.5, feh=0.5, vsini=3.0)],
                     times=np.array([0.0, 1.3]), rv_kms=np.array([5.0, -7.0]),
                     windows_A=[(5000.0, 5030.0), (5500.0, 5520.0)], snr=80.0,
                     noise="constant", seed=3, as_product=True)
    obs = Observations(survey="SIMULATED", simulation=sim)
    for k, spec in enumerate(obs.spec_list):
        _spectrum_cases(out, f"sim.ingest.{k}", spec)
        _spectrum_cases(out, f"sim.log.{k}",
                        copy.deepcopy(spec).interpolate(delta=0.2, method="log"))


def _record_templates(out):
    for key, kwargs in (("phoenix_vac", dict(air=False)), ("phoenix_air", dict(air=True))):
        t = Template.from_phoenix(5800, 3.5, 0.5, wave_range=(5000.0, 5100.0),
                                  allow_download=False, **kwargs)
        out[f"template.{key}.wv"] = _digest(t.model.wv)
        out[f"template.{key}.sp"] = _digest(t.model.sp)
    t = Template.from_phoenix(4900, 4.5, 0.0, 0.0, wave_range=(6000.0, 6020.0),
                              allow_download=False)
    out["template.phoenix_4900.wv"] = _digest(t.model.wv)
    out["template.phoenix_4900.sp"] = _digest(t.model.sp)

    wave, flux = _sb1_template_arrays()
    native = Template(Spectrum(wv=[wave[:6000]], sp=[flux[:6000]]).interpolate(delta=0.5))
    new_wv, new_sp = native.integrate_spec(3)
    out["template.integrate.native.wv"] = _digest(new_wv)
    out["template.integrate.native.sp"] = _digest(new_sp)
    out["template.integrate.native.native"] = _digest(native.model.native_steps())
    coarse = Template(Spectrum(wv=[wave[:6000]], sp=[flux[:6000]]).interpolate(delta=0.1))
    coarse.integrate_spec(25)
    out["template.integrate.coarse.native"] = _digest(coarse.model.native_steps())
    plain = Template.from_arrays(wave[:5000], flux[:5000])
    new_wv, new_sp = plain.integrate_spec(4)
    out["template.integrate.plain.wv"] = _digest(new_wv)
    out["template.integrate.plain.sp"] = _digest(new_sp)
    out["template.integrate.plain.native"] = _digest(plain.model.native_steps())


def _ccf1d_cases(out, name, ccf):
    corr = ccf.Corr
    for key in ("vel", "corr", "loglike", "RV", "eRV", "peakCorr", "n_eff"):
        out[f"{name}.Corr.{key}"] = _digest(corr[key])
    comb = ccf.CorrCombined
    for key in ("vel", "corr", "loglike"):
        out[f"{name}.CorrCombined.{key}"] = _digest(comb[key])
    out[f"{name}.CorrCombined.scalars"] = _digest(
        [comb["RV"], comb["eRV"], comb["peakCorr"], comb["n_eff"]])


def _record_ccf1d(out):
    wave, flux = _sb1_template_arrays()
    template_pre = Template(Spectrum(wv=[wave], sp=[flux]).preprocess())
    template_raw = Template.from_arrays(wave, flux)
    observed_pre = _raw_spectrum().preprocess()
    for mode in ("ccf", "loglike"):
        ccf = CCF1d().cross_correlate(observed_pre, template_pre, dv=0.5, vel_bound=80,
                                      errors_per_order=True, output=mode).combine()
        _ccf1d_cases(out, f"ccf1d.preprocessed.{mode}", ccf)
        ccf = CCF1d().cross_correlate(_normalized_spectrum(), template_raw, vel_bound=60,
                                      output=mode).combine()
        _ccf1d_cases(out, f"ccf1d.normalized.{mode}", ccf)
    log_spec = _normalized_spectrum().interpolate(delta=0.3, method="log")
    ccf = CCF1d().cross_correlate(log_spec, template_raw, vel_bound=[-40, 70],
                                  errors_per_order=True,
                                  velocity_convention="relativistic").combine()
    _ccf1d_cases(out, "ccf1d.log_input_relativistic", ccf)


SB2_T1 = _line_forest(201, 4985.0, 5135.0, 90, 8.0)
SB2_T2 = _line_forest(202, 4985.0, 5135.0, 60, 12.0)


def _sb2_case():
    """Two linear-grid orders (0.02 A, ~1.2 km/s) of a noisy SB2 and its templates."""
    rng = np.random.default_rng(31)
    tw = np.arange(4960.0, 5160.0, 0.005)
    t1, t2 = SB2_T1(tw), SB2_T2(tw)
    wv, sp = [], []
    for lo, hi in ((4990.0, 5050.0), (5070.0, 5130.0)):
        w = np.arange(lo, hi, 0.02)
        f = composite_flux(w, [(tw, t1), (tw, t2)], [35.0, -48.0], [0.7, 0.35])
        wv.append(w)
        sp.append(f + rng.normal(0.0, 0.005, w.size))
    return Spectrum(wv=wv, sp=sp, name="sb2"), Template.from_arrays(tw, t1), \
        Template.from_arrays(tw, t2)


_MAP_KEYS = ("v1_grid", "v2_grid", "R", "logL_profile", "logL_marginal", "a1", "a2", "chi2",
             "map")
_BEST_KEYS = ("v1", "v2", "a1", "a2", "chi2", "chi2_red", "R")
_FIT_KEYS = ("RV1", "RV2", "eRV1", "eRV2", "rho12", "cov", "peak", "a1_best", "a2_best",
             "R_best", "chi2_best", "chi2_red")
_WINDOW_KEYS = ("dv", "S11", "S22", "SD", "oversampling", "n_eff", "nu", "sigma")


def _record_ccf2d(out):
    spec, t1, t2 = _sb2_case()
    for sigma_key, sigma in (("sigma_none", None), ("sigma_0.005", 0.005)):
        for mode in ("ccf", "loglike"):
            name = f"ccf2d.{sigma_key}.{mode}"
            ccf = CCF2d().cross_correlate(spec, t1, t2, dv=1.0, vel_bound=100, dvel=2.0,
                                          sigma=sigma, alpha=0.1, output=mode)
            res = ccf.Corr2d
            for level, maps in (("coarse", res), ("fine", res["fine"])):
                for key in _MAP_KEYS:
                    out[f"{name}.{level}.{key}"] = _digest(maps[key])
                out[f"{name}.{level}.best"] = _digest([maps["best"][k] for k in _BEST_KEYS])
            for key in _FIT_KEYS:
                out[f"{name}.{key}"] = _digest(res[key])
            out[f"{name}.sigma"] = _digest(res["sigma"])
            for key in _WINDOW_KEYS:
                out[f"{name}.windows.{key}"] = _digest([w[key] for w in ccf.windows])
            for key in ("wv", "data_g", "g1", "g2"):
                out[f"{name}.windows.{key}"] = _digest(*[w[key] for w in ccf.windows])


def _tiravel_epochs(velocities, n_ord=2, seed=41):
    """Noisy log-lambda epochs (1 km/s) of one line forest, Doppler-shifted."""
    rng = np.random.default_rng(seed)
    forest = _line_forest(301, 4400.0, 4700.0, 120, 12.0)
    tw = np.arange(4380.0, 4720.0, 0.01)
    rest = forest(tw)
    bounds = ((4410.0, 4540.0), (4560.0, 4690.0))[:n_ord]
    spectra = []
    for v in velocities:
        wv, sp = [], []
        for lo, hi in bounds:
            grid = dsp.log_lambda_grid(lo, hi, 1.0)[0]
            f = dsp.resample(tw * (1.0 + v / C_KMS), rest, grid, fill=1.0)
            wv.append(grid)
            sp.append(f + rng.normal(0.0, 0.01, grid.size))
        spectra.append(Spectrum(wv=wv, sp=sp, name="tiravel"))
    return spectra


def _record_tiravel(out):
    velocities = np.array([-31.0, -12.5, 4.0, 18.2, 27.7, 40.1])
    spectra = _tiravel_epochs(velocities)
    for key, kwargs in (("cubic", {}),
                        ("linear_ref2_m5", dict(kind="linear", reference=2, margin=5))):
        t = TIRAVEL().build_template(spectra, velocities + 0.3, **kwargs)
        out[f"tiravel.{key}.rho"] = _digest(t.rho)
        out[f"tiravel.{key}.weights"] = _digest(t.weights)
        out[f"tiravel.{key}.correlation"] = _digest(t.correlation)
        out[f"tiravel.{key}.eigenvalues"] = _digest(t.diagnostics["eigenvalues"])
        out[f"tiravel.{key}.template.wv"] = _digest(*t.template.model.wv)
        out[f"tiravel.{key}.template.sp"] = _digest(*t.template.model.sp)


def _detector(series):
    return PeriodicityDetector(time_series=series, freq_range=(0.02, 1.0),
                               periodogram_grid_resolution=40)


def _record_periodograms(out):
    rng = np.random.default_rng(5)
    times = np.sort(rng.uniform(0.0, 100.0, 30))
    vals = np.sin(2 * np.pi * times / 5.0) + 0.3 * rng.normal(size=times.size)
    errors = rng.uniform(0.1, 0.4, times.size)

    def scalar():
        return _detector(TimeSeries(times=list(times), vals=list(vals), errors=list(errors)))

    plain = scalar()
    plain.calc_gls()
    plain.calc_pdc(calc_biased_flag=True, calc_unbiased_flag=True)
    aware = scalar()
    aware.calc_pdc(calc_biased_flag=True, calc_unbiased_flag=True, errors=True)
    out["periodogram.scalar.frequency"] = _digest(plain.results_frequency["PDC_unbiased"])
    for key in ("GLS", "PDC_biased", "PDC_unbiased"):
        out[f"periodogram.scalar.{key}"] = _digest(plain.results_power[key])
    for key in ("PDC_errors_biased", "PDC_errors_unbiased"):
        out[f"periodogram.scalar.{key}"] = _digest(aware.results_power[key])
    out["periodogram.pval"] = _digest(plain.calc_pdc_pval(plain.results_power["PDC_unbiased"]))
    out["periodogram.pval.inverse"] = _digest(
        plain.calc_pdc_pval([0.1, 0.01, 1e-3, 1e-4], inverse_sf=True))

    # A single-order spectrum series: an SB1 at P = 3.7 d plus a line-depth
    # change at 11 d, for USuRPER and the partial periodograms.
    rng = np.random.default_rng(6)
    times = np.sort(rng.uniform(0.0, 60.0, 14))
    rv = 15.0 * np.sin(2 * np.pi * times / 3.7)
    depth = 1.0 + 0.2 * np.sin(2 * np.pi * times / 11.0)
    spectra = _tiravel_epochs(rv, n_ord=1, seed=42)
    for spec, d in zip(spectra, depth):
        spec.sp[0] = 1.0 - d * (1.0 - spec.sp[0])
    series = TimeSeries(times=list(times), vals=spectra,
                        calculated_vrad_list=list(rv + 0.05 * rng.normal(size=rv.size)))
    det = _detector(series)
    det.calc_usurper(calc_biased_flag=True, calc_unbiased_flag=True)
    det.calc_partial_periodogram(partial_type="shift")
    det.calc_partial_periodogram(partial_type="shape")
    det.calc_gls()
    det.calc_pdc(calc_biased_flag=True)
    for key, power in det.results_power.items():
        out[f"periodogram.spectral.{key}"] = _digest(power)


def compute():
    '''
    Every baseline case.

    Returns
    -------
    dict of str to dict
        Case name -> {'sha256': str, 'sum': float}.
    '''
    out = {}
    _record_spectrum(out)
    _record_bundled(out)
    _record_templates(out)
    _record_ccf1d(out)
    _record_ccf2d(out)
    _record_tiravel(out)
    _record_periodograms(out)
    return out


def write():
    """Rewrite BASELINE_PATH from ``compute()`` on this environment."""
    baseline = {
        "scheme": "sha256 of repr(shape) + float64 bytes, per array in order; sum = nansum; "
                  "exact on the recorded environment only",
        "versions": environment(),
        "cases": compute(),
    }
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        raise SystemExit("usage: python tests/_atoms_baseline.py --write")
    write()
