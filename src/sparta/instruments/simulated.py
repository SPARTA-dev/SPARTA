"""Simulated observations: spectra at known radial velocities, recorded as a
chosen archive product would record them -- injection-recovery data for
UNICOR (CCF1d) and TODCOR (CCF2d).

Use it through Observations, which ingests the records exactly like files::

    sim = Simulation(like="HARPS", components=[dict(teff=5800, logg=3.5, feh=0.5)],
                     times=t, rv_kms=rv, snr=100, seed=1)
    obs = Observations(survey="SIMULATED", simulation=sim)
    obs.truth.rv_kms            # what the RVs should come out as

The ``like`` product (e.g. 'HARPS', 'NRES') supplies the resolving power, the
native (detector) step, the air/vacuum frame, SPECSYS and, for
``as_product=True``, the archive grid.

Forward model, per component and order (velocities in km/s, wavelengths in
Angstrom):

1. The intrinsic spectrum: a PHOENIX model (vacuum, converted to air with
   Morton 2000 when the product is in air; a product of undocumented frame is
   simulated in vacuum), or a user Template taken as rest-frame and in the
   product's frame.
2. Continuum normalisation of each component on its own, by its upper
   envelope (`upper_envelope`). The composite is
   ``1 + sum_i a_i (f_i - 1)``, so the amplitudes ``a_i`` are the line
   dilutions TODCOR fits (a1, a2).
3. Cubic resampling onto a log-lambda grid ``FINE_FACTOR`` times finer than
   the native step, covering the order plus a margin (`margin_kms`).
4. Rotational (vsini, epsilon), then instrumental (Gaussian, FWHM c / R)
   broadening.
5. Per epoch, every component Doppler-shifted (`unicor.dsp.doppler_factor`,
   via `todcor.simulate.composite_flux`, cubic interpolation) and summed on the
   native pixel grid: log-lambda at the instrument's native step over the
   order window. The flux is sampled at pixel centres, not integrated over
   the pixels.
6. Noise on the native pixels: 'photon' (sigma = sqrt(flux) / snr) or
   'constant' (sigma = 1 / snr), snr being per native pixel at the continuum.
7. With ``as_product=True``, the noisy spectrum is resampled onto the archive
   grid with `sparta.UNICOR.Spectrum.resample_flux` (the archive pipeline's
   own rebinning kernel is unknown), and the records carry the native step.

For a topocentric product (SPECSYS 'TOPOCENT', e.g. NRES) the records carry
``bcv`` and their wavelengths are topocentric: a component at ``v`` is
observed at ``lambda_rest * D(v) / (1 + bcv/c)``, so that
``Spectrum.BarycentricCorrection`` (``lambda * (1 + bcv/c)``) returns it to
``lambda_rest * D(v)``, D the Doppler factor of ``velocity_convention``.
"""
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from astropy import units as u
from scipy.interpolate import interp1d
from todcor.simulate import composite_flux
from unicor import dsp

from sparta.UNICOR.Spectrum import Spectrum, resample_flux
from sparta.UNICOR.Template import Template
from sparta.instruments.base import Instrument, LoadedSpectrum, Product, Recommended

TIME_SYSTEM = "days (simulated)"
FINE_FACTOR = 8               # native step / fine-grid step
ENVELOPE_WIDTH_A = 10.0       # Angstrom, bin width of the upper envelope
ENVELOPE_PERCENTILE = 95.0    # per-bin percentile taken as the continuum
MARGIN_PAD_KMS = 10.0         # km/s beyond the shifts and kernels, see margin_kms


@dataclass(frozen=True, eq=False)
class Component:
    """
    One star of a simulated system.

    Parameters
    ----------
    teff, logg, feh, alpha : float or None
        PHOENIX-ACES model: effective temperature (K), log surface gravity
        (cgs), [Fe/H] and [alpha/Fe] (dex). Give these or ``template``.
    template : Template or None
        The intrinsic (unbroadened) rest-frame spectrum instead, order 0 of
        its model, in the frame of the ``like`` product.
    vsini : float
        Projected rotational velocity (km/s); 0 skips rotational broadening.
    epsilon : float
        Linear limb-darkening coefficient (0-1).
    amplitude : float
        Line dilution of this component: its normalised lines enter the
        composite scaled by ``amplitude`` (TODCOR's a1, a2).
    """
    teff: Optional[float] = None
    logg: Optional[float] = None
    feh: float = 0.0
    alpha: float = 0.0
    template: Optional[Template] = None
    vsini: float = 0.0
    epsilon: float = 0.6
    amplitude: float = 1.0

    def __post_init__(self):
        phoenix = self.teff is not None and self.logg is not None
        if phoenix == (self.template is not None):
            raise ValueError("a component needs either PHOENIX parameters (teff, logg) "
                             "or a template, not both or neither")
        object.__setattr__(self, "vsini", u.Quantity(self.vsini, u.km / u.s).to_value(u.km / u.s))
        if self.vsini < 0:
            raise ValueError(f"vsini must be non-negative (km/s), got {self.vsini}")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError(f"epsilon must lie in [0, 1], got {self.epsilon}")


@dataclass(frozen=True, eq=False)
class Simulation:
    """
    What to simulate. Plain numbers are in the stated units; astropy
    Quantities are converted.

    Parameters
    ----------
    like : str
        Registry key of the product to emulate (e.g. 'HARPS', 'NRES'); its
        instrument must declare a resolving power and a native step.
    components : list of Component or dict
        The stars (dicts are passed to `Component`).
    times : array_like
        Epochs (days), shape (n_epochs,).
    rv_kms : array_like
        Radial velocities (km/s), shape (n_epochs, n_components); (n_epochs,)
        for one component.
    windows_A : list of (float, float) or None
        One (lo, hi) wavelength window (Angstrom) per order; None takes the
        product's recommended window as one order.
    bcv_kms : array_like or None
        Barycentric velocity correction (km/s) per epoch; only for a
        topocentric product. None means 0.
    snr : float, array_like or None
        Signal-to-noise per native pixel at the continuum, scalar or per
        epoch; None for noiseless spectra.
    noise : {'photon', 'constant'}
        Noise sigma sqrt(flux) / snr or 1 / snr.
    seed : int or None
        Seed of ``np.random.SeedSequence``, spawned into one generator per
        epoch; None draws fresh entropy (recorded in the truth).
    as_product : bool
        Also emulate the archive rebinning onto the product's grid (needs a
        resampled product with a documented ``grid_step_A``).
    velocity_convention : {'cz', 'relativistic'}
        Doppler convention of ``rv_kms`` (see `unicor.dsp.doppler_factor`).
    allow_download : bool
        Let Template fetch PHOENIX files that are not cached locally.
    """
    like: str
    components: list
    times: np.ndarray
    rv_kms: np.ndarray
    windows_A: Optional[list] = None
    bcv_kms: Optional[np.ndarray] = None
    snr: Optional[np.ndarray] = None
    noise: str = "photon"
    seed: Optional[int] = None
    as_product: bool = False
    velocity_convention: str = "cz"
    allow_download: bool = False
    product: Product = field(init=False, repr=False)

    def __post_init__(self):
        from sparta import instruments      # deferred: the registry imports this module

        def set_(name, value):
            object.__setattr__(self, name, value)

        product = instruments.get(self.like)
        instrument = product.instrument
        if product.synthetic or instrument.resolving_power is None \
                or instrument.native_step_kms is None:
            raise ValueError(f"cannot simulate like {self.like!r}: its instrument needs a "
                             "documented resolving power and native step")
        set_("product", product)

        components = [c if isinstance(c, Component) else Component(**c) for c in self.components]
        if not components:
            raise ValueError("at least one component is needed")
        set_("components", components)

        times = np.atleast_1d(u.Quantity(self.times, u.day).to_value(u.day)).astype(float)
        if times.ndim != 1:
            raise ValueError(f"times must be 1-D (days), got shape {times.shape}")
        set_("times", times)
        n_epochs, n_comp = times.size, len(components)

        rv = u.Quantity(self.rv_kms, u.km / u.s).to_value(u.km / u.s).astype(float)
        if rv.ndim == 1 and n_comp == 1:
            rv = rv[:, None]
        if rv.shape != (n_epochs, n_comp):
            raise ValueError(f"rv_kms has shape {rv.shape}; expected (n_epochs, n_components) "
                             f"= {(n_epochs, n_comp)}")
        set_("rv_kms", rv)

        windows = self.windows_A
        if windows is None:
            if product.recommended.window_A is None:
                raise ValueError(f"{self.like!r} recommends no window; pass windows_A")
            windows = [product.recommended.window_A]
        windows = u.Quantity(windows, u.Angstrom).to_value(u.Angstrom).astype(float)
        if windows.ndim != 2 or windows.shape[1] != 2 or np.any(windows[:, 0] >= windows[:, 1]):
            raise ValueError("windows_A must be one (lo, hi) pair per order, lo < hi (Angstrom)")
        set_("windows_A", [tuple(w) for w in windows])

        if self.bcv_kms is None:
            bcv = np.zeros(n_epochs)
        else:
            bcv = np.broadcast_to(u.Quantity(self.bcv_kms, u.km / u.s).to_value(u.km / u.s),
                                  (n_epochs,)).astype(float)
            if product.specsys != "TOPOCENT" and np.any(bcv != 0):
                raise ValueError(f"{self.like!r} is {product.specsys}, not topocentric: its "
                                 "wavelengths need no barycentric correction (bcv_kms)")
        set_("bcv_kms", bcv)

        if self.snr is not None:
            snr = np.broadcast_to(np.asarray(self.snr, float), (n_epochs,)).copy()
            if not np.all(np.isfinite(snr) & (snr > 0)):
                raise ValueError(f"snr must be finite and positive, got {self.snr}")
            set_("snr", snr)
        if self.noise not in ("photon", "constant"):
            raise ValueError(f"noise must be 'photon' or 'constant', got {self.noise!r}")
        dsp.doppler_factor(0.0, self.velocity_convention)     # refuses an unknown convention
        if self.as_product and not (product.resampled and product.grid_step_A):
            raise ValueError(f"as_product needs a resampled product with a documented grid "
                             f"step; {self.like!r} is not one")

    @property
    def wavelength_frame(self):
        """'air' if the emulated product is in air, else 'vacuum' (PHOENIX's frame)."""
        return "air" if self.product.wavelength_frame == "air" else "vacuum"


@dataclass(frozen=True, eq=False)
class Truth:
    """
    What a simulation injected.

    Parameters
    ----------
    times : ndarray
        Epochs (days).
    rv_kms : ndarray
        Radial velocities (km/s), (n_epochs, n_components), in
        ``velocity_convention``.
    amplitudes : ndarray
        Line dilution of each component.
    bcv_kms : ndarray
        Barycentric velocity correction per epoch (km/s).
    snr : ndarray or None
        Per-epoch signal-to-noise per native pixel at the continuum; None if
        noiseless.
    noise : str
    seed : int
        SeedSequence entropy that reproduces the noise.
    velocity_convention : str
    like : str
    as_product : bool
    wavelength_frame : str
    """
    times: np.ndarray
    rv_kms: np.ndarray
    amplitudes: np.ndarray
    bcv_kms: np.ndarray
    snr: Optional[np.ndarray]
    noise: str
    seed: int
    velocity_convention: str
    like: str
    as_product: bool
    wavelength_frame: str


def upper_envelope(wave, flux, width_A=ENVELOPE_WIDTH_A, percentile=ENVELOPE_PERCENTILE):
    '''
    A simple robust continuum: the ``percentile``-th percentile of the flux in
    consecutive equal bins of about ``width_A``, linearly interpolated between
    the bins' median wavelengths (and extrapolated beyond the outermost ones).

    A high percentile ignores the absorption lines and the odd spike; in
    line-blanketed (blue, cool-star) regions it is a pseudo-continuum. On a
    sloped continuum one pass reads each bin's continuum near its upper end
    (the slope times ~0.45 bin), so a second pass repeats it on the flux
    divided by the first estimate, where every bin is flat.

    Parameters
    ----------
    wave : ndarray
        Increasing wavelengths (Angstrom).
    flux : ndarray
        Flux at ``wave`` (any units).
    width_A : float
        Bin width (Angstrom).
    percentile : float
        Percentile (0-100) taken as the continuum of each bin.

    Returns
    -------
    ndarray
        The continuum at ``wave``, in the units of ``flux``.
    '''
    wave, flux = np.asarray(wave, float), np.asarray(flux, float)
    n_bins = max(1, round((wave[-1] - wave[0]) / width_A))
    bins = np.minimum(np.floor((wave - wave[0]) / (wave[-1] - wave[0]) * n_bins), n_bins - 1)
    starts = np.flatnonzero(np.diff(bins, prepend=-1))[1:]
    centres = np.array([np.median(w) for w in np.split(wave, starts)])

    def one_pass(f):
        levels = np.array([np.percentile(x, percentile) for x in np.split(f, starts)])
        if levels.size == 1:
            return np.full_like(wave, levels[0])
        return interp1d(centres, levels, fill_value="extrapolate", assume_sorted=True)(wave)

    first = one_pass(flux)
    return first * one_pass(flux / first)


def margin_kms(simulation):
    '''
    Velocity margin (km/s) of the fine grids beyond each order window: the
    largest shift (|rv| + |bcv|), the rotational and instrumental kernels
    (vsini + 3 c / R), and ``MARGIN_PAD_KMS`` for the interpolation.
    '''
    shift = np.max(np.abs(simulation.rv_kms)) + np.max(np.abs(simulation.bcv_kms))
    vsini = max(c.vsini for c in simulation.components)
    fwhm = dsp.C_KMS / simulation.product.instrument.resolving_power
    return float(shift + vsini + 3.0 * fwhm + MARGIN_PAD_KMS)


def component_spectra(simulation, margin=None):
    '''
    Each component's rest-frame model as the simulation builds it (steps 1-4
    of the module docstring): continuum-normalised and broadened, on a fine
    log-lambda grid per order.

    Parameters
    ----------
    simulation : Simulation
    margin : float or None
        Velocity (km/s) the fine grids extend beyond each order window;
        None for `margin_kms`. Widen it to cover a CCF scan of a template
        built from these spectra.

    Returns
    -------
    list of list of (ndarray, ndarray)
        ``[component][order] -> (wave, flux)``: Angstrom (in the simulation's
        wavelength frame) and normalised flux (continuum 1).
    '''
    margin = margin_kms(simulation) if margin is None else float(margin)
    native = simulation.product.instrument.native_step_kms
    fwhm = dsp.C_KMS / simulation.product.instrument.resolving_power
    stretch = math.exp(margin / dsp.C_KMS)
    windows = [(lo / stretch, hi * stretch) for lo, hi in simulation.windows_A]
    # Room for whole envelope bins beyond the outermost grid nodes.
    span = (min(lo for lo, _ in windows) - 2 * ENVELOPE_WIDTH_A,
            max(hi for _, hi in windows) + 2 * ENVELOPE_WIDTH_A)

    out = []
    for comp in simulation.components:
        wave, flux = _intrinsic(comp, span, simulation)
        flux = flux / upper_envelope(wave, flux)
        orders = []
        for lo, hi in windows:
            fine, _ = dsp.log_lambda_grid(lo, hi, native / FINE_FACTOR)
            f = dsp.resample(wave, flux, fine, kind="cubic")
            f = dsp.broaden_rotational(f, native / FINE_FACTOR, comp.vsini, comp.epsilon)
            orders.append((fine, dsp.broaden_gaussian(f, native / FINE_FACTOR, fwhm)))
        out.append(orders)
    return out


def component_templates(simulation, margin=None):
    '''
    The components' rest-frame models (`component_spectra`) as Templates, one
    order per simulated order -- the matched templates for CCF1d / CCF2d.

    Parameters
    ----------
    simulation : Simulation
    margin : float or None
        See `component_spectra`.

    Returns
    -------
    list of Template
        One per component.
    '''
    return [Template(template=Spectrum(wv=[w for w, _ in orders], sp=[f for _, f in orders],
                                       name="simulated"))
            for orders in component_spectra(simulation, margin)]


def _intrinsic(comp, span, simulation):
    '''The component's unnormalised rest-frame spectrum over ``span`` (Angstrom).'''
    if comp.template is not None:
        wave = np.asarray(comp.template.model.wv[0], float)
        flux = np.asarray(comp.template.model.sp[0], float)
        if wave[0] > span[0] or wave[-1] < span[1]:
            raise ValueError(f"the component template ({wave[0]:.1f}-{wave[-1]:.1f} A) does "
                             f"not cover {span[0]:.1f}-{span[1]:.1f} A (the windows plus "
                             "the velocity margin)")
        return wave, flux
    phoenix = Template(temp=comp.teff, log_g=comp.logg, metal=comp.feh, alpha=comp.alpha,
                       min_val=span[0], max_val=span[1],
                       air=simulation.wavelength_frame == "air",
                       allow_download=simulation.allow_download)
    return np.asarray(phoenix.model.wv[0], float), np.asarray(phoenix.model.sp[0], float)


def _archive_grid(wave, step):
    '''The linear archive grid (multiples of ``step``, Angstrom) strictly inside ``wave``.'''
    first = math.floor(wave[0] / step) + 1
    last = math.ceil(wave[-1] / step) - 1
    return np.arange(first, last + 1) * step


def simulate(simulation):
    '''
    Simulated records and their truth (see the module docstring for the model).

    Parameters
    ----------
    simulation : Simulation

    Returns
    -------
    records : list of LoadedSpectrum
        One per epoch, survey 'SIMULATED', ``extras['like']`` the emulated
        product; ``native_step_kms`` is set when ``as_product``.
    truth : Truth
    '''
    sim = simulation
    product = sim.product
    native = product.instrument.native_step_kms
    spectra = component_spectra(sim)
    grids = [dsp.log_lambda_grid(lo, hi, native)[0] for lo, hi in sim.windows_A]
    amplitudes = np.array([c.amplitude for c in sim.components])
    seeds = np.random.SeedSequence(sim.seed)
    rngs = [np.random.default_rng(s) for s in seeds.spawn(sim.times.size)]

    records = []
    for k, rng in enumerate(rngs):
        waves, fluxes = [], []
        for o, wave in enumerate(grids):
            # The pixels' wavelengths once BarycentricCorrection has been applied.
            corrected = dsp.doppler_shift_wave(wave, sim.bcv_kms[k], convention="cz")
            flux = composite_flux(corrected, [comp[o] for comp in spectra], sim.rv_kms[k],
                                  amplitudes, velocity_convention=sim.velocity_convention,
                                  kind="cubic")
            if sim.snr is not None:
                sigma = (np.sqrt(np.clip(flux, 0.0, None)) if sim.noise == "photon"
                         else np.ones_like(flux)) / sim.snr[k]
                flux = flux + sigma * rng.standard_normal(flux.size)
            if sim.as_product:
                archive = _archive_grid(wave, product.grid_step_A)
                wave, flux = archive, resample_flux(wave, flux, archive)
            waves.append(wave)
            fluxes.append(flux)
        records.append(LoadedSpectrum(
            wave=waves, flux=fluxes, time=float(sim.times[k]), time_system=TIME_SYSTEM,
            bcv=float(sim.bcv_kms[k]), specsys=product.specsys,
            wavelength_frame=sim.wavelength_frame,
            snr=math.inf if sim.snr is None else float(sim.snr[k]),
            native_step_kms=[native] * len(waves) if sim.as_product else None,
            survey="SIMULATED",
            extras={"like": sim.like, "epoch": k, "as_product": sim.as_product}))

    truth = Truth(times=sim.times.copy(), rv_kms=sim.rv_kms.copy(), amplitudes=amplitudes,
                  bcv_kms=sim.bcv_kms.copy(), snr=None if sim.snr is None else sim.snr.copy(),
                  noise=sim.noise, seed=seeds.entropy,
                  velocity_convention=sim.velocity_convention, like=sim.like,
                  as_product=sim.as_product, wavelength_frame=sim.wavelength_frame)
    return records, truth


SIMULATED = Instrument(
    name="Simulated",
    resolving_power=None,
    native_step_kms=None,
    notes="Resolving power, native step, wavelength frame and SPECSYS are those of the "
          "product named by Simulation(like=...).",
)

PRODUCTS = (
    Product(
        survey="SIMULATED",
        instrument=SIMULATED,
        description="Spectra simulated at known radial velocities (PHOENIX or user "
                    "templates, broadened, Doppler-shifted, noise added) as the `like` "
                    "product would record them; use Observations(survey='SIMULATED', "
                    "simulation=Simulation(...)). The injected values are on obs.truth.",
        resampled=False,
        grid="log-lambda at the like instrument's native step (pixel centres), or the "
             "like product's archive grid with as_product=True",
        wavelength_frame=None,
        specsys=None,
        time_system=TIME_SYSTEM,
        multi_order=True,
        recommended=Recommended(),
        verified=False,
        load=simulate,
        synthetic=True,
    ),
)
