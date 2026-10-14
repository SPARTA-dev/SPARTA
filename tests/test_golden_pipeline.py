"""Golden master for the analysis pipeline: RVs, TIRAVEL and periodograms.

Pins the numbers the user-facing workflows produce, so the API/layout refactor
that follows can be proven behaviour-preserving:

* a seeded simulated SB1 through ``Observations.calc_rv_against_template``
  (UNICOR): RV, eRV and CCF peak per epoch (km/s);
* a seeded simulated SB2 through ``Observations.calc_rv_against_two_templates``
  (TODCOR): RV1, RV2, eRV1, eRV2 (km/s) and the amplitudes a1, a2;
* TIRAVEL on the bundled TOI-193 HARPS spectra, as in
  ``examples/TIRAVEL_Quickstart.ipynb``: the library-template RVs, rho and the
  co-addition weights;
* periodograms (frequencies in 1/day): GLS, biased and unbiased PDC and the
  error-aware PDC on a seeded scalar series; GLS, USuRPER and the shift/shape
  partial periodograms on the simulated SB1 spectra.

Values, not objects: the golden file stores plain floats rounded to 12
significant digits, compared at 1e-9 relative (``RTOL``; ``ATOL`` for the
periodogram powers that cross zero). PHOENIX models come from the bundled cache
only (``allow_download=False``). The BIS is deliberately not pinned.

Regenerate only on purpose, after confirming a change in the reference is
intended: ``python tests/test_golden_pipeline.py --write``.
"""
import json
import os
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if __name__ == "__main__":
    # The checkout, as pytest's pythonpath (pyproject.toml) has it -- not
    # whatever sparta happens to be installed.
    sys.path[:0] = [str(ROOT / p) for p in ("src", "packages/unicor/src", "packages/todcor/src")]

from sparta.Auxil.TimeSeries import TimeSeries
from sparta.Observations import Observations
from sparta.instruments import Simulation
from sparta.instruments.simulated import component_templates
from sparta.UNICOR.Spectrum import Spectrum
from sparta.UNICOR.Template import Template

TOI193 = ROOT / "examples" / "data" / "TOI193"
GOLDEN_PATH = pathlib.Path(__file__).resolve().parent / "golden" / "pipeline.json"

RTOL = 1e-9
ATOL = 1e-12
SIG_DIGITS = 12

SUN_LIKE = dict(teff=5800, logg=3.5, feh=0.5, vsini=3.0)       # bundled lte05800-3.50+0.5
K_DWARF = dict(teff=4900, logg=4.5, vsini=2.0)                  # bundled lte04900-4.50-0.0

# Simulated SB1: P = 4.3 d, K = 2 km/s around 12 km/s, 16 epochs over 40 d,
# two orders, rebinned onto the HARPS archive grid (seeds the native step).
SB1_PERIOD, SB1_K, SB1_GAMMA, SB1_EPOCHS = 4.3, 2.0, 12.0, 16
# Scalar series: P = 5 d unit sine plus Gaussian noise (sigma 0.3), 30 epochs over 100 d.
SCALAR_PERIOD, SCALAR_EPOCHS = 5.0, 30
FREQ_RANGE, GRID_RESOLUTION = (0.02, 1.0), 40     # 1/day; grid points per 1/day


def _rounded(values):
    """Plain Python floats at ``SIG_DIGITS`` significant digits."""
    return [float(f"{float(v):.{SIG_DIGITS}g}") for v in np.ravel(values)]


def _golden():
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def _assert_close(got, want, atol=0.0):
    np.testing.assert_allclose(np.asarray(got, float), np.asarray(want, float),
                               rtol=RTOL, atol=atol)


# --- recorders: the pipeline under test, reduced to plain values -----------

def _sb1_observations():
    rng = np.random.default_rng(20261014)
    times = np.sort(rng.uniform(0.0, 40.0, SB1_EPOCHS))
    rv = SB1_GAMMA + SB1_K * np.sin(2 * np.pi * times / SB1_PERIOD)
    sim = Simulation(like="HARPS", components=[SUN_LIKE], times=times, rv_kms=rv,
                     windows_A=[(5000.0, 5060.0), (5500.0, 5560.0)], snr=80.0,
                     noise="constant", seed=11, as_product=True)
    return Observations(survey="SIMULATED", simulation=sim), component_templates(sim)[0]


def _record_sb1_and_spectral_periodograms():
    """UNICOR RVs of the simulated SB1, then its single-order periodograms."""
    obs, template = _sb1_observations()
    obs.calc_rv_against_template(template, dv=0.1, VelBound=60)
    sb1 = {"rv": _rounded(obs.vels), "erv": _rounded(obs.evels),
           "peak": _rounded(obs.ccf_peaks)}

    # USuRPER and the partial periodograms take one order; the partials need
    # one RV per epoch, here the measured ones.
    obs.SelectOrders([0], remove=False)
    obs.time_series.calculated_vrad_list = list(obs.vels)
    obs.initialize_periodicity_detector(freq_range=FREQ_RANGE,
                                        periodogram_grid_resolution=GRID_RESOLUTION)
    detector = obs.periodicity_detector
    detector.calc_GLS()
    detector.calc_USURPER(calc_biased_flag=True, calc_unbiased_flag=True)
    detector.calc_partial_periodogram(partial_type="shift")
    detector.calc_partial_periodogram(partial_type="shape")
    spectral = {"frequency": _rounded(detector.results_frequency["USURPER"])}
    spectral.update({name: _rounded(power) for name, power in detector.results_power.items()})
    return sb1, spectral


def _record_sb2():
    rng = np.random.default_rng(20261015)
    rv = np.column_stack([rng.uniform(-30.0, 30.0, 4), rng.uniform(-30.0, 30.0, 4)])
    sim = Simulation(like="HARPS", windows_A=[(5000.0, 5200.0)], times=np.arange(4.0),
                     rv_kms=rv, snr=50.0, noise="constant", seed=12,
                     components=[dict(SUN_LIKE, vsini=4.0, amplitude=0.7),
                                 dict(K_DWARF, amplitude=0.3)])
    obs = Observations(survey="SIMULATED", simulation=sim)
    template1, template2 = component_templates(sim)
    obs.calc_rv_against_two_templates(template1, template2, dv=0.3, VelBound=40,
                                      continuum=1.0, alpha=0.3)
    return {"rv1": _rounded(obs.vels1), "rv2": _rounded(obs.vels2),
            "erv1": _rounded(obs.evels1), "erv2": _rounded(obs.evels2),
            "a1": _rounded(obs.amps1), "a2": _rounded(obs.amps2)}


def _record_scalar_periodograms():
    rng = np.random.default_rng(5)
    times = np.sort(rng.uniform(0.0, 100.0, SCALAR_EPOCHS))
    vals = np.sin(2 * np.pi * times / SCALAR_PERIOD) + 0.3 * rng.normal(size=SCALAR_EPOCHS)
    errors = rng.uniform(0.1, 0.4, SCALAR_EPOCHS)

    def detector():
        series = TimeSeries(times=list(times), vals=list(vals), errors=list(errors))
        obs = Observations(time_series=series)
        obs.initialize_periodicity_detector(freq_range=FREQ_RANGE,
                                            periodogram_grid_resolution=GRID_RESOLUTION)
        return obs.periodicity_detector

    plain = detector()
    plain.calc_GLS()
    plain.calc_PDC(calc_biased_flag=True, calc_unbiased_flag=True)
    aware = detector()
    aware.calc_PDC(calc_biased_flag=True, calc_unbiased_flag=True, errors=True)
    return {"frequency": _rounded(plain.results_frequency["PDC_unbiased"]),
            "GLS": _rounded(plain.results_power["GLS"]),
            "PDC_biased": _rounded(plain.results_power["PDC_biased"]),
            "PDC_unbiased": _rounded(plain.results_power["PDC_unbiased"]),
            "PDC_errors_biased": _rounded(aware.results_power["PDC_biased"]),
            "PDC_errors_unbiased": _rounded(aware.results_power["PDC_unbiased"])}


def _tiravel_files():
    """Every second bundled TOI-193 file (the quickstart's sample_rate=2), in
    name order -- os.walk order is filesystem-dependent, so it is not used."""
    return sorted(TOI193.glob("*.fits"))[::2]


def _record_tiravel(directory):
    """TIRAVEL_Quickstart.ipynb: library-template RVs, then the TIRAVEL template."""
    obs = Observations(survey="HARPS", target_visits_lib=str(directory),
                       min_wv=4990, max_wv=5090)
    for spec in obs.time_series.vals:
        spec.SpecPreProccess()

    lib = Template(temp=5500, log_g=4.5, metal=0, alpha=0, min_val=4980, max_val=5100,
                   air=False, allow_download=False)
    lib = Template(template=Spectrum(wv=lib.model.wv, sp=lib.model.sp).InterpolateSpectrum())
    lib.RotationalBroadening(epsilon=0.5, vsini=1.5)
    lib = Template(template=Spectrum(wv=lib.model.wv, sp=lib.model.sp).SpecPreProccess())
    lib = Template(template=Spectrum(wv=lib.model.wv, sp=lib.model.sp)
                   .InterpolateSpectrum(delta=0.5))
    lib.GaussianBroadening(resolution=115_000)
    template_library = Template(template=Spectrum(wv=lib.model.wv, sp=lib.model.sp)
                                .SpecPreProccess())

    rv_library = obs.calc_rv_against_template(template_library, dv=0.1, VelBound=[-117, 117],
                                              combine_ccfs=False).vels
    names = [os.path.basename(p) for p in obs.file_list]
    # The reference epoch fixes the common grid; take the first by name.
    obs.build_tiravel_template(rv_library, reference=names.index(min(names)))
    return {"rho": _rounded([obs.tiravel.rho])[0],
            "rv_library": dict(zip(names, _rounded(rv_library))),
            "weights": dict(zip(names, _rounded(obs.tiravel.weights)))}


def _tiravel_dir(directory):
    for path in _tiravel_files():
        (directory / path.name).symlink_to(path)
    return directory


# --- tests ------------------------------------------------------------------

@pytest.fixture(scope="module")
def sb1_and_spectral():
    return _record_sb1_and_spectral_periodograms()


def test_sb1_unicor_rvs_match_golden(sb1_and_spectral):
    got, want = sb1_and_spectral[0], _golden()["sb1"]
    assert sorted(got) == sorted(want)
    for key in want:
        _assert_close(got[key], want[key])


def test_sb2_todcor_rvs_and_amplitudes_match_golden():
    got, want = _record_sb2(), _golden()["sb2"]
    assert sorted(got) == sorted(want)
    for key in want:
        _assert_close(got[key], want[key])


def test_scalar_periodograms_match_golden():
    got, want = _record_scalar_periodograms(), _golden()["periodograms"]["scalar"]
    assert sorted(got) == sorted(want)
    for key in want:
        _assert_close(got[key], want[key], atol=ATOL)


def test_spectral_periodograms_match_golden(sb1_and_spectral):
    got, want = sb1_and_spectral[1], _golden()["periodograms"]["spectral"]
    assert sorted(got) == sorted(want)
    for key in want:
        _assert_close(got[key], want[key], atol=ATOL)


@pytest.mark.skipif(not _tiravel_files(), reason="bundled TOI-193 data not present")
def test_tiravel_on_toi193_matches_golden(tmp_path):
    got, want = _record_tiravel(_tiravel_dir(tmp_path)), _golden()["tiravel"]
    _assert_close(got["rho"], want["rho"])
    for key in ("rv_library", "weights"):
        assert sorted(got[key]) == sorted(want[key])
        _assert_close([got[key][n] for n in sorted(want[key])],
                      [want[key][n] for n in sorted(want[key])])


def _write_golden():
    import tempfile
    sb1, spectral = _record_sb1_and_spectral_periodograms()
    with tempfile.TemporaryDirectory() as tmp:
        tiravel = _record_tiravel(_tiravel_dir(pathlib.Path(tmp)))
    golden = {
        "scheme": f"plain floats at {SIG_DIGITS} significant digits; "
                  "velocities km/s, frequencies 1/day",
        "sb1": sb1,
        "sb2": _record_sb2(),
        "tiravel": tiravel,
        "periodograms": {"scalar": _record_scalar_periodograms(), "spectral": spectral},
    }
    with open(GOLDEN_PATH, "w") as f:
        json.dump(golden, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        raise SystemExit("usage: python tests/test_golden_pipeline.py --write")
    _write_golden()
