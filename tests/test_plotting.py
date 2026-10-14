"""The SPARTA figure style (``sparta.plotting``) and the plot methods using it.

Data: black, mid-centred steps. Model: red line. One extra series: blue line.
Maps: grey-scale images, white at the lowest value and black at the peak, with
a grey colour bar and no other colour. Rendered with the non-interactive Agg
backend.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from astropy import units as u  # noqa: E402
from matplotlib.collections import QuadMesh  # noqa: E402
from matplotlib.colors import SymLogNorm, to_rgba  # noqa: E402
from matplotlib.contour import ContourSet  # noqa: E402
from matplotlib.image import AxesImage  # noqa: E402

import unicor  # noqa: E402
from sparta import plotting as sp_plot  # noqa: E402
from sparta.ccf1d import CCF1d  # noqa: E402
from sparta.ccf2d import CCF2d  # noqa: E402
from sparta.periodogram import PeriodicityDetector  # noqa: E402
from sparta.spectrum import Spectrum  # noqa: E402
from sparta.timeseries import TimeSeries  # noqa: E402
from sparta.tiravel import TIRAVEL  # noqa: E402

BLACK, RED, BLUE = to_rgba("k"), to_rgba(sp_plot.MODEL_COLOR), to_rgba(sp_plot.EXTRA_COLOR)


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


@pytest.fixture(autouse=True)
def _close_all():
    yield
    plt.close("all")


def _rgba(artist):
    return to_rgba(artist.get_color())


def _meshes(fig):
    # A colour bar draws its ramp as a QuadMesh too; count the maps only.
    return [c for a in fig.axes if a.get_label() != "<colorbar>"
            for c in a.get_children() if isinstance(c, QuadMesh)]


def _assert_grey_map_peak_black(mesh, z):
    # One grey ramp, white at the bottom of the scale and black at the peak.
    assert mesh.get_cmap().name == sp_plot.MAP_CMAP == "Greys"
    rgba = mesh.to_rgba(np.asarray(mesh.get_array()))
    assert np.allclose(rgba[..., 0], rgba[..., 1]) and np.allclose(rgba[..., 1], rgba[..., 2])
    k = np.unravel_index(np.nanargmax(z), np.shape(z))
    assert np.allclose(rgba[k][:3], 0.0, atol=1e-3)
    assert np.allclose(mesh.get_cmap()(0.0)[:3], 1.0, atol=0.03)


def _assert_only_a_grey_image(fig):
    (mesh,) = _meshes(fig)
    for a in fig.axes:
        for child in a.get_children():
            assert not isinstance(child, (AxesImage, ContourSet)), child
    return mesh


# =============================================================================
# The helpers
# =============================================================================
def test_plot_data_is_black_mid_centred_steps(ax):
    (line,) = sp_plot.plot_data(ax, np.arange(5.0), np.ones(5))
    assert line.get_drawstyle() == "steps-mid"
    assert _rgba(line) == BLACK
    assert line.get_linewidth() < 1.0


def test_model_is_a_red_line_and_extra_a_blue_line(ax):
    (model,) = sp_plot.plot_model(ax, np.arange(5.0), np.ones(5))
    (extra,) = sp_plot.plot_extra(ax, np.arange(5.0), np.ones(5))
    (curve,) = sp_plot.plot_curve(ax, np.arange(5.0), np.ones(5))
    assert (_rgba(model), _rgba(extra), _rgba(curve)) == (RED, BLUE, BLACK)
    assert {model.get_drawstyle(), extra.get_drawstyle(), curve.get_drawstyle()} == {"default"}


def test_points_are_black_markers_with_error_bars(ax):
    container = sp_plot.plot_points(ax, np.arange(4.0), np.zeros(4), 0.1 * np.ones(4))
    marker_line, _, (bars,) = container.lines
    assert _rgba(marker_line) == BLACK and marker_line.get_linestyle() == "None"
    assert marker_line.get_marker() == "o"
    assert np.allclose(bars.get_colors(), BLACK)


def test_fit_and_truth_markers(ax):
    (fit,) = sp_plot.mark_fit(ax, 1.0, 2.0)
    (truth,) = sp_plot.mark_truth(ax, 1.0, 2.0)
    assert (fit.get_marker(), _rgba(fit)) == ("+", RED)
    assert (truth.get_marker(), _rgba(truth)) == ("x", BLUE)
    # A white edge keeps them visible on the black peak of a map.
    for marker in (fit, truth):
        (edge,) = marker.get_path_effects()
        assert to_rgba(edge._gc["foreground"]) == to_rgba("white")


def test_map_image_is_white_at_zero_and_black_at_the_peak(ax):
    # The grey is anchored to the map's own scale (user request): zero is
    # white, the peak black, with a power-law contrast stretch.
    from matplotlib.colors import PowerNorm
    x, y = np.linspace(-5, 5, 41), np.linspace(-4, 4, 31)
    r = 0.9 * np.exp(-0.1 * (x[None, :] ** 2 + y[:, None] ** 2)) + 0.05
    mesh = sp_plot.map_image(ax, x, y, r)
    assert isinstance(mesh.norm, PowerNorm)
    assert (mesh.norm.vmin, mesh.norm.vmax) == (0.0, r.max())
    assert mesh.norm.gamma == sp_plot.MAP_GAMMA
    assert mesh.norm(0.0) == 0.0 and mesh.norm(r.max()) == 1.0
    assert mesh.norm(-0.3) == 0.0                 # below zero is white
    assert not isinstance(mesh.norm, SymLogNorm)
    _assert_grey_map_peak_black(mesh, r)
    _assert_only_a_grey_image(ax.figure)


def test_map_image_of_a_log_likelihood_is_symlog_delta_lnl(ax):
    x, y = np.linspace(-5, 5, 41), np.linspace(-4, 4, 31)
    lnl = 7.0 - 3000.0 * (x[None, :] ** 2 + y[:, None] ** 2)    # any offset, deep
    mesh = sp_plot.map_image(ax, x, y, lnl, loglike=True)
    norm = mesh.norm
    assert isinstance(norm, SymLogNorm) and norm.linthresh == sp_plot.SIGMA_LEVELS[-1]
    assert norm.vmax == 0.0 and norm.vmin == pytest.approx(-(lnl.max() - lnl.min()))
    np.testing.assert_allclose(mesh.get_array(), lnl - lnl.max())
    _assert_grey_map_peak_black(mesh, lnl)


def test_dlnl_norm_is_linear_to_three_sigma_and_logarithmic_beyond():
    norm = sp_plot.dlnl_norm(1e5)
    assert (norm.vmin, norm.vmax, norm.linthresh) == (-1e5, 0.0, 4.5)
    # Equal shares of the grey scale per decade beyond the linear part.
    steps = np.diff(norm(np.array([-1e5, -1e4, -1e3])))
    assert steps[0] == pytest.approx(steps[1])
    assert sp_plot.dlnl_norm(0.1).vmin == -4.5


def test_map_colorbar_is_thin_and_labelled(ax):
    mesh = sp_plot.map_image(ax, np.arange(3.0), np.arange(4.0), np.arange(12.0).reshape(4, 3))
    cbar = sp_plot.map_colorbar(mesh, "$R$")
    assert cbar.ax.get_ylabel() == "$R$"
    assert cbar.mappable is mesh


def test_apply_style_sets_the_rcparams():
    with matplotlib.rc_context():
        sp_plot.apply_style()
        assert plt.rcParams["legend.frameon"] is False
        assert plt.rcParams["xtick.direction"] == "in"


# =============================================================================
# The plot methods
# =============================================================================
def _todcor_result(output):
    v1, v2 = np.arange(-50.0, 51.0, 2.0), np.arange(-60.0, 61.0, 2.0)
    chi2 = 0.04 * ((v1[:, None] - 10.0) ** 2 + 2 * (v2[None, :] + 20.0) ** 2)
    sd = 2.0 * chi2.max()
    fit = CCF2d()
    fit.Corr2d = {"v1_grid": v1, "v2_grid": v2, "logL_profile": 123.0 - 0.5 * chi2,
                  "R": np.sqrt(1.0 - chi2 / sd), "RV1": 10.0, "RV2": -20.0,
                  "output": output}
    return fit


@pytest.mark.parametrize("output", ["ccf", "loglike"])
@pytest.mark.parametrize("marginals", [True, False])
def test_ccf2d_plot_is_a_grey_image_peaked_black(output, marginals):
    fit = _todcor_result(output)
    fig = fit.plot(marginals=marginals)
    mesh = _assert_only_a_grey_image(fig)
    res = fit.Corr2d
    if output == "ccf":
        np.testing.assert_allclose(mesh.get_array(), res["R"].T)
        assert not isinstance(mesh.norm, SymLogNorm)
    else:
        dlnl = res["logL_profile"] - res["logL_profile"].max()
        np.testing.assert_allclose(mesh.get_array(), dlnl.T)
        assert isinstance(mesh.norm, SymLogNorm)
    _assert_grey_map_peak_black(mesh, mesh.get_array())
    colorbars = [a for a in fig.axes if a.get_label() == "<colorbar>"]
    assert len(colorbars) == 1
    label = colorbars[0].get_ylabel()
    assert label == ("$R$" if output == "ccf" else sp_plot.DLNL_LABEL)
    fits = [ln for a in fig.axes for ln in a.get_lines() if ln.get_marker() == "+"]
    assert len(fits) == 1 and _rgba(fits[0]) == RED
    assert (fits[0].get_xdata()[0], fits[0].get_ydata()[0]) == (10.0, -20.0)
    assert len(fig.axes) == (5 if marginals else 2)      # + the colour bar
    profiles = [ln for a in fig.axes for ln in a.get_lines() if len(ln.get_xdata()) > 2]
    assert len(profiles) == (2 if marginals else 0)
    assert all(_rgba(ln) == BLACK for ln in profiles)
    if marginals:
        top = max(profiles, key=lambda ln: len(ln.get_xdata()) == res["v1_grid"].size)
        want = res["R"][:, 20] if output == "ccf" else dlnl[:, 20]
        np.testing.assert_allclose(top.get_ydata(), want)


def test_ccf1d_plot_draws_black_ccfs_and_a_red_rv():
    vel = np.linspace(-20, 20, 81)
    corr = np.exp(-0.5 * ((vel - 3.0) / 4.0) ** 2)
    ccf = CCF1d()
    ccf.n_ord = 2
    ccf.Corr = {"vel": vel * u.km / u.s, "corr": np.vstack([corr, 0.9 * corr]),
                "output": "ccf"}
    ccf.CorrCombined = {"vel": vel * u.km / u.s, "corr": corr, "RV": 3.0, "eRV": 0.2}
    (ax,) = ccf.plot().axes
    curves = [ln for ln in ax.get_lines() if len(ln.get_xdata()) == vel.size]
    assert len(curves) == 3 and all(_rgba(ln)[:3] == BLACK[:3] for ln in curves)
    assert all(ln.get_drawstyle() == "default" for ln in curves)
    rv_lines = [ln for ln in ax.get_lines() if len(ln.get_xdata()) == 2]
    assert len(rv_lines) == 1 and _rgba(rv_lines[0]) == RED


def test_ccf1d_plot_without_combine_draws_the_orders_only():
    ccf = CCF1d()
    ccf.n_ord = 1
    ccf.Corr = {"vel": np.linspace(-5, 5, 11) * u.km / u.s, "corr": np.ones((1, 11)),
                "output": "ccf"}
    (ax,) = ccf.plot().axes
    assert len(ax.get_lines()) == 1


def test_ccf1d_plot_in_loglike_mode_draws_the_log_likelihoods():
    vel = np.linspace(-20, 20, 81)
    corr = np.vstack([0.9, 0.8])[:, :1] * np.exp(-0.5 * ((vel - 3.0) / 4.0) ** 2)
    n_eff = np.array([300.0, 500.0])
    lnl = unicor.loglike_from_ccf(corr, n_eff)
    ccf = CCF1d()
    ccf.n_ord = 2
    ccf.Corr = {"vel": vel * u.km / u.s, "corr": corr, "loglike": lnl, "output": "loglike"}
    ccf.CorrCombined = {"vel": vel * u.km / u.s, "corr": corr[0], "loglike": lnl.sum(axis=0),
                        "RV": 3.0, "eRV": 0.2}
    (ax,) = ccf.plot().axes
    curves = [ln.get_ydata() for ln in ax.get_lines() if len(ln.get_xdata()) == vel.size]
    np.testing.assert_allclose(curves, [lnl[0], lnl[1], lnl.sum(axis=0)])
    assert "ln L" in ax.get_ylabel()


def test_velocities_are_black_points_and_fold_to_phase():
    ts = TimeSeries(times=[0.0, 1.3, 2.9, 4.4], vals=[1.0, -1.0, 0.5, 0.2],
                    errors=[0.1, 0.1, 0.2, 0.1], fold_period=2.0)
    fig = ts.plot_velocities()
    assert len(fig.axes) == 2
    for a in fig.axes:
        (container,) = a.containers
        assert _rgba(container.lines[0]) == BLACK
    np.testing.assert_allclose(fig.axes[1].containers[0].lines[0].get_xdata(),
                               [0.0, 0.65, 0.45, 0.2])


def test_periodogram_is_a_black_line_with_blue_known_periods():
    times = np.linspace(0, 40, 30)
    ts = TimeSeries(times=list(times), vals=list(np.sin(2 * np.pi * times / 7.3)))
    det = PeriodicityDetector(time_series=ts, freq_range=(0.05, 0.5),
                              periodogram_grid_resolution=200)
    det.period_truth = [7.3]
    det.calc_gls()
    fig = det.plot_periodograms(plot_vals=True)
    gls_ax, vals_ax = fig.axes
    (power,) = [ln for ln in gls_ax.get_lines() if len(ln.get_xdata()) > 2]
    assert _rgba(power) == BLACK and power.get_drawstyle() == "default"
    (marker,) = [ln for ln in gls_ax.get_lines() if len(ln.get_xdata()) == 2]
    assert _rgba(marker) == BLUE and marker.get_xdata()[0] == pytest.approx(1 / 7.3)
    assert _rgba(vals_ax.containers[0].lines[0]) == BLACK


def test_tiravel_template_is_red_over_black_stepped_epochs():
    wave, _ = unicor.log_lambda_grid(5000.0, 5050.0, 1.0)
    centres = np.linspace(5005, 5045, 9)
    velocities = [-30.0, 0.0, 25.0, 60.0]

    def star(w):
        return 1 - 0.5 * np.exp(-0.5 * ((w[:, None] - centres) / 0.08) ** 2).sum(axis=1)

    spectra = [Spectrum(wv=[wave.copy()], sp=[star(wave / unicor.doppler_factor(v))],
                        bjd=float(i), name="sim") for i, v in enumerate(velocities)]
    tir = TIRAVEL().build_template(spectra, velocities, keep_aligned=True)
    (ax,) = tir.plot_template().axes
    epochs = [ln for ln in ax.get_lines() if ln.get_drawstyle() == "steps-mid"]
    (template,) = [ln for ln in ax.get_lines() if ln.get_drawstyle() == "default"]
    assert len(epochs) == len(velocities) and all(_rgba(ln)[:3] == BLACK[:3] for ln in epochs)
    assert _rgba(template) == RED
    tir.plot_weights()


def test_tiravel_template_is_drawn_on_the_scale_of_the_epochs():
    # The stored template is a co-add of K unit-norm rows with unit-norm
    # weights, ~sqrt(K) times an epoch; drawn as is, the epochs would look
    # like a shrunken copy of it. Nine noisy epochs: sqrt(K) = 3.
    rng = np.random.default_rng(3)
    wave, _ = unicor.log_lambda_grid(5000.0, 5050.0, 1.0)
    centres = np.linspace(5005, 5045, 9)
    velocities = np.linspace(-40.0, 40.0, 9)

    def star(w):
        return 1 - 0.5 * np.exp(-0.5 * ((w[:, None] - centres) / 0.08) ** 2).sum(axis=1)

    spectra = [Spectrum(wv=[wave.copy()],
                        sp=[star(wave / unicor.doppler_factor(v)) + rng.normal(0, 0.01, wave.size)],
                        name="sim") for v in velocities]
    tir = TIRAVEL().build_template(spectra, velocities)
    (ax,) = tir.plot_template().axes
    epochs = [ln.get_ydata() for ln in ax.get_lines() if ln.get_drawstyle() == "steps-mid"]
    (template,) = [ln.get_ydata() for ln in ax.get_lines() if ln.get_drawstyle() == "default"]
    epoch_amplitude = np.median([np.ptp(y) for y in epochs])
    assert np.ptp(template) == pytest.approx(epoch_amplitude, rel=0.1)
    assert np.max(np.abs(template - np.mean(epochs, axis=0))) < 0.1 * epoch_amplitude
    assert "mean-subtracted" in ax.get_ylabel()
