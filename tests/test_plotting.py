"""The SPARTA figure style (``sparta.plotting``) and the plot methods using it.

Data: black, mid-centred steps. Model: red line. One extra series: blue line.
Maps: black unfilled contours, no colour map, no colour bar. Rendered with
the non-interactive Agg backend.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from astropy import units as u  # noqa: E402
from matplotlib.collections import QuadMesh  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
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


def _contour_sets(fig):
    return [c for a in fig.axes for c in a.get_children() if isinstance(c, ContourSet)]


def _assert_no_colour_map(fig):
    for a in fig.axes:
        for child in a.get_children():
            assert not isinstance(child, (QuadMesh, AxesImage)), child
            assert not (isinstance(child, ContourSet) and child.filled), "filled contours"
    assert all(a.get_label() != "<colorbar>" for a in fig.axes), "a colour bar"


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


def test_dlnl_levels_are_dense_at_the_peak_and_bounded_by_the_depth():
    levels = sp_plot.dlnl_levels(5000.0)
    np.testing.assert_allclose(levels[-3:], [-4.5, -2.0, -0.5])
    np.testing.assert_allclose(levels[:-3], [-3000, -1000, -300, -100, -30, -10])
    assert np.all(np.diff(levels) > 0)
    np.testing.assert_allclose(sp_plot.dlnl_levels(-3.0), [-2.0, -0.5])
    np.testing.assert_allclose(sp_plot.dlnl_levels(0.1), [-0.5])


def test_contour_map_is_black_unfilled_with_thick_sigma_levels(ax):
    x, y = np.linspace(-5, 5, 41), np.linspace(-4, 4, 31)
    lnl = 7.0 - 3.0 * (x[None, :] ** 2 + y[:, None] ** 2)     # any offset
    cs = sp_plot.contour_map(ax, x, y, lnl)
    assert not cs.filled
    np.testing.assert_allclose(cs.levels, 7.0 + sp_plot.dlnl_levels(lnl.max() - lnl.min()))
    assert np.allclose(cs.get_edgecolor(), BLACK)
    widths = np.asarray(cs.get_linewidth())
    np.testing.assert_allclose(widths, np.where(cs.levels >= 7.0 - 4.5,
                                                sp_plot.CONTOUR_INNER_LW, sp_plot.CONTOUR_LW))
    # matplotlib dashes negative levels of a monochrome map unless told not to.
    assert all(ls[1] is None for ls in cs.get_linestyle())
    _assert_no_colour_map(ax.figure)


def test_legend_entries_name_the_sigma_levels():
    handles, labels = sp_plot.dlnl_legend_entries(sp_plot.dlnl_levels(1000.0))
    assert len(handles) == 2
    assert "-0.5, -2, -4.5" in labels[0] and r"\sigma" in labels[0]
    assert labels[1].startswith("$-10, -30, -100") and r"\ldots" in labels[1]


def test_apply_style_sets_the_rcparams():
    with matplotlib.rc_context():
        sp_plot.apply_style()
        assert plt.rcParams["legend.frameon"] is False
        assert plt.rcParams["xtick.direction"] == "in"


# =============================================================================
# The plot methods
# =============================================================================
def _todcor_result():
    v1, v2 = np.arange(-50.0, 51.0, 2.0), np.arange(-60.0, 61.0, 2.0)
    logl = 123.0 - 0.02 * ((v1[:, None] - 10.0) ** 2 + 2 * (v2[None, :] + 20.0) ** 2)
    fit = CCF2d()
    fit.Corr2d = {"v1_grid": v1, "v2_grid": v2, "logL_profile": logl,
                  "RV1": 10.0, "RV2": -20.0}
    return fit


@pytest.mark.parametrize("marginals", [True, False])
def test_ccf2d_plot_is_an_unfilled_black_contour_map(marginals):
    fig = _todcor_result().plot(marginals=marginals)
    _assert_no_colour_map(fig)
    (cs,) = _contour_sets(fig)
    assert np.allclose(cs.get_edgecolor(), BLACK)
    assert cs.levels.max() == pytest.approx(-0.5)
    fits = [ln for a in fig.axes for ln in a.get_lines() if ln.get_marker() == "+"]
    assert len(fits) == 1 and _rgba(fits[0]) == RED
    assert (fits[0].get_xdata()[0], fits[0].get_ydata()[0]) == (10.0, -20.0)
    assert len(fig.axes) == (4 if marginals else 1)
    profiles = [ln for a in fig.axes for ln in a.get_lines() if len(ln.get_xdata()) > 2]
    assert len(profiles) == (2 if marginals else 0)
    assert all(_rgba(ln) == BLACK for ln in profiles)


def test_ccf2d_plot_takes_explicit_levels():
    fig = _todcor_result().plot(levels=[-1.0, -8.0])
    (cs,) = _contour_sets(fig)
    np.testing.assert_allclose(cs.levels, [-8.0, -1.0])


def test_ccf1d_plot_draws_black_ccfs_and_a_red_rv():
    vel = np.linspace(-20, 20, 81)
    corr = np.exp(-0.5 * ((vel - 3.0) / 4.0) ** 2)
    ccf = CCF1d()
    ccf.n_ord = 2
    ccf.Corr = {"vel": vel * u.km / u.s, "corr": np.vstack([corr, 0.9 * corr])}
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
    ccf.Corr = {"vel": np.linspace(-5, 5, 11) * u.km / u.s, "corr": np.ones((1, 11))}
    (ax,) = ccf.plot().axes
    assert len(ax.get_lines()) == 1


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
