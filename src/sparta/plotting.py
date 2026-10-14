"""The SPARTA figure style: data, model, one extra series, unfilled maps.

Every figure in SPARTA (the plot methods and the examples) draws each kind of
curve the same way:

=========================  ============================================  ===============
what                       how                                           helper
=========================  ============================================  ===============
data (spectra)             black, thin, mid-centred steps                ``plot_data``
data at scattered epochs   black points with error bars (RVs, folds)     ``plot_points``
model / fit                red line                                      ``plot_model``
one extra series           blue line (truth, 2nd component, comparison)  ``plot_extra``
derived curves             black line (CCFs, periodograms, profiles)     ``plot_curve``
further series             grey lines / linestyles, never more colours   ``GREY``
2-D maps                   black unfilled contours, no colour map        ``contour_map``
best fit / truth on a map  red ``+`` / blue ``x``                         ``mark_fit`` / ``mark_truth``
=========================  ============================================  ===============

Steps are for data only: a pixel holds the flux over its whole width, which a
step drawn about the pixel centre shows and a line does not. Scattered epochs
have no width to show, so they are points. A CCF, a periodogram or a profile
likelihood is computed on a grid of trial values chosen by the analysis and
is drawn as a line.

Red marks what was fitted or detected (a model curve, an RV, a best-fit
point); blue marks the truth (a true RV or period, an injected (v1, v2)).

Log-likelihood maps are contoured in Delta lnL below the maximum
(``dlnl_levels``): -0.5, -2, -4.5 (1, 2, 3 sigma for one parameter, drawn
thicker, ``dlnl_linewidths``), then -10, -30, -100, ... -- dense at the peak,
sparse in the far field. ``dlnl_legend_entries`` lists the levels in a legend.

``apply_style`` sets matching rcParams (font sizes, ticks, layout) for a
session; the plot methods do not call it, so they never change global state.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA_COLOR = 'k'
MODEL_COLOR = 'tab:red'
EXTRA_COLOR = 'tab:blue'
GREY = '0.6'

DATA_LW = 0.75
CURVE_LW = 1.0
MODEL_LW = 1.2
EXTRA_LW = 1.0
CONTOUR_LW = 0.5
CONTOUR_INNER_LW = 1.1

# Delta lnL = -k^2/2 is the k-sigma bound of one Gaussian parameter.
SIGMA_LEVELS = (0.5, 2.0, 4.5)

STYLE = {
    'font.size': 11.0, 'axes.labelsize': 11.0, 'axes.titlesize': 11.0,
    'xtick.labelsize': 10.0, 'ytick.labelsize': 10.0, 'legend.fontsize': 9.0,
    'legend.frameon': False,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'lines.linewidth': CURVE_LW,
    'figure.figsize': (8.0, 3.2), 'figure.dpi': 100,
    'figure.constrained_layout.use': True,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': False,
}


def apply_style():
    '''
    Set the SPARTA rcParams (``STYLE``) for the rest of the session.

    Font sizes (11 pt labels, 10 pt ticks, 9 pt legends), inward ticks on all
    four sides, constrained layout, an 8 x 3.2 inch default figure and 300 dpi
    tight saving. Colours and line widths are set by the plotting helpers,
    not here.

    Returns
    -------
    None
    '''
    plt.rcParams.update(STYLE)


def plot_data(ax, x, y, **kw):
    '''
    Draw data (e.g. a spectrum): black, thin, steps centred on the samples.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x, y : array_like
        Sample positions (e.g. wavelength, Angstrom) and values.
    **kw
        Passed to ``ax.plot`` (``label``, ``alpha``, a different ``lw``...).

    Returns
    -------
    list of matplotlib.lines.Line2D
    '''
    kw = {'color': DATA_COLOR, 'lw': DATA_LW, **kw}
    return ax.plot(x, y, drawstyle='steps-mid', **kw)


def plot_model(ax, x, y, **kw):
    '''
    Draw a model or fit: a red line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x, y : array_like
    **kw
        Passed to ``ax.plot``.

    Returns
    -------
    list of matplotlib.lines.Line2D
    '''
    kw = {'color': MODEL_COLOR, 'lw': MODEL_LW, **kw}
    return ax.plot(x, y, **kw)


def plot_extra(ax, x, y, **kw):
    '''
    Draw the one additional series (truth, second component, comparison): a
    blue line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x, y : array_like
    **kw
        Passed to ``ax.plot``.

    Returns
    -------
    list of matplotlib.lines.Line2D
    '''
    kw = {'color': EXTRA_COLOR, 'lw': EXTRA_LW, **kw}
    return ax.plot(x, y, **kw)


def plot_curve(ax, x, y, **kw):
    '''
    Draw a derived curve evaluated on a chosen grid (a CCF, a periodogram, a
    likelihood profile): a black line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x, y : array_like
        Grid (e.g. velocity, km/s, or frequency, 1/day) and values.
    **kw
        Passed to ``ax.plot``.

    Returns
    -------
    list of matplotlib.lines.Line2D
    '''
    kw = {'color': DATA_COLOR, 'lw': CURVE_LW, **kw}
    return ax.plot(x, y, **kw)


def plot_points(ax, x, y, yerr=None, **kw):
    '''
    Draw data at scattered epochs (RVs, a phase fold): black points with
    error bars.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x, y : array_like
        Epochs (e.g. days or phase) and values (e.g. km/s).
    yerr : array_like, optional
        One-sigma uncertainties of ``y``, same units.
    **kw
        Passed to ``ax.errorbar`` (``fmt`` to change the marker, ``label``...).

    Returns
    -------
    matplotlib.container.ErrorbarContainer
    '''
    # Above any model curve drawn through the points.
    kw = {'fmt': 'o', 'color': DATA_COLOR, 'ms': 3.5, 'elinewidth': 0.8,
          'capsize': 0, 'zorder': 3, **kw}
    return ax.errorbar(x, y, yerr=yerr, **kw)


def dlnl_levels(depth):
    '''
    Contour levels for a log-likelihood map, as Delta lnL below its maximum.

    -0.5, -2, -4.5 (the 1, 2, 3 sigma bounds of one parameter), then
    -10, -30, -100, -300, ... (a 1-3 sequence) down to ``depth``: dense at
    the peak, sparse in the far field.

    Parameters
    ----------
    depth : float
        How far below the maximum the map reaches (positive, or the negative
        Delta lnL of its lowest point; only the magnitude is used).

    Returns
    -------
    ndarray
        Increasing, negative Delta lnL levels.
    '''
    depth = abs(float(depth))
    decades = 10.0 ** np.arange(1, np.ceil(np.log10(max(depth, 10.0))) + 1)
    mags = np.concatenate([SIGMA_LEVELS, np.sort(np.concatenate([decades, 3 * decades]))])
    mags = mags[mags < depth] if depth > SIGMA_LEVELS[0] else mags[:1]
    return -mags[::-1]


def dlnl_linewidths(levels):
    '''
    Contour line widths for Delta lnL levels: thicker at the 1-3 sigma levels.

    Parameters
    ----------
    levels : sequence of float
        Delta lnL levels (negative).

    Returns
    -------
    ndarray
        ``CONTOUR_INNER_LW`` where ``|level| <= 4.5``, else ``CONTOUR_LW``.
    '''
    mags = np.abs(np.asarray(levels, float))
    return np.where(mags <= SIGMA_LEVELS[-1], CONTOUR_INNER_LW, CONTOUR_LW)


def dlnl_legend_entries(levels):
    '''
    Legend handles and labels that list Delta lnL contour levels.

    One entry for the thick (``|level| <= 4.5``) and one for the thin levels,
    each present only if it has levels; at most three thin values are spelled
    out.

    Parameters
    ----------
    levels : sequence of float
        Delta lnL levels (negative).

    Returns
    -------
    handles : list of matplotlib.lines.Line2D
    labels : list of str
    '''
    mags = np.sort(np.abs(np.asarray(levels, float)))
    inner, outer = mags[mags <= SIGMA_LEVELS[-1]], mags[mags > SIGMA_LEVELS[-1]]
    handles, labels = [], []
    if inner.size:
        label = '$' + ', '.join(f'-{m:g}' for m in inner) + '$'
        if np.array_equal(inner, SIGMA_LEVELS):
            label += r' (1$-$3$\sigma$)'
        handles.append(Line2D([], [], color=DATA_COLOR, lw=CONTOUR_INNER_LW))
        labels.append(label)
    if outer.size:
        label = ('$' + ', '.join(f'-{m:g}' for m in outer[:3])
                 + (r', \ldots$' if outer.size > 3 else '$'))
        handles.append(Line2D([], [], color=DATA_COLOR, lw=CONTOUR_LW))
        labels.append(label)
    return handles, labels


def contour_map(ax, x, y, z, levels=None, **kw):
    '''
    Draw a 2-D map as black, unfilled contour lines (no colour map).

    With ``levels=None`` the map is read as a log-likelihood with any offset
    and contoured at ``max(z) + dlnl_levels(...)``; the 1-3 sigma levels are
    drawn thicker. Explicit ``levels`` are used as given, in the units of
    ``z``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x, y : 1-D array_like
        Grid along the horizontal and vertical axes (e.g. v1, v2 in km/s).
    z : 2-D array_like
        Map of shape ``(len(y), len(x))``; non-finite cells are not contoured.
    levels : sequence of float, optional
        Contour values in the units of ``z``.
    **kw
        Passed to ``ax.contour``.

    Returns
    -------
    matplotlib.contour.QuadContourSet
    '''
    z = np.asarray(z, float)
    if levels is None:
        top = np.nanmax(z)
        dlevels = dlnl_levels(top - np.nanmin(z))
        levels = top + dlevels
        lw = dlnl_linewidths(dlevels)
    else:
        levels = np.sort(np.asarray(levels, float))
        lw = CONTOUR_LW
    kw = {'colors': DATA_COLOR, 'linewidths': lw, 'linestyles': 'solid', **kw}
    return ax.contour(x, y, np.ma.masked_invalid(z), levels=levels, **kw)


def mark_fit(ax, x, y, **kw):
    '''
    Mark the best fit on a map: a red ``+``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x, y : float
        Position in data coordinates.
    **kw
        Passed to ``ax.plot``.

    Returns
    -------
    list of matplotlib.lines.Line2D
    '''
    kw = {'color': MODEL_COLOR, 'ms': 12, 'mew': 1.6, 'ls': 'none', 'zorder': 4, **kw}
    return ax.plot(x, y, marker='+', **kw)


def mark_truth(ax, x, y, **kw):
    '''
    Mark the true value on a map: a blue ``x``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x, y : float
        Position in data coordinates.
    **kw
        Passed to ``ax.plot``.

    Returns
    -------
    list of matplotlib.lines.Line2D
    '''
    kw = {'color': EXTRA_COLOR, 'ms': 9, 'mew': 1.6, 'ls': 'none', 'zorder': 4, **kw}
    return ax.plot(x, y, marker='x', **kw)
