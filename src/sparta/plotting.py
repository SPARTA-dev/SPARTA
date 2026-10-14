"""The SPARTA figure style: data, model, one extra series, grey-scale maps.

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
2-D maps                   grey-scale image, peak black, no colour       ``map_image``
best fit / truth on a map  red ``+`` / blue ``x``                         ``mark_fit`` / ``mark_truth``
=========================  ============================================  ===============

Steps are for data only: a pixel holds the flux over its whole width, which a
step drawn about the pixel centre shows and a line does not. Scattered epochs
have no width to show, so they are points. A CCF, a periodogram or a profile
likelihood is computed on a grid of trial values chosen by the analysis and
is drawn as a line.

Red marks what was fitted or detected (a model curve, an RV, a best-fit
point); blue marks the truth (a true RV or period, an injected (v1, v2)).

Maps are grey-scale images, white at their lowest value and black at the
peak (``map_image``, with a thin grey ``map_colorbar``). A correlation map is
shaded linearly; a log-likelihood map as Delta lnL below its maximum on a
symmetric-log scale (``dlnl_norm``): linear to -4.5 (3 sigma for one
parameter), logarithmic beyond, so a map 1e5 deep still shows its structure.
The fit and truth markers carry a thin white edge, so they read on black.

``apply_style`` sets matching rcParams (font sizes, ticks, layout) for a
session; the plot methods do not call it, so they never change global state.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import SymLogNorm

DATA_COLOR = 'k'
MODEL_COLOR = 'tab:red'
EXTRA_COLOR = 'tab:blue'
GREY = '0.6'

DATA_LW = 0.75
CURVE_LW = 1.0
MODEL_LW = 1.2
EXTRA_LW = 1.0

# Maps: high values black (the peak), low ones white.
MAP_CMAP = 'Greys'

# Delta lnL = -k^2/2 is the k-sigma bound of one Gaussian parameter.
SIGMA_LEVELS = (0.5, 2.0, 4.5)
DLNL_LABEL = r'$\Delta\ln L$ (linear to $-4.5$, log beyond)'

# Keeps the red / blue markers legible on the black peak of a map.
_WHITE_EDGE = [patheffects.withStroke(linewidth=3.2, foreground='white')]

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


def dlnl_norm(depth):
    '''
    The display scale of a Delta lnL map: linear within the 3-sigma level,
    logarithmic beyond.

    A Delta lnL map of a high-S/N spectrum runs to -1e5 or deeper, so on a
    linear grey scale all its structure near the peak would share the darkest
    shade. A symmetric-log scale is linear for ``|Delta lnL| <= 4.5`` (the
    3-sigma bound of one parameter) and gives every decade beyond it the same
    share of the grey scale.

    Parameters
    ----------
    depth : float
        How far below the maximum the map reaches (positive, or the negative
        Delta lnL of its lowest point; only the magnitude is used).

    Returns
    -------
    matplotlib.colors.SymLogNorm
        From ``-depth`` (white) to 0 (the peak, black).
    '''
    depth = max(abs(float(depth)), SIGMA_LEVELS[-1])
    return SymLogNorm(linthresh=SIGMA_LEVELS[-1], linscale=1.0, vmin=-depth, vmax=0.0)


def map_image(ax, x, y, z, loglike=False, **kw):
    '''
    Draw a 2-D map as a grey-scale image: white at the lowest value, black at
    the peak (no colour).

    With ``loglike=False`` the grey scale is linear from ``min(z)`` to
    ``max(z)`` (e.g. a correlation map, R in [0, 1]). With ``loglike=True``
    ``z`` is a log-likelihood with any offset, drawn as Delta lnL below its
    maximum on ``dlnl_norm`` (linear to -4.5, logarithmic beyond); label the
    colour bar with ``DLNL_LABEL``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x, y : 1-D array_like
        Grid along the horizontal and vertical axes (e.g. v1, v2 in km/s);
        each cell is centred on its node.
    z : 2-D array_like
        Map of shape ``(len(y), len(x))``; non-finite cells are left blank.
    loglike : bool
        Read ``z`` as a log-likelihood (see above).
    **kw
        Passed to ``ax.pcolormesh`` (e.g. ``norm`` to override the scale).

    Returns
    -------
    matplotlib.collections.QuadMesh
        Pass it to ``map_colorbar``.
    '''
    z = np.ma.masked_invalid(np.asarray(z, float))
    if loglike:
        z = z - z.max()
        kw = {'norm': dlnl_norm(z.min()), **kw}
    else:
        kw = {'vmin': z.min(), 'vmax': z.max(), **kw}
    kw = {'cmap': MAP_CMAP, 'shading': 'nearest', 'rasterized': True, **kw}
    return ax.pcolormesh(x, y, z, **kw)


def map_colorbar(mesh, label, ax=None, **kw):
    '''
    A thin grey-scale colour bar for a ``map_image``.

    Parameters
    ----------
    mesh : matplotlib.collections.QuadMesh
        The ``map_image`` result.
    label : str
        What the shade means, e.g. ``'$R$'`` or ``DLNL_LABEL``.
    ax : matplotlib.axes.Axes, optional
        The axes to take the room from; default the map's own. Pass
        ``cax=`` (in ``**kw``) to draw into a given axes instead.
    **kw
        Passed to ``Figure.colorbar`` (``cax``, ``orientation``...).

    Returns
    -------
    matplotlib.colorbar.Colorbar
    '''
    if 'cax' not in kw:
        kw = {'ax': mesh.axes if ax is None else ax, 'aspect': 40, 'pad': 0.02, **kw}
    cbar = mesh.axes.figure.colorbar(mesh, label=label, **kw)
    cbar.outline.set_linewidth(0.5)
    return cbar


def mark_fit(ax, x, y, **kw):
    '''
    Mark the best fit on a map: a red ``+`` with a thin white edge.

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
    kw = {'color': MODEL_COLOR, 'ms': 12, 'mew': 1.6, 'ls': 'none', 'zorder': 4,
          'path_effects': _WHITE_EDGE, **kw}
    return ax.plot(x, y, marker='+', **kw)


def mark_truth(ax, x, y, **kw):
    '''
    Mark the true value on a map: a blue ``x`` with a thin white edge.

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
    kw = {'color': EXTRA_COLOR, 'ms': 9, 'mew': 1.6, 'ls': 'none', 'zorder': 4,
          'path_effects': _WHITE_EDGE, **kw}
    return ax.plot(x, y, marker='x', **kw)
