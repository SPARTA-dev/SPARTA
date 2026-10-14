"""TIRAVEL: build a template out of the observations themselves.

The template-independent arm alongside UNICOR (``CCF1d``, one template) and
TODCOR (``CCF2d``, two templates): given trial radial velocities, every epoch is
shifted to rest and resampled onto a common grid, the alignment measure
``rho = (lambda_max - 1)/(K - 1)`` is read from the epochs' correlation matrix,
and its leading eigenvector weights the co-added template, returned as a
``Template`` for UNICOR or TODCOR. This is the fixed-velocity half of the
method: ``rho`` is evaluated at the supplied velocities, as the objective a
velocity optimizer would maximise. The array-level measure is in
``sparta.tiravel_ops``.

Conventions: velocities in km/s (only their differences matter), wavelengths
in Angstrom.

Reference: Zucker & Mazeh 2006, MNRAS 371, 1513.
"""

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from sparta import plotting as sp_plot
from sparta.tiravel_ops import (
    align_spectra, alignment, coadd, common_grid, correlation_matrix,
    normalize_rows,
)
from sparta.spectrum import Spectrum
from sparta.template import Template


class TIRAVEL:
    '''
    A template co-added from the observations at trial velocities (Zucker &
    Mazeh 2006); ``build_template`` sets the results as attributes.
    '''
    # =============================================================================
    # =============================================================================
    def __init__(self):
        '''
        Set the default edge margin ``self.default_margin`` (10 pixels).

        The margin is trimmed from each end of the epochs' common rest-frame
        range, as insurance against interpolation edge effects.
        '''
        self.default_margin = 10

    # =============================================================================
    # =============================================================================
    def build_template(self, spectra, velocities, orders=None, reference=0,
                       margin=None, kind='cubic', keep_aligned=True):
        '''
        Build the optimal template for a set of observations at given velocities.

        Every epoch is Doppler-shifted to its rest frame and resampled onto one
        common grid (per order). The epochs are mean-subtracted per order and
        L2-normalized, their K x K correlation matrix is formed as a single Gram
        product, and its leading eigenpair gives both the alignment measure
        ``rho = (lambda_max - 1)/(K - 1)`` and the weights with which the epochs
        should be summed. Orders are pooled into one correlation matrix, so a
        single weight vector describes the whole spectrum.

        Parameters
        ----------
        spectra : list of Spectrum
            K spectra (one per epoch, K >= 2), all with the same number of
            orders; not modified.
        velocities : array_like
            K radial velocities (km/s), the trial alignment. Only their
            *differences* matter: adding a constant to all of them leaves rho
            untouched and shifts the returned template by the same amount, so
            the template's rest frame is whatever zero point these carry.
        orders : sequence of int, optional
            Order indices to use. Default all.
        reference : int
            Index of the epoch whose sampling defines the common grid (that
            epoch is then not resampled).
        margin : int, optional
            Pixels trimmed from each end of the common range. Default
            ``self.default_margin`` (10).
        kind : {'cubic', 'linear'}
            Resampling. The shifts run to many pixels, and linear
            interpolation smooths the lines enough to depress the very
            correlations being measured.
        keep_aligned : bool
            Retain the aligned per-order matrices (needed by
            :meth:`plot_template`).

        Returns
        -------
        TIRAVEL
            self, with these attributes:

            ``template``
                Template holding the co-added spectrum, one order per order
                used (wavelengths in Angstrom; normalized, dimensionless
                flux).
            ``rho``
                The alignment measure in [0, 1]: how well one principal
                component describes the set. TIRAVEL's objective function.
            ``weights``
                (K,) co-addition weights (unit norm).
            ``correlation``
                (K, K) epoch correlation matrix.
            ``diagnostics``
                dict with 'lambda_max', 'eigenvalues', 'explained', 'n_eff'
                (see ``sparta.tiravel_ops.alignment``).
            ``velocities``
                The velocities used (km/s).
            ``grids``
                The common rest-frame grid of each order (Angstrom).
            ``aligned``
                If ``keep_aligned``, the per-order aligned, normalized epoch
                matrices; else None.

        Raises
        ------
        ValueError
            For mismatched or too few epochs, non-finite velocities or flux,
            epochs with different numbers of orders, or epochs that do not
            overlap once de-shifted.
        '''
        if len(spectra) != len(velocities):
            raise ValueError(
                f'got {len(spectra)} spectra but {len(velocities)} velocities')
        if len(spectra) < 2:
            raise ValueError('TIRAVEL needs at least 2 epochs')

        specs = [deepcopy(s) for s in spectra]
        velocities = np.asarray(velocities, float)
        if not np.all(np.isfinite(velocities)):
            raise ValueError('the supplied velocities contain NaN/inf; drop '
                             'those epochs before building a template')

        n_ord = len(specs[0].wv)
        if any(len(s.wv) != n_ord for s in specs):
            raise ValueError('all epochs must have the same number of orders')
        if orders is None:
            orders = range(n_ord)
        if margin is None:
            margin = self.default_margin

        # Align every epoch, order by order, onto that order's common grid.
        grids, aligned = [], []
        for I in orders:
            waves = [np.asarray(s.wv[I], float) for s in specs]
            fluxes = [np.asarray(s.sp[I], float) for s in specs]
            # A single NaN would otherwise zero that epoch's norm (linear) or
            # stop the spline with an opaque message (cubic).
            for k, f in enumerate(fluxes):
                if not np.all(np.isfinite(f)):
                    raise ValueError(f'epoch {k}, order {I}: flux contains NaN/inf; '
                                     'mask or interpolate those pixels first')
            grid = common_grid(waves, velocities, reference=reference,
                               margin=margin)
            grids.append(grid)
            aligned.append(align_spectra(waves, fluxes, velocities, grid,
                                         kind=kind))

        # One normalization, one correlation matrix, one weight vector for the
        # whole spectrum: the orders are different views of the same alignment.
        normalized = normalize_rows(aligned)
        correlation = correlation_matrix(normalized)
        stats = alignment(correlation)
        weights = stats.pop('weights')
        rho = stats.pop('rho')

        # Split the normalized rows back per order and co-add each.
        sizes = np.cumsum([0] + [len(g) for g in grids])
        blocks = [normalized[:, a:b] for a, b in zip(sizes[:-1], sizes[1:])]
        template_sp = [coadd(block, weights) for block in blocks]

        self.template = Template(Spectrum(
            wv=[np.asarray(g, float) for g in grids],
            sp=template_sp,
            name=getattr(specs[0], 'name', 'TIRAVEL')))
        self.rho = rho
        self.weights = weights
        self.correlation = correlation
        self.diagnostics = stats
        self.velocities = velocities
        self.grids = grids
        self.aligned = blocks if keep_aligned else None
        self.n_ord = len(grids)

        return self

    # =============================================================================
    # =============================================================================
    def plot_weights(self, times=None, figsize=(7, 3.5)):
        '''
        Plot the per-epoch co-addition weights, as black stems.

        A near-uniform set means every epoch contributes; a few dominant weights
        mean the template rests on a handful of spectra, which ``n_eff`` in
        self.diagnostics quantifies. A *negative* weight is a warning sign --
        that epoch anti-correlates with the rest, usually a misaligned velocity
        or a bad spectrum.

        Parameters
        ----------
        times : array_like, optional
            x-axis values (days, e.g. BJD). Default: the epoch index.
        figsize : (float, float)
            Figure size (inches).

        Returns
        -------
        matplotlib.figure.Figure
        '''
        x = np.arange(len(self.weights)) if times is None else np.asarray(times)

        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
        markers, stems, _ = ax.stem(x, self.weights, basefmt=' ')
        markers.set(color=sp_plot.DATA_COLOR, markersize=3.5)
        stems.set(color=sp_plot.DATA_COLOR, linewidth=0.8)
        ax.axhline(0.0, color=sp_plot.GREY, lw=0.6)
        ax.set_xlabel('Epoch' if times is None else 'Time (days)')
        ax.set_ylabel('Co-addition weight')
        ax.set_title(rf'$\rho$ = {self.rho:.4f},  '
                     rf'$n_{{\rm eff}}$ = {self.diagnostics["n_eff"]:.1f}'
                     rf' of {len(self.weights)}')
        return fig

    # =============================================================================
    # =============================================================================
    def plot_template(self, order=0, show_epochs=True, alpha=0.15,
                      figsize=(11, 4)):
        '''
        Plot the co-added template over the aligned epochs that made it.

        The epochs are data (black, mid-centred steps, faint); the template is
        the model (red line), as in ``sparta.plotting``.

        Parameters
        ----------
        order : int
            Index into the orders that were used.
        show_epochs : bool
            Underlay the individual aligned spectra.
        alpha : float
            Opacity of the individual epochs.
        figsize : (float, float)
            Figure size (inches).

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ValueError
            If ``show_epochs`` but the aligned epochs were not kept.
        '''
        if show_epochs and self.aligned is None:
            raise ValueError('the aligned epochs were not retained; re-run '
                             'build_template with keep_aligned=True')

        grid = self.grids[order]
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
        if show_epochs:
            for k, row in enumerate(self.aligned[order]):
                sp_plot.plot_data(ax, grid, row, lw=0.5, alpha=alpha,
                                  label='aligned epochs' if k == 0 else None)
        sp_plot.plot_model(ax, grid, self.template.model.sp[order], label='TIRAVEL template')
        ax.set_xlabel(r'Rest-frame wavelength ($\AA$)')
        ax.set_ylabel('Normalized flux')
        ax.legend(loc='lower right')
        return fig
