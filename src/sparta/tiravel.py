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

from sparta.tiravel_ops import (
    align_spectra, alignment, coadd, common_grid, correlation_matrix,
    normalize_rows,
)
from sparta.spectrum import Spectrum
from sparta.template import Template


class TIRAVEL:
    # =============================================================================
    # =============================================================================
    def __init__(self):
        '''
        No input required. Sets the default edge margin ``self.default_margin``
        (10 pixels), trimmed from each end of the epochs' common rest-frame
        range as insurance against interpolation edge effects.
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

        :param: spectra    - a list of K Spectrum objects (one per epoch), all
                            with the same number of orders.
        :param: velocities - a list of K radial velocities (km/s), the trial
                            alignment. Only their *differences* matter: adding a
                            constant to all of them leaves rho untouched and
                            shifts the returned template by the same amount, so
                            the template's rest frame is whatever zero point
                            these carry.
        :param: orders     - optional list of order indices to use. Default all.
        :param: reference  - index of the epoch whose sampling defines the
                            common grid (that epoch is then not resampled).
        :param: margin     - pixels trimmed from each end of the common range.
                            Default 10. The legacy implementation hardcoded 100
                            here; what actually matters is that the de-shifted
                            epochs overlap at all, which is now checked.
        :param: kind       - 'cubic' (default) or 'linear' resampling. The
                            shifts run to many pixels, and linear interpolation
                            smooths the lines enough to depress the very
                            correlations being measured.
        :param: keep_aligned - Boolean. Retain the aligned per-order matrices
                            (needed by :meth:`plot_template`).

        :return: self, with the following fields:
                 self.template     - a Template object holding the co-added
                                     template, one order per input order.
                 self.rho          - the alignment measure in [0, 1]: how well
                                     one principal component describes the set.
                                     This is TIRAVEL's objective function.
                 self.weights      - (K,) the co-addition weights (unit norm).
                 self.correlation  - (K, K) the epoch correlation matrix.
                 self.diagnostics  - dict with 'lambda_max', 'eigenvalues',
                                     'explained', 'n_eff' (see tiravel_ops).
                 self.velocities   - the velocities used.
                 self.grids        - the common rest-frame grid of each order.
                 self.aligned      - (if keep_aligned) the per-order aligned,
                                     normalized epoch matrices.
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
        Plot the per-epoch co-addition weights.

        A near-uniform set means every epoch contributes; a few dominant weights
        mean the template rests on a handful of spectra, which ``n_eff`` in
        self.diagnostics quantifies. A *negative* weight is a warning sign --
        that epoch anti-correlates with the rest, usually a misaligned velocity
        or a bad spectrum.

        :param: times   - optional x-axis values (e.g. BJD). Default: epoch index.
        :param: figsize - figure size in inches.
        :return: the figure object.
        '''
        x = np.arange(len(self.weights)) if times is None else np.asarray(times)

        fig, ax = plt.subplots(figsize=figsize)
        ax.stem(x, self.weights, basefmt=' ')
        ax.axhline(0.0, color='k', lw=0.6)
        ax.set_xlabel('epoch' if times is None else 'time [d]')
        ax.set_ylabel('co-addition weight')
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

        :param: order       - index into the orders that were used.
        :param: show_epochs - Boolean. Underlay the individual aligned spectra.
        :param: alpha       - transparency of the individual epochs.
        :param: figsize     - figure size in inches.
        :return: the figure object.
        '''
        if show_epochs and self.aligned is None:
            raise ValueError('the aligned epochs were not retained; re-run '
                             'build_template with keep_aligned=True')

        grid = self.grids[order]
        fig, ax = plt.subplots(figsize=figsize)
        if show_epochs:
            for row in self.aligned[order]:
                ax.plot(grid, row, 'k', lw=0.5, alpha=alpha)
        ax.plot(grid, self.template.model.sp[order], 'C3', lw=1.2,
                label='TIRAVEL template')
        ax.set_xlabel(r'rest-frame wavelength [$\AA$]')
        ax.set_ylabel('normalized flux')
        ax.legend(loc='lower right')
        return fig
