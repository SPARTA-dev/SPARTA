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

import numpy as np

from sparta.tiravel_ops import build_tiravel
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
        single weight vector describes the whole spectrum. The arrays are
        built by ``sparta.tiravel_ops.build_tiravel``; this wraps them.

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
        if margin is None:
            margin = self.default_margin
        res = build_tiravel([s.wv for s in spectra], [s.sp for s in spectra], velocities,
                            orders=orders, reference=reference, margin=margin, kind=kind)

        self.template = Template(Spectrum(
            wv=[np.asarray(g, float) for g in res.grids],
            sp=res.template_flux,
            name=getattr(spectra[0], 'name', 'TIRAVEL')))
        self.rho = res.rho
        self.weights = res.weights
        self.correlation = res.correlation
        self.diagnostics = res.diagnostics
        self.velocities = res.velocities
        self.grids = res.grids
        self.aligned = res.aligned if keep_aligned else None
        self.n_ord = len(res.grids)

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
        # The plotting stack loads on the first plot, not on import.
        import matplotlib.pyplot as plt
        from sparta import plotting as sp_plot

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
        the model (red line), as in ``sparta.plotting``. Both are drawn on the
        scale of the aligned epochs -- mean-subtracted, each of unit L2 norm
        across the orders used -- so the template appears divided by the sum
        of the weights, i.e. as the weighted mean of the epochs. (The stored
        co-add, of unit-norm epochs with unit-norm weights, is ~sqrt(K) times
        larger; its scale is arbitrary.)

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
        import matplotlib.pyplot as plt
        from sparta import plotting as sp_plot

        if show_epochs and self.aligned is None:
            raise ValueError('the aligned epochs were not retained; re-run '
                             'build_template with keep_aligned=True')

        grid = self.grids[order]
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
        if show_epochs:
            for k, row in enumerate(self.aligned[order]):
                sp_plot.plot_data(ax, grid, row, lw=0.5, alpha=alpha,
                                  label='aligned epochs' if k == 0 else None)
        sp_plot.plot_model(ax, grid, self.template.model.sp[order] / np.sum(self.weights),
                           label='TIRAVEL template (weighted mean)')
        ax.set_xlabel(r'Rest-frame wavelength ($\AA$)')
        ax.set_ylabel('Flux (mean-subtracted, unit-norm epochs)')
        ax.legend(loc='lower right')
        return fig
