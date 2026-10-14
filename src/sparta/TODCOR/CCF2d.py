#                   ----------------------------------
#                     CCF2d.py (SPARTA TODCOR class)
#                   ----------------------------------
# This file defines the "CCF2d" class -- the two-component (double-lined)
# counterpart of UNICOR's CCF1d. Where CCF1d correlates one observation against
# one template and reads a 1-D CCF peak, CCF2d fits *both* components of an SB2
# simultaneously against two templates, on a 2-D (v1, v2) grid: TODCOR
# (Zucker & Mazeh 1994) recast as a Gaussian profile likelihood in the Fourier
# domain.
#
# ---------------------------------------------
# A CCF2d class stores the following methods:
# ---------------------------------------------
# 1) CrossCorrelateSpec - two-dimensional correlation of a spectrum against two
#                         templates; each spectral order is one "window" and the
#                         windows are pooled into a single joint fit.
# 2) extract_RVs        - velocities, amplitudes and CRLB uncertainties from the
#                         likelihood surface.
# 3) plotTODCOR         - plot the (v1, v2) likelihood map.
# 4) BestFitSpectrum    - the best-fit composite of a given order, for residuals.
#
# The estimator itself (sufficient statistics, profiled amplitudes, CRLB) lives
# in todcor.estimator; the forward model in todcor.simulate; the
# grid/FFT/broadening primitives in unicor.dsp. This class handles the
# SPARTA-object orchestration: log-lambda gridding, per-order preparation,
# window combination, and reporting.
#
# Conventions: radial velocities in km/s; wavelengths in Angstrom; the fit is
# performed on a log-lambda (velocity-even) grid.
#
# Dependencies: numpy, astropy, matplotlib, copy.
"""Two-dimensional (TODCOR) cross-correlation of an SB2 against two templates."""

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as consts, units as u

from unicor import dsp
from todcor.simulate import model_flux
from todcor.estimator import (
    combine_stats, crlb_from_map, profile_from_stats, window_stats,
)


class CCF2d:
    # =============================================================================
    # =============================================================================
    def __init__(self):
        '''
        No input required. Sets the speed of light ``self.c`` (km/s), the
        default log-lambda grid step ``self.default_dv`` (0.1 km/s, matching
        CCF1d) and the default step of the trial-velocity map
        ``self.default_dvel`` (1 km/s).

        The map step is deliberately coarser than the spectral grid step: the
        cost of the fit scales as ``n1 * n2 * K`` (velocity nodes squared times
        Fourier modes), and the sub-grid peak refinement in
        :meth:`extract_RVs` recovers the precision a finer map would buy.
        '''
        self.c = consts.c.to('km/s')                       # speed of light, km/s
        self.default_dv = 0.1 * u.kilometer / u.second     # log-lambda grid step
        self.default_dvel = 1.0                            # (v1, v2) map step, km/s

    # =============================================================================
    # =============================================================================
    def CrossCorrelateSpec(self, spec_in, template1_in, template2_in,
                           dv=None, VelBound=150, dvel=None,
                           v1_grid=None, v2_grid=None, sigma=1.0, alpha=0.1,
                           nonneg=True, zeta_max=None, continuum='median',
                           orders=None, keep_windows=True):
        '''
        Fit both components of a double-lined spectrum simultaneously.

        Every spectral order is prepared as one TODCOR *window* (resampled
        templates, continuum removed, apodised) and reduced to its six
        sufficient statistics; the windows are then summed and profiled into a
        single joint fit with shared ``(v1, v2, a1, a2)``. Pooling orders this
        way is valid because they belong to one epoch -- see
        :func:`todcor.estimator.combine_stats`.

        :param: spec_in    - a Spectrum object with the observed data.
        :param: template1_in, template2_in - Template objects for the primary
                            and the secondary. A single-order template is reused
                            for every order of the observation.
        :param: dv         - scalar (km/s). If the spectrum is not already on a
                            log-lambda grid it is resampled with this step.
                            Default 0.1 km/s.
        :param: VelBound   - scalar or [lo, hi] (km/s). The trial-velocity range,
                            ``[-VelBound, VelBound]`` if scalar. Used for both
                            components unless explicit grids are given.
        :param: dvel       - scalar (km/s). Step of the trial-velocity map.
                            Default 1 km/s.
        :param: v1_grid, v2_grid - explicit trial-velocity arrays (km/s), which
                            override ``VelBound``/``dvel``.
        :param: sigma      - per-pixel noise of the observed flux, either a
                            scalar (all orders) or one value per order. Sets the
                            absolute scale of the uncertainties only.
        :param: alpha      - Tukey apodisation parameter applied to each window.
                            Set 0 to skip (e.g. if the spectrum was already
                            tapered by ``SpecPreProccess``).
        :param: nonneg     - Boolean. Constrain both amplitudes to be
                            non-negative (physical light fractions). Default True.
        :param: zeta_max   - optional low-pass on the Fourier axis (s/km), e.g.
                            ``R / c`` to drop scales finer than the resolution.
        :param: continuum  - 'median' (default) to remove each array's median,
                            or a float to subtract a fixed continuum level. The
                            same convention is applied to data and templates,
                            which is what makes the amplitudes meaningful.
        :param: orders     - optional list of order indices to use. Default all.
        :param: keep_windows - Boolean. Retain the per-order prepared arrays
                            (needed by :meth:`BestFitSpectrum`). Set False when
                            fitting a long time series: the arrays are the size
                            of the spectrum, so keeping them for every epoch is
                            what dominates memory.

        :return: self, with the following fields:
                 self.Corr2d  - a dictionary with 'v1_grid', 'v2_grid' (km/s),
                                'logL_profile', 'logL_marginal', 'a1', 'a2',
                                'chi2' (all (n1, n2) maps), 'best' (the profile
                                argmax) and 'units'.
                 self.stats   - the *combined* sufficient statistics, enough to
                                re-profile (e.g. with a different 'nonneg')
                                without touching the spectrum again.
                 self.windows - per-order diagnostics: the order index, the
                                window's own 'S11'/'S22'/'SD', and (if
                                keep_windows) the prepared arrays.
                 self.n_ord   - the number of pooled windows.
                 plus the fields set by :meth:`extract_RVs`.
        '''
        spec = deepcopy(spec_in)
        template1 = deepcopy(template1_in)
        template2 = deepcopy(template2_in)

        if dv is None:
            dv = self.default_dv
        elif not isinstance(dv, u.quantity.Quantity):
            dv = float(dv) << u.kilometer / u.second

        # The Fourier shift theorem holds only on a velocity-even grid.
        if spec.Info.get('GridType') != 'log':
            spec.InterpolateSpectrum(delta=dv, InterpMethod='log')
        if 'GridDelta' not in spec.Info:
            raise ValueError(
                "the spectrum claims a log-lambda grid but carries no "
                "'GridDelta'; set Info['GridDelta'] to its velocity step "
                "(km/s) or let InterpolateSpectrum build the grid.")
        dv_kms = u.Quantity(spec.Info['GridDelta']).to(u.km / u.s).value

        v1_grid, v2_grid = self._velocity_grids(VelBound, dvel, v1_grid, v2_grid)

        if orders is None:
            orders = range(len(spec.wv))
        sigma = np.broadcast_to(np.atleast_1d(np.asarray(sigma, float)),
                                (len(spec.wv),))

        stats, windows = [], []
        for I in orders:
            w = np.asarray(spec.wv[I], float)
            data_g = self._residual(np.asarray(spec.sp[I], float), continuum)
            g1 = self._residual(self._template_on_grid(template1, I, w), continuum)
            g2 = self._residual(self._template_on_grid(template2, I, w), continuum)

            if alpha > 0:
                taper = dsp.tukey_window(len(w), alpha=alpha)
                data_g, g1, g2 = data_g * taper, g1 * taper, g2 * taper

            st = window_stats(data_g, g1, g2, dv_kms, v1_grid, v2_grid,
                              sigma=sigma[I], zeta_max=zeta_max)
            stats.append(st)

            # Keep the scalar powers always -- they say which component each
            # order can constrain, and they cost nothing. The full arrays are
            # optional (see keep_windows).
            win = {'order': I, 'dv': dv_kms,
                   'S11': st['S11'], 'S22': st['S22'], 'SD': st['SD']}
            if keep_windows:
                win.update(wv=w, data_g=data_g, g1=g1, g2=g2)
            windows.append(win)

        # Only the *combined* statistics are retained: each window's A12 is an
        # (n1, n2) array, so keeping one per order would dominate memory on a
        # many-order spectrum. The combined dict is all that re-profiling needs.
        self.stats = combine_stats(stats)
        res = profile_from_stats(self.stats, v1_grid, v2_grid, nonneg=nonneg)
        res['units'] = u.km / u.s

        self.windows = windows
        self.n_ord = len(windows)
        self.Corr2d = res

        return self.extract_RVs()

    # =============================================================================
    # =============================================================================
    def extract_RVs(self, half=3):
        '''
        Velocities, amplitudes and uncertainties from the likelihood surface.

        The uncertainties are the Cramer-Rao bound from the local curvature of
        the *profile* log-likelihood at its peak, so they already account for
        the (linear) nuisance amplitudes. They are only as good as the ``sigma``
        fed to the fit: if the reduced chi2 is not ~1, rescale by
        ``sqrt(chi2_red)``.

        :param: half - half-width, in grid nodes, of the local quadratic patch.
        :return: self, with these added to self.Corr2d: 'RV1', 'RV2' (km/s,
                 sub-grid refined), 'eRV1', 'eRV2' (km/s), 'rho12' (the v1-v2
                 correlation coefficient), 'cov' (2x2, km^2/s^2), 'peak' (the
                 un-refined argmax node), and 'a1_best', 'a2_best', 'chi2_best'
                 at that peak. Note 'a1'/'a2' remain the full (n1, n2) maps.
        '''
        res = self.Corr2d
        crlb = crlb_from_map(res['logL_profile'], res['v1_grid'], res['v2_grid'],
                             half=half)

        res.update(RV1=crlb['peak_refined'][0], RV2=crlb['peak_refined'][1],
                   eRV1=crlb['sigma_v1'], eRV2=crlb['sigma_v2'],
                   rho12=crlb['rho'], cov=crlb['cov'], peak=crlb['peak'])
        res.update(a1_best=res['best']['a1'], a2_best=res['best']['a2'],
                   chi2_best=res['best']['chi2'])
        self.crlb = crlb
        return self

    # =============================================================================
    # =============================================================================
    def BestFitSpectrum(self, order=0):
        '''
        The best-fit composite flux for one window, for residual inspection.

        Uses the same Fourier phase-shift model as the fit, evaluated at the
        peak velocities and amplitudes.

        :param: order - index into ``self.windows`` (not the spectrum's own
                        order numbering, which is ``self.windows[i]['order']``).
        :return: (wv, model, data) - wavelength (Angstrom), the modelled
                 composite (continuum 1) and the windowed data residual + 1.
        '''
        win = self.windows[order]
        if 'wv' not in win:
            raise ValueError(
                'the prepared arrays were not retained; re-run '
                'CrossCorrelateSpec with keep_windows=True to inspect residuals')
        best = self.Corr2d['best']
        model = model_flux(win['wv'], win['dv'], win['g1'], win['g2'],
                           self.Corr2d['RV1'], self.Corr2d['RV2'],
                           best['a1'], best['a2'])
        return win['wv'], model, win['data_g'] + 1.0

    # =============================================================================
    # =============================================================================
    def plotTODCOR(self, levels=30, marginals=True, figsize=(6.5, 6.0)):
        '''
        Plot the (v1, v2) profile log-likelihood surface.

        :param: levels     - number of filled contour levels.
        :param: marginals  - Boolean. Add the two profile cuts through the peak.
        :param: figsize    - figure size in inches.
        :return: the figure object.
        '''
        res = self.Corr2d
        v1, v2, logL = res['v1_grid'], res['v2_grid'], res['logL_profile']
        # Plot Delta-logL relative to the peak: the absolute level carries the
        # arbitrary SD offset and swamps the structure.
        dlogL = logL - logL.max()

        fig = plt.figure(figsize=figsize)
        if marginals:
            gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                                  wspace=0.05, hspace=0.05)
            ax = fig.add_subplot(gs[1, 0])
            ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
            ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
        else:
            ax = fig.add_subplot(111)

        # The map is indexed [v1, v2]; transpose so v1 runs along x.
        im = ax.contourf(v1, v2, dlogL.T, levels=levels)
        ax.plot(res['RV1'], res['RV2'], 'w+', markersize=12, markeredgewidth=1.5)
        ax.set_xlabel(r'$v_1$ [km s$^{-1}$]')
        ax.set_ylabel(r'$v_2$ [km s$^{-1}$]')

        if marginals:
            i, j = np.unravel_index(np.argmax(logL), logL.shape)
            ax_top.plot(v1, dlogL[:, j], 'k')
            ax_right.plot(dlogL[i, :], v2, 'k')
            ax_top.tick_params(labelbottom=False)
            ax_right.tick_params(labelleft=False)
            ax_top.set_ylabel(r'$\Delta\ln L$')
            ax_right.set_xlabel(r'$\Delta\ln L$')
        else:
            fig.colorbar(im, ax=ax, label=r'$\Delta\ln L$')

        return fig

    # =============================================================================
    # Internals
    # =============================================================================
    def _velocity_grids(self, VelBound, dvel, v1_grid, v2_grid):
        '''
        Build the trial-velocity grids (km/s) from the bounds and step, unless
        explicit grids were supplied. Both components share one grid by default,
        which is also what combine_stats requires across windows.
        '''
        if v1_grid is not None and v2_grid is not None:
            return np.asarray(v1_grid, float), np.asarray(v2_grid, float)

        if isinstance(VelBound, u.quantity.Quantity):
            VelBound = VelBound.to(u.km / u.s).value
        bound = np.atleast_1d(np.asarray(VelBound, float))
        lo, hi = (-np.abs(bound[0]), np.abs(bound[0])) if bound.size == 1 else \
                 (float(np.min(bound)), float(np.max(bound)))

        if dvel is None:
            dvel = self.default_dvel
        elif isinstance(dvel, u.quantity.Quantity):
            dvel = dvel.to(u.km / u.s).value
        if dvel <= 0:
            raise ValueError(f'dvel must be positive, got {dvel}')

        grid = np.arange(lo, hi + 0.5 * dvel, dvel)
        if v1_grid is None:
            v1_grid = grid
        if v2_grid is None:
            v2_grid = grid
        return np.asarray(v1_grid, float), np.asarray(v2_grid, float)

    @staticmethod
    def _template_on_grid(template, order, wave):
        '''
        Resample one template order onto the observation's wavelength grid.

        A single-order template (e.g. a synthetic library spectrum covering the
        whole range) is reused for every order. Points outside the template's
        coverage are filled with the continuum (1.0) rather than clamped to the
        edge value, so an uncovered margin contributes no spurious lines.
        '''
        model = template.model
        I = order if len(model.wv) > 1 else 0
        return dsp.resample(np.asarray(model.wv[I], float),
                                     np.asarray(model.sp[I], float),
                                     wave, fill=1.0)

    @staticmethod
    def _residual(flux, continuum):
        '''
        Remove the continuum, so the arrays entering the fit are line residuals.

        'median' is a robust proxy for absorption-dominated, continuum-
        normalized spectra and is also correct (~0) for spectra already
        high-pass filtered by ``SpecPreProccess``. Pass a float to override.
        The continuum must be removed *before* apodisation: dropping the DC
        Fourier mode removes a constant offset for free, but not a
        window-shaped one.
        '''
        flux = np.asarray(flux, float)
        level = np.median(flux) if continuum == 'median' else float(continuum)
        return flux - level
