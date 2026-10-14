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
#                         windows are pooled into a single joint fit. A coarse
#                         (v1, v2) map locates the peak, a fine local map refines
#                         it, and the velocities, amplitudes and CRLB
#                         uncertainties are read from the fine map.
# 2) plotTODCOR         - plot the (v1, v2) likelihood map.
# 3) BestFitSpectrum    - the best-fit composite of a given order, for residuals.
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
import warnings

import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as consts, units as u

from unicor import dsp
from todcor.simulate import model_flux
from todcor.estimator import (
    combine_stats, crlb_from_map, profile_from_stats, resampled_noise_psd,
    window_stats,
)
from sparta.UNICOR.Spectrum import Spectrum


class CCF2d:
    # =============================================================================
    # =============================================================================
    def __init__(self):
        '''
        No input required. Sets the speed of light ``self.c`` (km/s), the
        default log-lambda grid step ``self.default_dv`` (0.1 km/s, matching
        CCF1d), the default step of the trial-velocity map
        ``self.default_dvel`` (1 km/s), and the peak-refinement geometry.

        The map step is deliberately coarser than the spectral grid step: the
        cost of the fit scales as ``n1 * n2 * K`` (velocity nodes squared times
        Fourier modes). The coarse map only has to *locate* the peak; a second,
        exact evaluation on a fine local grid (``refine_span`` coarse steps
        either side of the coarse argmax, ``refine_factor`` times finer) then
        measures it, for ~(2 * 2 * 10 + 1)^2 = 1681 extra nodes.

        Why a fine map rather than a quadratic fit to the coarse one: logL
        follows the *squared* correlation, so its peak is only ~sigma_ccf/sqrt(2)
        wide (~2.4 km/s for 8 km/s lines), and a quadratic over +/-3 coarse
        nodes picks up its quartic fall-off -- a bias of several CRLB sigma.
        For the coarse argmax to be trustworthy at all the coarse step must
        already resolve the peak (dvel <~ sigma_ccf); a tenth of it puts the
        +/-``crlb_half``-node quadratic patch within ~0.3 sigma_ccf of the peak,
        where the quartic term changes the curvature by only a few per cent and
        the peak location negligibly.
        '''
        self.c = consts.c.to('km/s')                       # speed of light, km/s
        self.default_dv = 0.1 * u.kilometer / u.second     # log-lambda grid step
        self.default_dvel = 1.0                            # (v1, v2) map step, km/s
        self.refine_span = 2       # fine-map half-width, in coarse steps
        self.refine_factor = 10    # coarse step / fine step
        self.crlb_half = 3         # half-width (nodes) of the quadratic patch

    # =============================================================================
    # =============================================================================
    def CrossCorrelateSpec(self, spec_in, template1_in, template2_in,
                           dv=None, VelBound=150, dvel=None,
                           v1_grid=None, v2_grid=None, sigma=None, alpha=0.1,
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

        The coarse map locates the peak; the statistics are then re-evaluated
        on a fine local grid around it (see ``__init__``), and the velocities
        and their Cramer-Rao errors come from the fine map. The amplitudes and
        chi2 are evaluated exactly at the refined velocities. If the coarse
        peak is on the grid edge no refinement is done: a warning is issued and
        the errors are NaN.

        Each order counts its *native* pixels, not its resampled ones: the
        spectrum's ``native_steps()`` (captured before any resampling) give
        ``oversampling = max(1, dv_native / dv)``, and the Fisher weights are
        divided by it. Otherwise the ~``dv_native / dv`` analysis pixels that
        interpolate one native pixel would each count as independent, and the
        CRLB would shrink by ``sqrt(dv_native / dv)``.

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
        :param: sigma      - per-pixel noise of the observed flux on its
                            *native* pixels (flux units), either a scalar (all
                            orders) or one value per order; taken at face value.
                            Default None: fit at unit noise and estimate one
                            common noise level from the fit, scaling the errors
                            by sqrt(chi2_red). Either way it sets the absolute
                            scale of the uncertainties only, not the RVs.
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
                 self.Corr2d  - a dictionary with the coarse map: 'v1_grid',
                                'v2_grid' (km/s), 'logL_profile',
                                'logL_marginal', 'a1', 'a2', 'chi2' (all
                                (n1, n2) maps), 'best' (its argmax node) and
                                'units'; 'fine', the same for the fine local map
                                (None if refinement was skipped); and the
                                results:
                                'RV1', 'RV2' (km/s, refined), 'eRV1', 'eRV2'
                                (km/s), 'rho12' (v1-v2 correlation), 'cov'
                                (2x2, km^2/s^2), 'peak' (the coarse argmax
                                node, km/s), 'a1_best', 'a2_best', 'chi2_best'
                                (at the refined RVs), 'chi2_red' (chi2_best per
                                noise degree of freedom, see below), 'sigma'
                                (the native per-pixel noise behind the errors:
                                the input, or the fitted estimate) and
                                'sigma_from_fit'.
                 self.stats   - the *combined* coarse-map sufficient statistics,
                                enough to re-profile (e.g. with a different
                                'nonneg') without touching the spectrum again.
                 self.windows - per-order diagnostics: the order index, the
                                window's own 'S11'/'S22'/'SD', 'oversampling'
                                (dv_native / dv, at least 1), 'n_eff' (the
                                independent-pixel count n_pix / oversampling),
                                'nu' (the expected chi2 of its pure noise), and
                                (if keep_windows) the prepared arrays.
                 self.n_ord   - the number of pooled windows.
                 self.crlb    - the crlb_from_map output behind the RVs.

        chi2_red = chi2_best / (sum(nu) - 4). Interpolated noise is smoother
        than the native noise -- a cubic spline keeps ~0.87 of its variance,
        and the taper ~0.94 more -- so dividing by the bare n_eff would leave
        chi2_red ~0.8 for honest noise. ``nu`` instead counts the noise power
        the fit actually sees (todcor.estimator.resampled_noise_psd, from the
        resampler's impulse response), so chi2_red ~ 1 for honest noise.
        '''
        spec = deepcopy(spec_in)
        template1 = deepcopy(template1_in)
        template2 = deepcopy(template2_in)

        if dv is None:
            dv = self.default_dv
        elif not isinstance(dv, u.quantity.Quantity):
            dv = float(dv) << u.kilometer / u.second

        native = spec.native_steps()

        # The Fourier shift theorem holds only on a velocity-even grid.
        if spec.Info.get('GridType') != 'log':
            spec.InterpolateSpectrum(delta=dv, InterpMethod='log')
        if 'GridDelta' not in spec.Info:
            raise ValueError(
                "the spectrum claims a log-lambda grid but carries no "
                "'GridDelta'; set Info['GridDelta'] to its velocity step "
                "(km/s) or let InterpolateSpectrum build the grid.")
        dv_kms = u.Quantity(spec.Info['GridDelta']).to(u.km / u.s).value
        for I, w in enumerate(spec.wv):
            step = dsp.log_grid_step(w)
            if not np.isclose(step, dv_kms, rtol=1e-3):
                raise ValueError(
                    f"order {I}: the wavelength grid step is {step:.6g} km/s but "
                    f"Info['GridDelta'] declares {dv_kms:.6g} km/s")

        v1_grid, v2_grid = self._velocity_grids(VelBound, dvel, v1_grid, v2_grid)

        if orders is None:
            orders = range(len(spec.wv))
        sigma_from_fit = sigma is None
        sigma_in = 1.0 if sigma_from_fit else sigma
        sigma = np.broadcast_to(np.atleast_1d(np.asarray(sigma_in, float)),
                                (len(spec.wv),))

        # The prepared arrays are needed until the fine pass and the evaluation
        # at the refined peak are done; only keep_windows decides whether they
        # outlive this call.
        prepared, stats, windows = [], [], []
        for I in orders:
            w = np.asarray(spec.wv[I], float)
            data_g = self._residual(np.asarray(spec.sp[I], float), continuum)
            g1 = self._residual(self._template_on_grid(template1, I, w), continuum)
            g2 = self._residual(self._template_on_grid(template2, I, w), continuum)
            for name, arr in (('spectrum', data_g), ('template1', g1), ('template2', g2)):
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"order {I}: {name} flux contains NaN/inf; "
                                     "mask or interpolate those pixels first")

            taper_power = 1.0
            if alpha > 0:
                taper = dsp.tukey_window(len(w), alpha=alpha)
                data_g, g1, g2 = data_g * taper, g1 * taper, g2 * taper
                taper_power = float(np.mean(taper ** 2))

            ratio = native[I] / dv_kms
            oversampling = max(1.0, ratio)
            noise_psd = taper_power * self._resampling_noise_psd(
                w, native[I], dv, ratio)
            kw = dict(sigma=sigma[I], zeta_max=zeta_max,
                      oversampling=oversampling, noise_psd=noise_psd)
            st = window_stats(data_g, g1, g2, dv_kms, v1_grid, v2_grid, **kw)
            stats.append(st)
            prepared.append((data_g, g1, g2, kw))

            # Keep the scalar powers always -- they say which component each
            # order can constrain, and they cost nothing. The full arrays are
            # optional (see keep_windows).
            win = {'order': I, 'dv': dv_kms,
                   'S11': st['S11'], 'S22': st['S22'], 'SD': st['SD'],
                   'oversampling': oversampling, 'n_eff': len(w) / oversampling,
                   'nu': st['nu']}
            if keep_windows:
                win.update(wv=w, data_g=data_g, g1=g1, g2=g2)
            windows.append(win)

        # Only the *combined* statistics are retained: each window's A12 is an
        # (n1, n2) array, so keeping one per order would dominate memory on a
        # many-order spectrum. The combined dict is all that re-profiling needs.
        self.stats = combine_stats(stats)
        # A template with no lines over the fitted orders makes the 2x2 amplitude
        # system singular, and the map collapses to a flat, meaningless argmax.
        for k, name in (('S11', 'template1'), ('S22', 'template2')):
            if not self.stats[k] > 0:
                raise ValueError(
                    f"{name} has no spectral structure over the fitted orders "
                    f"({k} = {self.stats[k]}); check its wavelength coverage and "
                    "air/vacuum frame against the observation")
        res = profile_from_stats(self.stats, v1_grid, v2_grid, nonneg=nonneg)
        res['units'] = u.km / u.s

        coarse = crlb_from_map(res['logL_profile'], v1_grid, v2_grid,
                               half=self.crlb_half)
        if coarse['on_edge']:
            warnings.warn("the TODCOR peak lies on the edge of the velocity grid; "
                          "the RVs are grid-limited and their errors are NaN. "
                          "Widen VelBound (or the explicit grids).", stacklevel=2)
            res['fine'] = None
        else:
            res['fine'] = self._refine(prepared, dv_kms, res, nonneg)
        crlb = coarse if res['fine'] is None else crlb_from_map(
            res['fine']['logL_profile'], res['fine']['v1_grid'],
            res['fine']['v2_grid'], half=self.crlb_half)

        rv1, rv2 = crlb['peak_refined']
        at_peak = self._profile_at(prepared, dv_kms, rv1, rv2, nonneg)
        chi2_red = at_peak['chi2_red']
        noise, scale = sigma_in, 1.0
        if sigma_from_fit:
            if not (np.isfinite(chi2_red) and chi2_red > 0):
                raise ValueError(
                    f"cannot estimate the noise from the fit (chi2_red = {chi2_red}); "
                    "pass an explicit sigma")
            # The fit ran at unit noise, so sqrt(chi2_red) *is* the common
            # native per-pixel noise; chi2 and the map curvature scale as
            # 1/sigma^2, the velocity errors as sigma.
            noise = scale = float(np.sqrt(chi2_red))

        res.update(RV1=rv1, RV2=rv2,
                   eRV1=crlb['sigma_v1'] * scale, eRV2=crlb['sigma_v2'] * scale,
                   rho12=crlb['rho'], cov=crlb['cov'] * scale ** 2,
                   peak=coarse['peak'],
                   a1_best=at_peak['a1'], a2_best=at_peak['a2'],
                   chi2_best=at_peak['chi2'], chi2_red=chi2_red,
                   sigma=noise, sigma_from_fit=sigma_from_fit)

        self.windows = windows
        self.n_ord = len(windows)
        self.Corr2d = res
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
        r = self.Corr2d
        model = model_flux(win['wv'], win['dv'], win['g1'], win['g2'],
                           r['RV1'], r['RV2'], r['a1_best'], r['a2_best'])
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
        # Explicit grids may carry units; anything else is taken as km/s.
        v1_grid, v2_grid = (g.to_value(u.km / u.s) if isinstance(g, u.Quantity) else g
                            for g in (v1_grid, v2_grid))
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

    def _refine(self, prepared, dv_kms, res, nonneg):
        '''
        Re-evaluate the statistics on a fine (v1, v2) grid around the coarse
        argmax and profile them -- the second, exact pass (see ``__init__``).

        :return: the fine-map profile_from_stats dict, or None (with a warning)
                 if its argmax lands on its own edge -- the coarse argmax was
                 then not close to the true peak (e.g. a ridge or a tie), and
                 the coarse map is the more honest answer.
        '''
        v1, v2 = res['v1_grid'], res['v2_grid']
        i, j = np.unravel_index(np.argmax(res['logL_profile']), res['logL_profile'].shape)
        # Local steps, so explicit non-uniform grids refine sensibly too.
        step1 = 0.5 * (v1[i + 1] - v1[i - 1])
        step2 = 0.5 * (v2[j + 1] - v2[j - 1])
        nodes = np.arange(-self.refine_span * self.refine_factor,
                          self.refine_span * self.refine_factor + 1) / self.refine_factor
        f1, f2 = v1[i] + step1 * nodes, v2[j] + step2 * nodes

        stats = combine_stats(window_stats(d, g1, g2, dv_kms, f1, f2, **kw)
                              for d, g1, g2, kw in prepared)
        fine = profile_from_stats(stats, f1, f2, nonneg=nonneg)
        fi, fj = np.unravel_index(np.argmax(fine['logL_profile']), fine['logL_profile'].shape)
        if fi in (0, len(f1) - 1) or fj in (0, len(f2) - 1):
            warnings.warn("the refined TODCOR peak lies outside the fine map "
                          f"(+/-{self.refine_span} coarse steps around the coarse "
                          "argmax); reporting the coarse-map estimate, whose "
                          "sub-grid refinement may be biased.", stacklevel=3)
            return None
        return fine

    @staticmethod
    def _profile_at(prepared, dv_kms, v1, v2, nonneg):
        '''
        Profile amplitudes, chi2 and chi2_red exactly at one (v1, v2) (km/s),
        so the reported amplitudes belong to the reported velocities.
        '''
        g1_, g2_ = np.array([v1]), np.array([v2])
        stats = combine_stats(window_stats(d, g1, g2, dv_kms, g1_, g2_, **kw)
                              for d, g1, g2, kw in prepared)
        return profile_from_stats(stats, g1_, g2_, nonneg=nonneg)['best']

    @staticmethod
    def _resampling_noise_psd(wave, dv_native, dv, ratio, n_imp=8, spacing=40):
        '''
        Noise spectrum (todcor.estimator.resampled_noise_psd) of one order's
        native white noise after Spectrum.InterpolateSpectrum to step ``dv``.

        Resampling is linear, so it is fully described by its impulse response:
        unit impulses on a synthetic native log grid of step ``dv_native``
        (km/s), well separated (``spacing`` native pixels, far beyond the
        spline's support) are pushed through the same InterpolateSpectrum call
        CCF2d uses. This models a single resampling from a uniform native grid;
        earlier filtering or repeated resampling is not represented.

        :param: wave      - the order's analysis wavelength grid (Angstrom).
        :param: dv_native - the order's native velocity step (km/s).
        :param: dv        - the analysis step (astropy Quantity, km/s).
        :param: ratio     - dv_native / dv (dimensionless).
        :return: relative noise power per positive Fourier mode, length
                 len(wave) // 2 -- all ones for an order on its native sampling.
        '''
        n = len(wave)
        if np.isclose(ratio, 1.0, rtol=1e-3):
            return np.ones(n // 2)
        lam0 = float(np.median(wave))
        native, _ = dsp.log_lambda_grid(
            lam0, lam0 * np.exp((n_imp + 1) * spacing * dv_native / dsp.C_KMS), dv_native)
        hits = spacing // 2 + spacing * np.arange(n_imp)
        flux = np.zeros(len(native))
        flux[hits] = 1.0
        probe = Spectrum(wv=[native], sp=[flux]).InterpolateSpectrum(
            delta=dv, InterpMethod='log')
        wa, ya = np.asarray(probe.wv[0]), np.asarray(probe.sp[0])
        responses = [ya[np.abs(dsp.C_KMS * np.log(wa / native[h]))
                        < 0.5 * spacing * dv_native] for h in hits]
        return resampled_noise_psd(responses, n, ratio)

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
