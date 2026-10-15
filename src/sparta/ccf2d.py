"""CCF2d: TODCOR fit of a double-lined (SB2) Spectrum against two Templates.

The two-component counterpart of UNICOR's ``CCF1d``: both components are fitted
simultaneously on a (v1, v2) grid -- TODCOR recast as a Gaussian profile
likelihood in the Fourier domain. Each spectral order is one TODCOR window;
the windows' sufficient statistics are pooled into one joint fit, a coarse map
locates the peak and a fine local map measures the velocities, amplitudes and
Cramer-Rao uncertainties. The estimator lives in ``todcor.estimator``, the
forward model in ``todcor.simulate``, the grid/FFT/broadening primitives in
``unicor.dsp``, the noise model of the resampling in ``sparta.preprocess``;
this class handles the SPARTA-object orchestration. ``cross_correlate`` is a
composition of per-window atoms on plain arrays -- ``todcor.prepare_window``,
``todcor.window_stats``, ``todcor.joint_fit``, ``todcor.window_noise`` and
``sparta.preprocess.resampling_noise_psd`` / ``native_phases`` -- so chaining
them by hand reproduces its result exactly.

Two output modes (``output``): 'ccf' (default) makes the two-dimensional
correlation R(v1, v2) of Zucker & Mazeh (1994) the primary map, 'loglike' the
profile log-likelihood. Both maps are always stored and share their argmax;
the RVs, amplitudes and Cramer-Rao errors come from the log-likelihood in both
modes and are identical.

Conventions: velocities in km/s, 'cz' (lambda_obs/lambda_rest = 1 + v/c) unless
``velocity_convention='relativistic'``; wavelengths in Angstrom; the fit runs on
a log-lambda (velocity-even) grid.

Reference: Zucker & Mazeh 1994, ApJ 420, 806.
"""

from copy import deepcopy
import warnings

import numpy as np
from astropy import units as u

from unicor import dsp
from unicor.ccf import check_output
from todcor.simulate import model_flux
from todcor.estimator import (
    combine_stats, joint_fit, prepare_window, window_noise, window_stats,
)
from sparta.preprocess import native_phases, resampling_noise_psd


class CCF2d:
    '''
    TODCOR fit of one double-lined (multi-order) spectrum against two templates.

    ``cross_correlate`` fills ``Corr2d`` with the RVs, amplitudes and
    Cramer-Rao errors of both components, and the (v1, v2) maps: the
    two-dimensional correlation 'R' and the log-likelihood 'logL_profile'.
    '''
    # =============================================================================
    # =============================================================================
    def __init__(self):
        '''
        Set the default grid and map steps and the peak-refinement geometry.

        The defaults: the log-lambda grid step ``self.default_dv`` (0.1 km/s,
        matching CCF1d); the trial-velocity map step ``self.default_dvel``
        (1 km/s); ``refine_span``, ``refine_factor`` and ``crlb_half`` (below);
        and ``self.min_noise_dof`` (50), the fewest noise degrees of freedom
        an order needs for its own noise estimate under ``sigma=None`` (below
        that its estimate would scatter by more than ~10 per cent; see
        ``cross_correlate``).

        The map step is deliberately coarser than the spectral grid step: the
        cost of the fit scales as ``n1 * n2 * K`` (velocity nodes squared times
        Fourier modes). The coarse map only has to *locate* the peak; a second,
        exact evaluation on a fine local grid (``refine_span`` coarse steps
        either side of the coarse argmax, ``refine_factor`` times finer) then
        measures it, for ~(2 * 2 * 10 + 1)^2 = 1681 extra nodes.

        The peak and its errors come from a degree-6 polynomial fitted to the
        fine log-likelihood map over +/-``crlb_half`` nodes
        (``todcor.crlb_from_map``): logL follows the *squared* correlation,
        so its peak is narrow (~sigma_ccf/sqrt(2)), and a quadratic patch
        misread it by up to ~6 m/s for lines of ~2 km/s FWHM; degree 6 reads it to
        ~0.02 m/s. For the coarse argmax to be trustworthy the coarse step must
        resolve the peak (dvel <~ sigma_ccf).
        '''
        self.default_dv = 0.1 * u.kilometer / u.second     # log-lambda grid step
        self.default_dvel = 1.0                            # (v1, v2) map step, km/s
        self.refine_span = 2       # fine-map half-width, in coarse steps
        self.refine_factor = 10    # coarse step / fine step
        self.crlb_half = 4         # half-width (nodes) of the degree-6 peak patch
        self.min_noise_dof = 50    # fewest dof to estimate one window's noise

    # =============================================================================
    # =============================================================================
    def cross_correlate(self, spectrum, template1, template2,
                        dv=None, vel_bound=150, dvel=None,
                        v1_grid=None, v2_grid=None, sigma=None, nonneg=True,
                        amplitudes='shared', orders=None, keep_windows=True,
                        velocity_convention='cz', output='ccf'):
        '''
        Fit both components of a double-lined spectrum simultaneously.

        Every spectral order is prepared as one TODCOR *window*: its observed
        pixels, and the two templates on that grid extended by the largest
        trial velocity, so that every shift reads the templates' real
        spectrum -- nothing wraps around, no line goes missing at the edges,
        and no taper is needed (``todcor.prepare_window``). Orders whose
        templates do not reach that far lose the uncovered pixels. Each window
        is reduced to its sufficient statistics over its observed pixels, with
        a free continuum offset profiled out, and the windows are profiled
        into one joint fit with shared ``(v1, v2)``: with shared ``(a1, a2)``
        too (``amplitudes='shared'``, by summing the statistics), or with each
        order's own amplitudes, i.e. its own light ratio
        (``amplitudes='per_order'``). Pooling orders is valid because they
        belong to one epoch -- see :func:`todcor.estimator.combine_stats`.

        The coarse map locates the peak; the statistics are then re-evaluated
        on a fine local grid around it (see ``__init__``), and the velocities
        and their Cramer-Rao errors come from the fine map. The amplitudes and
        chi2 are evaluated exactly at the refined velocities. If the coarse
        peak is on the grid edge no refinement is done: a warning is issued and
        the errors are NaN.

        Each order counts its *native* pixels, not its resampled ones: the
        spectrum's ``native_steps()`` (the coarsest sampling it has had before
        this call) give
        ``oversampling = max(1, dv_native / dv) / noise_fraction``, and the
        Fisher weights are divided by it. Otherwise the ~``dv_native / dv``
        analysis pixels that interpolate one native pixel would each count as
        independent, and the CRLB would shrink by ``sqrt(dv_native / dv)``.
        ``noise_fraction`` is ``Info['NoiseFraction']`` (1 when absent): after
        a band-pass only that share of the native noise is left, and of the
        native pixels' information, as in CCF1d.

        sigma=None runs the joint fit twice. Pass 1 fits every order at unit
        noise and locates the joint peak (coarse and fine maps). Each order's
        noise is then estimated from its own share of the joint chi2 there,
        sigma_i^2 = chi2_i / dof_i, with the shared parameters' dof charged
        in proportion to nu_i and, under ``amplitudes='per_order'``, each
        order's own two amplitudes charged to it (see todcor.window_noise).
        Pass 2 re-weights each order's statistics by 1 / sigma_i^2 (no new
        FFTs: the statistics are linear in the weights, and the per-order fine
        maps are reused when the coarse argmax has not moved) and re-profiles;
        the RVs, amplitudes and errors come from pass 2. An order with fewer
        than ``self.min_noise_dof`` dof, or no residual (e.g. a masked order
        filled with continuum), gets the common estimate instead, with a
        warning. With one order this is the single common noise
        sqrt(chi2_red). One pass leaves the cleaner of two orders of very
        different S/N with a slightly high estimate -- its pass-1 residual
        includes the misfit of a fit pulled by the noisier order: +3 per cent
        at a 10x noise contrast, +0.5 per cent at 3x -- i.e. its errors are
        that much conservative; the RVs are unaffected.

        chi2_red = chi2_best / (sum(nu) - 4) for shared amplitudes, and
        chi2_best / (sum(nu) - 2 - 2 n_orders) for per-order ones. Interpolated noise is smoother
        than the native noise -- a cubic spline keeps ~0.87 of its variance
        when its nodes fall at random between the native pixels -- so dividing
        by the bare n_eff would leave chi2_red
        ~0.8 for honest noise. ``nu`` instead counts the noise power the fit
        actually sees (sparta.preprocess.resampling_noise_psd, from the
        resampler's impulse responses at the sub-pixel phases of the native
        pixels), so chi2_red ~ 1 for honest noise on any analysis step, finer
        or coarser than native. The phases are measured when CCF2d does the
        resampling itself (native positions known); for a spectrum resampled
        beforehand they are taken as uniformly spread. That includes an
        archive product whose native step Observations seeded from the
        instrument (e.g. HARPS ADP, 0.01 A bins over ~0.609 km/s detector
        pixels): it is modelled as the native pixels resampled once, by the
        spline, onto the analysis grid. The archive pipeline's own rebinning
        kernel is unknown, so for such products nu and the noise budget are
        approximate.

        That calibration -- nu, chi2_red, the sigma=None estimate, and the
        native-pixel count behind the CRLB -- assumes white noise on the
        native pixels, resampled once. After ``filter_spectrum`` /
        ``preprocess`` the ``noise_fraction`` correction is approximate: on
        simulated HARPS-like SB2s the errors are ~1.1-1.2x the RV scatter
        (without it they were ~0.55x); an exact error model for filtered
        spectra is an open item (docs/roadmap.md). Calibrate eRV by
        simulation (or by the RV scatter of a constant star) where it
        matters.

        Parameters
        ----------
        spectrum : Spectrum
            The observed, barycentric-corrected spectrum (not modified).
        template1, template2 : Template
            The primary's and the secondary's templates, at rest (not
            modified). A single-order template is reused for every order.
        dv : float or astropy.units.Quantity, optional
            Log-lambda grid step (km/s if a plain number) for a spectrum not
            already on one. Default ``self.default_dv``, 0.1 km/s.
        vel_bound : float or (float, float)
            Trial-velocity range (km/s): ``[-vel_bound, vel_bound]`` if
            scalar, else [min, max]. Used for both components unless explicit
            grids are given. Default 150 km/s, as in CCF1d and Observations.
        dvel : float, optional
            Step of the coarse trial-velocity map (km/s). Default
            ``self.default_dvel``, 1 km/s.
        v1_grid, v2_grid : array_like, optional
            Explicit trial-velocity grids (km/s); a given grid replaces the
            ``vel_bound`` / ``dvel`` one for its component.
        sigma : float or array_like, optional
            Per-pixel noise of the observed flux on its *native* pixels (flux
            units), a scalar (all orders) or one value per order; taken at
            face value. Its order-to-order ratios weight the orders; its
            overall scale sets the uncertainties only. Default None: estimate
            each order's noise from its own residual and weight the orders by
            it (see below). Use None after ``preprocess`` /
            ``filter_spectrum``: the flux is then normalised and band-passed,
            so a native-pixel noise no longer describes it.
        nonneg : bool
            Constrain the amplitudes to be non-negative (physical light
            fractions).
        amplitudes : {'shared', 'per_order'}
            One amplitude pair for all orders (default), or one per order --
            its own light ratio, for two stars of different temperature whose
            ratio changes across the spectrum. Per-order costs two free
            parameters per order; with a constant ratio, shared is tighter.
        orders : sequence of int, optional
            Order indices to use. Default all.
        keep_windows : bool
            Retain the per-order prepared arrays (needed by
            :meth:`best_fit_spectrum`). Set False when fitting a long time
            series: the arrays are the size of the spectrum, so keeping them
            for every epoch is what dominates memory.
        velocity_convention : {'cz', 'relativistic'}
            Doppler convention of the trial velocities and the RVs (see
            ``unicor.dsp.doppler_factor``). A component at v is shifted by
            c ln(1 + v/c) ('cz'; IAU 2000, IRAF fxcor) or c artanh(v/c)
            ('relativistic') along the log-lambda axis, measured in each
            order's actual ln(lambda) step. For the same spectrum the 'cz'
            RVs exceed the 'relativistic' ones by ~v^2/2c (37.5 m/s at
            150 km/s).
        output : {'ccf', 'loglike'}
            The primary map, Corr2d['map'], drawn by ``plot``: the
            two-dimensional correlation R (default) or the profile
            log-likelihood. R = sqrt(1 - chi2 / SD) is the fraction of the
            data's (weighted) power that the best-fit amplitude pair explains
            at each node -- Zucker & Mazeh's (1994) R with the light ratio
            fitted -- in [0, 1] and free of the noise scale. At fixed data it
            is monotonic in the log-likelihood, logL = -(SD/2) (1 - R^2), so
            both maps share their argmax: the RVs, amplitudes and errors
            (which come from the log-likelihood curvature) are identical in
            both modes. Both maps are stored either way, as they cost one
            element-wise operation each.

        Returns
        -------
        CCF2d
            self, with these attributes:

            ``Corr2d``
                dict with the coarse map: 'v1_grid', 'v2_grid' (km/s), 'R'
                (the two-dimensional correlation), 'logL_profile',
                'logL_marginal', 'a1', 'a2', 'chi2' (all (n1, n2) maps;
                under ``amplitudes='per_order'`` 'a1' / 'a2' are
                (n_orders, n1, n2), kept only with ``keep_windows``, else
                None),
                'map' (whichever of 'R' / 'logL_profile' ``output`` selects)
                and 'output', 'best' (its argmax node) and 'units'; 'fine',
                the same for the fine local map (None if refinement was
                skipped); and the results: 'RV1', 'RV2' (km/s, refined),
                'eRV1', 'eRV2' (km/s), 'rho12' (v1-v2 correlation), 'cov'
                (2x2, km^2/s^2), 'peak' (the coarse argmax node, km/s),
                'a1_best', 'a2_best' (one value per window, aligned with
                ``windows``, under ``amplitudes='per_order'``) and 'R_best'
                (at the refined RVs), 'amplitudes', 'sigma' (the native
                per-pixel noise behind the fit and its errors: the input as
                given, or under sigma=None an array of the per-window
                estimates, aligned with ``windows``), 'sigma_from_fit' (bool:
                True when the noise was estimated, i.e. sigma=None), the
                misfit at the refined RVs at that noise, 'chi2_best', and
                'chi2_red' (chi2_best per noise degree of freedom, see below;
                ~1 by construction under sigma=None), and
                'velocity_convention' (that of every velocity above). The maps
                ('logL_profile', 'logL_marginal', 'chi2', and those under
                'fine') are on the same noise scale; 'R' does not depend on
                it, only on the orders' relative weights.
            ``stats``
                The *combined* coarse-map sufficient statistics, on that noise
                scale, enough to re-profile (e.g. with a different
                ``nonneg``) without touching the spectrum again. They pool
                the orders with shared amplitudes: under
                ``amplitudes='per_order'`` re-profiling them gives the shared
                fit, not the one reported.
            ``windows``
                Per-order diagnostics: the order index, the window's own
                'S11' / 'S22' (the templates' power over the window, median
                over the trial velocities) / 'SD' (at the input sigma, unit
                noise under sigma=None), 'dv' (its analysis step, c times the actual
                ln(lambda) step, km/s), 'oversampling' (dv_native / dv, at
                least 1, divided by the noise fraction), 'n_eff' (the
                independent-pixel count n_pix / oversampling), 'nu' (the expected chi2 of its pure
                noise), 'sigma' (its noise: input or estimated),
                'sigma_fallback' (True if sigma=None could not estimate it and
                used the common estimate), and (if ``keep_windows``) the
                prepared arrays.
            ``n_ord``
                The number of pooled windows.
            ``crlb``
                The ``todcor.crlb_from_map`` output behind the RVs.

        Raises
        ------
        ValueError
            If ``output`` is unknown, the grid is not uniform in ln(lambda)
            with the declared step, ``dvel`` is not positive, a flux is not
            finite, a template has no lines over the fitted orders, or the
            noise cannot be estimated.

        Warns
        -----
        UserWarning
            If the peak lies on the edge of the coarse or the fine map, or an
            order falls back to the common noise estimate.
        '''
        check_output(output)
        spec = deepcopy(spectrum)
        template1 = deepcopy(template1)
        template2 = deepcopy(template2)

        if dv is None:
            dv = self.default_dv
        elif not isinstance(dv, u.quantity.Quantity):
            dv = float(dv) << u.kilometer / u.second

        native = spec.native_steps()
        # A band-pass keeps only part of the native noise (and of the
        # information): count that share of the native pixels, as CCF1d does.
        noise_fraction = np.asarray(spec.Info.get('NoiseFraction', [1.0] * len(spec.wv)), float)
        # Where the native pixels sit is known only if nothing has resampled
        # the spectrum yet: then they are its current grid.
        native_wv = None if 'NativeStep' in spec.Info else [np.asarray(w, float) for w in spec.wv]

        # The Fourier shift theorem holds only on a velocity-even grid.
        if spec.Info.get('GridType') != 'log':
            spec.interpolate(delta=dv, method='log')
        resampled = 'NativeStep' in spec.Info
        if 'GridDelta' not in spec.Info:
            raise ValueError(
                "the spectrum claims a log-lambda grid but carries no "
                "'GridDelta'; set Info['GridDelta'] to its velocity step "
                "(km/s) or let Spectrum.interpolate build the grid.")
        dv_kms = u.Quantity(spec.Info['GridDelta']).to(u.km / u.s).value
        # The shift s = c ln(D(v)) is measured in each order's actual step:
        # Spectrum.interpolate's ln(1 + dv/c) is dv/2c (relative) short of the
        # nominal dv, which would scale every velocity by 1 + dv/2c.
        steps = [dsp.log_grid_step(w) for w in spec.wv]
        for I, step in enumerate(steps):
            if not np.isclose(step, dv_kms, rtol=1e-3):
                raise ValueError(
                    f"order {I}: the wavelength grid step is {step:.6g} km/s but "
                    f"Info['GridDelta'] declares {dv_kms:.6g} km/s")

        v1_grid, v2_grid = self._velocity_grids(vel_bound, dvel, v1_grid, v2_grid)

        if orders is None:
            orders = range(len(spec.wv))
        sigma_from_fit = sigma is None
        sigma_in = 1.0 if sigma_from_fit else sigma
        sigma = np.broadcast_to(np.atleast_1d(np.asarray(sigma_in, float)),
                                (len(spec.wv),))

        # The prepared arrays are needed until the fine pass and the evaluation
        # at the refined peak are done; only keep_windows decides whether they
        # outlive this call.
        if amplitudes not in ('shared', 'per_order'):
            raise ValueError(f"amplitudes must be 'shared' or 'per_order', not {amplitudes!r}")
        # The largest |velocity| any window is evaluated at: the coarse grids
        # plus the fine map around their edge nodes.
        reach = self.refine_span * max(np.max(np.diff(v1_grid), initial=0.0),
                                       np.max(np.diff(v2_grid), initial=0.0))
        v_max = max(np.max(np.abs(v1_grid)), np.max(np.abs(v2_grid))) + reach
        prepared, stats, windows = [], [], []
        for I in orders:
            try:
                win = prepare_window(spec.wv[I], spec.sp[I],
                                     *_template_order(template1, I),
                                     *_template_order(template2, I), v_max=v_max)
            except ValueError as err:
                raise ValueError(f"order {I}: {err}") from err
            w, step = win.wave, win.dv
            ratio = native[I] / step
            oversampling = max(1.0, ratio) / noise_fraction[I]
            if resampled:
                phases = None if native_wv is None else native_phases(native_wv[I], w, step)
                psd = resampling_noise_psd(len(w), native[I], step, phases)
            else:
                psd = np.ones(len(w) // 2)
            kw = dict(sigma=sigma[I], oversampling=oversampling, noise_psd=psd,
                      velocity_convention=velocity_convention, reach=win.reach)
            st = window_stats(win.data_g, win.g1, win.g2, step, v1_grid, v2_grid, **kw)
            stats.append(st)
            prepared.append((win, kw))

            # Keep the scalar powers always -- they say which component each
            # order can constrain, and they cost nothing. The full arrays are
            # optional (see keep_windows).
            diag = {'order': I, 'dv': step,
                    'S11': float(np.median(st['S11'])), 'S22': float(np.median(st['S22'])),
                    'SD': st['SD'],
                    'oversampling': oversampling, 'n_eff': len(w) / oversampling,
                    'nu': st['nu'], 'sigma': float(sigma[I]), 'sigma_fallback': False}
            if keep_windows:
                diag.update(wv=w, data_g=win.data_g, g1=win.g1, g2=win.g2)
            windows.append(diag)

        # Only *pooled* statistics outlive this call: each window's A12 is an
        # (n1, n2) array, so keeping one per order would dominate memory on a
        # many-order spectrum. The per-window list lives until the re-weighted
        # pass (sigma=None) has re-pooled it.
        pooled = combine_stats(stats)
        # A template with no lines over the fitted orders (prepare_window
        # zeroes a flat one) leaves its velocity unconstrained: the map is flat
        # along it and its argmax meaningless.
        for k, name in (('S11', 'template1'), ('S22', 'template2')):
            if not np.max(pooled[k]) > 0:
                raise ValueError(
                    f"{name} has no spectral structure over the fitted orders "
                    f"(max {k} = {np.max(pooled[k]):.3g}); check its wavelength coverage "
                    "and air/vacuum frame against the observation")

        # Per-window fine-map statistics, keyed by the coarse node they are
        # centred on: the re-weighted pass reuses them if its coarse argmax
        # has not moved.
        fit_kw = dict(nonneg=nonneg, output=output, amplitudes=amplitudes, half=self.crlb_half,
                      span=self.refine_span, factor=self.refine_factor, fine_cache={})
        fit = joint_fit(stats, None, prepared, v1_grid, v2_grid, **fit_kw)
        noise = sigma_in
        if sigma_from_fit:
            noise, fallback = window_noise(fit.window_chi2, [st['nu'] for st in fit.at_peak],
                                           min_dof=self.min_noise_dof, amplitudes=amplitudes)
            if fallback.any():
                bad = [diag['order'] for diag, fb in zip(windows, fallback) if fb]
                warnings.warn(
                    f"order(s) {bad}: too few noise degrees of freedom (< "
                    f"{self.min_noise_dof}) or no residual to measure their own noise; "
                    "using the common noise estimate for them", stacklevel=2)
            for diag, s_i, fb in zip(windows, noise, fallback):
                diag.update(sigma=float(s_i), sigma_fallback=bool(fb))
            fit = joint_fit(stats, 1.0 / noise ** 2, prepared, v1_grid, v2_grid, **fit_kw)

        coarse, crlb, at_peak = fit.coarse, fit.crlb, fit.best
        res = dict(fit.profile)
        if amplitudes == 'per_order' and not keep_windows:
            # One (n1, n2) amplitude map per order: dropped like the windows'
            # arrays, as they would dominate memory on a many-order spectrum.
            res['a1'] = res['a2'] = None
        res['units'] = u.km / u.s
        res['fine'] = fit.fine
        if coarse['on_edge']:
            warnings.warn("the TODCOR peak lies on the edge of the velocity grid; "
                          "the RVs are grid-limited and their errors are NaN. "
                          "Widen vel_bound (or the explicit grids).", stacklevel=2)
        elif res['fine'] is None:
            warnings.warn("the refined TODCOR peak lies outside the fine map "
                          f"(+/-{self.refine_span} coarse steps around the coarse "
                          "argmax); reporting the coarse-map estimate, whose "
                          "sub-grid refinement may be biased.", stacklevel=2)

        rv1, rv2 = crlb['peak_refined']
        res.update(RV1=rv1, RV2=rv2,
                   eRV1=crlb['sigma_v1'], eRV2=crlb['sigma_v2'],
                   rho12=crlb['rho'], cov=crlb['cov'],
                   peak=coarse['peak'],
                   a1_best=at_peak['a1'], a2_best=at_peak['a2'], R_best=at_peak['R'],
                   chi2_best=at_peak['chi2'], chi2_red=at_peak['chi2_red'],
                   amplitudes=amplitudes,
                   sigma=noise, sigma_from_fit=sigma_from_fit,
                   velocity_convention=velocity_convention)

        self.stats = fit.pooled
        self.windows = windows
        self.n_ord = len(windows)
        self.Corr2d = res
        self.crlb = crlb
        return self

    # =============================================================================
    # =============================================================================
    def best_fit_spectrum(self, order=0):
        '''
        The best-fit composite flux for one window, for residual inspection.

        Uses the fit's own model -- the extended templates shifted to the
        refined velocities, scaled by the amplitudes, plus the window's
        fitted continuum offset -- on the window's observed pixels.

        Parameters
        ----------
        order : int
            Index into ``self.windows`` (not the spectrum's own order
            numbering, which is ``self.windows[i]['order']``).

        Returns
        -------
        wv : ndarray
            Wavelengths (Angstrom).
        model : ndarray
            The modelled composite (continuum 1).
        data : ndarray
            The window's data (its median removed) + 1.

        Raises
        ------
        ValueError
            If the fit ran with ``keep_windows=False``.
        '''
        win = self.windows[order]
        if 'wv' not in win:
            raise ValueError(
                'the prepared arrays were not retained; re-run '
                'cross_correlate with keep_windows=True to inspect residuals')
        r = self.Corr2d
        a1, a2 = r['a1_best'], r['a2_best']
        if np.ndim(a1):                       # amplitudes='per_order': this window's
            a1, a2 = a1[order], a2[order]
        model = model_flux(win['g1'], win['g2'], len(win['wv']), win['dv'],
                           r['RV1'], r['RV2'], a1, a2,
                           velocity_convention=r['velocity_convention'])
        # The window's continuum offset is a free parameter of the fit.
        model = model + np.mean(win['data_g'] - model)
        return win['wv'], model + 1.0, win['data_g'] + 1.0

    # =============================================================================
    # =============================================================================
    def plot(self, marginals=True, figsize=(6.5, 6.0)):
        '''
        Plot the primary (v1, v2) map as a grey-scale image, peak black.

        ``output='ccf'`` (the default of ``cross_correlate``) draws the
        two-dimensional correlation R on a linear grey scale from the map's
        minimum (white) to its maximum (black); ``output='loglike'`` draws the
        profile log-likelihood as Delta lnL below its maximum, linear to -4.5
        (3 sigma) and logarithmic beyond (``sparta.plotting.dlnl_norm``), as a
        high-S/N map is ~1e5 deep. A thin grey colour bar gives the scale; the
        best fit is a red ``+`` (white-edged), also marked by red lines on the
        marginals. Add the truth with
        ``sparta.plotting.mark_truth(fig.axes[0], v1, v2)``.

        Parameters
        ----------
        marginals : bool
            Add the two cuts through the peak (R, or Delta lnL, against v1 and
            against v2, km/s), as black lines.
        figsize : (float, float)
            Figure size (inches).

        Returns
        -------
        matplotlib.figure.Figure
        '''
        # The plotting stack loads on the first plot, not on import.
        import matplotlib.pyplot as plt
        from sparta import plotting as sp_plot

        res = self.Corr2d
        v1, v2 = res['v1_grid'], res['v2_grid']
        loglike = res['output'] == 'loglike'
        if loglike:
            # Delta lnL relative to the peak: the absolute level carries the
            # arbitrary SD offset.
            z = res['logL_profile'] - np.nanmax(res['logL_profile'])
            label = sp_plot.DLNL_LABEL
        else:
            z, label = res['R'], '$R$'

        fig = plt.figure(figsize=figsize, layout='constrained')
        if marginals:
            gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4))
            ax = fig.add_subplot(gs[1, 0])
            ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
            ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
            ax_key = fig.add_subplot(gs[0, 1])
            ax_key.axis('off')
        else:
            ax = fig.add_subplot(111)

        # The map is indexed [v1, v2]; transpose so v1 runs along x.
        mesh = sp_plot.map_image(ax, v1, v2, z.T, loglike=loglike)
        (fit_marker,) = sp_plot.mark_fit(ax, res['RV1'], res['RV2'])
        ax.set_xlabel(r'$v_1$ (km s$^{-1}$)')
        ax.set_ylabel(r'$v_2$ (km s$^{-1}$)')

        if marginals:
            i, j = np.unravel_index(np.nanargmax(z), z.shape)
            sp_plot.plot_curve(ax_top, v1, z[:, j])
            sp_plot.plot_curve(ax_right, z[i, :], v2)
            ax_top.axvline(res['RV1'], color=sp_plot.MODEL_COLOR, lw=0.8)
            ax_right.axhline(res['RV2'], color=sp_plot.MODEL_COLOR, lw=0.8)
            ax_top.tick_params(labelbottom=False)
            ax_right.tick_params(labelleft=False)
            short = r'$\Delta\ln L$' if loglike else '$R$'
            ax_top.set_ylabel(short)
            ax_right.set_xlabel(short)
            if loglike:
                # High-S/N maps run to |Delta lnL| ~ 1e5: keep the tick labels short.
                ax_top.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
                ax_right.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
            # Beside the right marginal, so the map keeps the marginals' axes.
            sp_plot.map_colorbar(mesh, label, ax=ax_right)
            ax_key.legend([fit_marker], ['best fit'], loc='center left', fontsize=8,
                          borderaxespad=0)
        else:
            sp_plot.map_colorbar(mesh, label)
            fig.legend([fit_marker], ['best fit'], loc='outside upper left', fontsize=8)

        return fig

    # =============================================================================
    # Internals
    # =============================================================================
    def _velocity_grids(self, vel_bound, dvel, v1_grid, v2_grid):
        '''
        Build the trial-velocity grids (km/s) from the bounds and step, unless
        explicit grids were supplied. Both components share one grid by default,
        which is also what combine_stats requires across windows.

        Parameters
        ----------
        vel_bound, dvel, v1_grid, v2_grid
            See ``cross_correlate``.

        Returns
        -------
        v1_grid, v2_grid : ndarray
            Trial velocities (km/s).
        '''
        # Explicit grids may carry units; anything else is taken as km/s.
        v1_grid, v2_grid = (g.to_value(u.km / u.s) if isinstance(g, u.Quantity) else g
                            for g in (v1_grid, v2_grid))
        if v1_grid is not None and v2_grid is not None:
            return np.asarray(v1_grid, float), np.asarray(v2_grid, float)

        if isinstance(vel_bound, u.quantity.Quantity):
            vel_bound = vel_bound.to(u.km / u.s).value
        bound = np.atleast_1d(np.asarray(vel_bound, float))
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


def _template_order(template, order):
    '''
    One template order's arrays for an observed order.

    A single-order template (e.g. a synthetic library spectrum covering the
    whole range) is reused for every order.

    Parameters
    ----------
    template : Template
    order : int
        The observed order's index.

    Returns
    -------
    wave, flux : ndarray
        The template's wavelengths (Angstrom) and flux for that order.
    '''
    model = template.model
    I = order if len(model.wv) > 1 else 0
    return model.wv[I], model.sp[I]
