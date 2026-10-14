"""CCF2d: TODCOR fit of a double-lined (SB2) Spectrum against two Templates.

The two-component counterpart of UNICOR's ``CCF1d``: both components are fitted
simultaneously on a (v1, v2) grid -- TODCOR recast as a Gaussian profile
likelihood in the Fourier domain. Each spectral order is one TODCOR window;
the windows' sufficient statistics are pooled into one joint fit, a coarse map
locates the peak and a fine local map measures the velocities, amplitudes and
Cramer-Rao uncertainties. The estimator lives in ``todcor.estimator``, the
forward model in ``todcor.simulate``, the grid/FFT/broadening primitives in
``unicor.dsp``; this class handles the SPARTA-object orchestration.

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
from functools import lru_cache
import warnings

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from unicor import dsp
from unicor.ccf import check_output
from todcor.simulate import model_flux
from todcor.estimator import (
    chi2_from_stats, combine_stats, crlb_from_map, profile_from_stats,
    resampled_noise_psd, window_stats,
)
from sparta import plotting as sp_plot
from sparta.spectrum import resample_flux


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
        self.default_dv = 0.1 * u.kilometer / u.second     # log-lambda grid step
        self.default_dvel = 1.0                            # (v1, v2) map step, km/s
        self.refine_span = 2       # fine-map half-width, in coarse steps
        self.refine_factor = 10    # coarse step / fine step
        self.crlb_half = 3         # half-width (nodes) of the quadratic patch
        self.min_noise_dof = 50    # fewest dof to estimate one window's noise

    # =============================================================================
    # =============================================================================
    def cross_correlate(self, spectrum, template1, template2,
                        dv=None, vel_bound=150, dvel=None,
                        v1_grid=None, v2_grid=None, sigma=None, alpha=0.1,
                        nonneg=True, zeta_max=None, continuum='median',
                        orders=None, keep_windows=True, velocity_convention='cz',
                        output='ccf'):
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
        spectrum's ``native_steps()`` (the coarsest sampling it has had before
        this call) give
        ``oversampling = max(1, dv_native / dv)``, and the Fisher weights are
        divided by it. Otherwise the ~``dv_native / dv`` analysis pixels that
        interpolate one native pixel would each count as independent, and the
        CRLB would shrink by ``sqrt(dv_native / dv)``.

        sigma=None runs the joint fit twice. Pass 1 fits every order at unit
        noise and locates the joint peak (coarse and fine maps). Each order's
        noise is then estimated from its own share of the joint chi2 there,
        sigma_i^2 = chi2_i / (nu_i * (1 - 4 / sum(nu))) -- the four shared
        parameters' dof charged in proportion to nu_i (see _window_noise).
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

        chi2_red = chi2_best / (sum(nu) - 4). Interpolated noise is smoother
        than the native noise -- a cubic spline keeps ~0.87 of its variance
        when its nodes fall at random between the native pixels, and the
        taper ~0.94 more -- so dividing by the bare n_eff would leave chi2_red
        ~0.8 for honest noise. ``nu`` instead counts the noise power the fit
        actually sees (todcor.estimator.resampled_noise_psd, from the
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
        native pixels, resampled once. It does not hold after
        ``filter_spectrum`` / ``preprocess`` (band-pass filtering,
        per-order normalisation, a linear resampling before the log one): in
        Monte Carlo on such spectra the RV scatter is only 0.72-0.74 of eRV,
        i.e. the errors are ~1.4x conservative. Calibrate eRV by simulation
        (or by the RV scatter of a constant star) for preprocessed spectra.

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
            it (see below).
        alpha : float
            Tukey apodisation parameter applied to each window. Set 0 to skip
            (e.g. if the spectrum was already tapered by
            ``Spectrum.preprocess``).
        nonneg : bool
            Constrain both amplitudes to be non-negative (physical light
            fractions).
        zeta_max : float, optional
            Low-pass on the Fourier axis (s/km), e.g. ``R / c`` to drop scales
            finer than the resolution.
        continuum : 'median' or float
            Remove each array's median (default), or subtract this fixed
            continuum level. The same is applied to data and templates, which
            is what makes the amplitudes meaningful.
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
                'logL_marginal', 'a1', 'a2', 'chi2' (all (n1, n2) maps),
                'map' (whichever of 'R' / 'logL_profile' ``output`` selects)
                and 'output', 'best' (its argmax node) and 'units'; 'fine',
                the same for the fine local map (None if refinement was
                skipped); and the results: 'RV1', 'RV2' (km/s, refined),
                'eRV1', 'eRV2' (km/s), 'rho12' (v1-v2 correlation), 'cov'
                (2x2, km^2/s^2), 'peak' (the coarse argmax node, km/s),
                'a1_best', 'a2_best' and 'R_best' (at the refined RVs), 'sigma' (the native
                per-pixel noise behind the fit and its errors: the input as
                given, or under sigma=None an array of the per-window
                estimates, aligned with ``windows``), 'sigma_from_fit', the
                misfit at the refined RVs at that noise, 'chi2_best' and
                'chi2_red' (chi2_best per noise degree of freedom, see below;
                ~1 by construction under sigma=None), and
                'velocity_convention' (that of every velocity above). The maps
                ('logL_profile', 'logL_marginal', 'chi2', and those under
                'fine') are on the same noise scale; 'R' does not depend on
                it, only on the orders' relative weights.
            ``stats``
                The *combined* coarse-map sufficient statistics, on that noise
                scale, enough to re-profile (e.g. with a different
                ``nonneg``) without touching the spectrum again.
            ``windows``
                Per-order diagnostics: the order index, the window's own
                'S11' / 'S22' / 'SD' (at the input sigma, unit noise under
                sigma=None), 'dv' (its analysis step, c times the actual
                ln(lambda) step, km/s), 'oversampling' (dv_native / dv, at
                least 1), 'n_eff' (the independent-pixel count
                n_pix / oversampling), 'nu' (the expected chi2 of its pure
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

            step = steps[I]
            ratio = native[I] / step
            oversampling = max(1.0, ratio)
            if resampled:
                phases = None if native_wv is None else _native_phases(native_wv[I], w, step)
                psd = _resampling_noise_psd(len(w), native[I], step, phases)
            else:
                psd = np.ones(len(w) // 2)
            noise_psd = taper_power * psd
            kw = dict(sigma=sigma[I], zeta_max=zeta_max,
                      oversampling=oversampling, noise_psd=noise_psd,
                      velocity_convention=velocity_convention)
            st = window_stats(data_g, g1, g2, step, v1_grid, v2_grid, **kw)
            stats.append(st)
            prepared.append((data_g, g1, g2, step, kw))

            # Keep the scalar powers always -- they say which component each
            # order can constrain, and they cost nothing. The full arrays are
            # optional (see keep_windows).
            win = {'order': I, 'dv': step,
                   'S11': st['S11'], 'S22': st['S22'], 'SD': st['SD'],
                   'oversampling': oversampling, 'n_eff': len(w) / oversampling,
                   'nu': st['nu'], 'sigma': float(sigma[I]), 'sigma_fallback': False}
            if keep_windows:
                win.update(wv=w, data_g=data_g, g1=g1, g2=g2)
            windows.append(win)

        # Only *pooled* statistics outlive this call: each window's A12 is an
        # (n1, n2) array, so keeping one per order would dominate memory on a
        # many-order spectrum. The per-window list lives until the re-weighted
        # pass (sigma=None) has re-pooled it.
        pooled = combine_stats(stats)
        # A template with no lines over the fitted orders makes the 2x2 amplitude
        # system singular, and the map collapses to a flat, meaningless argmax.
        for k, name in (('S11', 'template1'), ('S22', 'template2')):
            if not pooled[k] > 0:
                raise ValueError(
                    f"{name} has no spectral structure over the fitted orders "
                    f"({k} = {pooled[k]}); check its wavelength coverage and "
                    "air/vacuum frame against the observation")

        # Per-window fine-map statistics, keyed by the coarse node they are
        # centred on: the re-weighted pass reuses them if its coarse argmax
        # has not moved.
        fine_cache = {}
        fit = self._joint_fit(stats, None, prepared, v1_grid, v2_grid,
                              nonneg, fine_cache, output)
        noise = sigma_in
        if sigma_from_fit:
            noise, fallback = self._window_noise(fit, windows)
            for win, s_i, fb in zip(windows, noise, fallback):
                win.update(sigma=float(s_i), sigma_fallback=bool(fb))
            fit = self._joint_fit(stats, 1.0 / noise ** 2, prepared,
                                  v1_grid, v2_grid, nonneg, fine_cache, output)

        res, coarse, crlb, at_peak = fit['res'], fit['coarse'], fit['crlb'], fit['best']
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
                   sigma=noise, sigma_from_fit=sigma_from_fit,
                   velocity_convention=velocity_convention)

        self.stats = fit['pooled']
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

        Uses the same Fourier phase-shift model as the fit, evaluated at the
        peak velocities and amplitudes.

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
            The prepared (continuum-removed, tapered) data residual + 1.

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
        model = model_flux(win['wv'], win['dv'], win['g1'], win['g2'],
                           r['RV1'], r['RV2'], r['a1_best'], r['a2_best'],
                           velocity_convention=r['velocity_convention'])
        return win['wv'], model, win['data_g'] + 1.0

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

    def _joint_fit(self, stats, weights, prepared, v1_grid, v2_grid, nonneg, fine_cache,
                   output):
        '''
        One pass of the joint fit: pool the per-window coarse statistics with
        the given weights, locate the peak on the coarse map, refine it on the
        fine map, and evaluate every window exactly at the refined peak.

        Parameters
        ----------
        stats : list of dict
            Per-window coarse-map ``window_stats``.
        weights : array_like or None
            Per-window factors on the statistics (1/sigma_i^2 for unit-noise
            windows), or None for all ones.
        prepared : list of tuple
            Per-window (data_g, g1, g2, dv, window_stats kwargs).
        v1_grid, v2_grid : ndarray
            Coarse trial velocities (km/s).
        nonneg : bool
            See ``cross_correlate``.
        fine_cache : dict
            Per-window fine-map statistics by coarse node, filled here and
            reused across passes.
        output : {'ccf', 'loglike'}
            The primary map of the profile dicts (see ``cross_correlate``).

        Returns
        -------
        dict
            'pooled' (the weighted coarse statistics), 'res' (the coarse
            profile_from_stats dict, plus 'fine': the fine one, or None if
            refinement was skipped or failed), 'coarse' and 'crlb'
            (crlb_from_map on the coarse map and on the map behind the RVs),
            'at_peak' (per-window window_stats at the refined peak) and
            'best' (the joint profile there).
        '''
        pooled = combine_stats(stats, weights)
        res = profile_from_stats(pooled, v1_grid, v2_grid, nonneg=nonneg, output=output)
        res['units'] = u.km / u.s

        coarse = crlb_from_map(res['logL_profile'], v1_grid, v2_grid,
                               half=self.crlb_half)
        res['fine'] = None if coarse['on_edge'] else self._refine(
            prepared, res, nonneg, weights, fine_cache, output)
        crlb = coarse if res['fine'] is None else crlb_from_map(
            res['fine']['logL_profile'], res['fine']['v1_grid'],
            res['fine']['v2_grid'], half=self.crlb_half)

        # Amplitudes and chi2 exactly at the refined velocities, so the
        # reported RVs and amplitudes belong to one point.
        p1, p2 = np.array([crlb['peak_refined'][0]]), np.array([crlb['peak_refined'][1]])
        at_peak = [window_stats(d, g1, g2, dv, p1, p2, **kw)
                   for d, g1, g2, dv, kw in prepared]
        best = profile_from_stats(combine_stats(at_peak, weights), p1, p2,
                                  nonneg=nonneg)['best']
        return dict(pooled=pooled, res=res, coarse=coarse, crlb=crlb,
                    at_peak=at_peak, best=best)

    def _refine(self, prepared, res, nonneg, weights, fine_cache, output):
        '''
        Re-evaluate the statistics on a fine (v1, v2) grid around the coarse
        argmax and profile them -- the second, exact pass (see ``__init__``).
        The per-window fine statistics are cached in ``fine_cache`` by coarse
        node, so a re-weighted pass with the same coarse argmax only re-pools
        them.

        Parameters
        ----------
        prepared, nonneg, weights, fine_cache, output
            See ``_joint_fit``.
        res : dict
            The coarse ``profile_from_stats`` result.

        Returns
        -------
        dict or None
            The fine-map profile_from_stats dict, or None if its argmax lands
            on its own edge -- the coarse argmax was then not close to the
            true peak (e.g. a ridge or a tie), and the coarse map is the more
            honest answer.
        '''
        v1, v2 = res['v1_grid'], res['v2_grid']
        i, j = np.unravel_index(np.argmax(res['logL_profile']), res['logL_profile'].shape)
        # Local steps, so explicit non-uniform grids refine sensibly too.
        step1 = 0.5 * (v1[i + 1] - v1[i - 1])
        step2 = 0.5 * (v2[j + 1] - v2[j - 1])
        nodes = np.arange(-self.refine_span * self.refine_factor,
                          self.refine_span * self.refine_factor + 1) / self.refine_factor
        f1, f2 = v1[i] + step1 * nodes, v2[j] + step2 * nodes

        if (i, j) not in fine_cache:
            fine_cache[(i, j)] = [window_stats(d, g1, g2, dv, f1, f2, **kw)
                                  for d, g1, g2, dv, kw in prepared]
        fine = profile_from_stats(combine_stats(fine_cache[(i, j)], weights), f1, f2,
                                  nonneg=nonneg, output=output)
        fi, fj = np.unravel_index(np.argmax(fine['logL_profile']), fine['logL_profile'].shape)
        if fi in (0, len(f1) - 1) or fj in (0, len(f2) - 1):
            return None
        return fine

    def _window_noise(self, fit, windows):
        '''
        Each window's native per-pixel noise, from its own residual at the
        joint best fit of a unit-noise pass.

        Window i's share of the joint chi2, chi2_i
        (todcor.estimator.chi2_from_stats at the joint amplitudes), is set
        against its noise degrees of freedom:

            sigma_i^2 = chi2_i / (nu_i * (1 - 4 / sum(nu)))

        The four shared parameters (v1, v2, a1, a2) take four dof from the
        joint chi2, charged to the windows in proportion to nu_i. (A window's
        exact charge -- the trace of its block of the hat matrix -- follows its
        line content instead; the two differ by at most 4 / nu_i in sigma_i^2.)
        The charges sum to four, so with one window this is sqrt(chi2_red).

        A window with fewer than ``self.min_noise_dof`` dof (its estimate
        would scatter by more than ~10 per cent), or with no residual to
        measure (chi2_i <= 0, e.g. a masked order filled with continuum), gets
        the common estimate instead, with a warning: sum(chi2_i) / sum(dof_i)
        over the windows that measured their own noise -- a flat order's
        zero chi2 would drag a pooled estimate down -- or over all windows
        (i.e. chi2_red) if none did.

        Parameters
        ----------
        fit : dict
            A unit-noise ``_joint_fit`` result.
        windows : list of dict
            The per-window diagnostics (for the warning).

        Returns
        -------
        sigma : ndarray
            Per-window noise (flux units, per native pixel).
        fallback : ndarray of bool
            The windows that got the common estimate.

        Raises
        ------
        ValueError
            If no positive, finite noise can be estimated.
        '''
        best = fit['best']
        chi2 = np.array([chi2_from_stats(st, best['a1'], best['a2']).item()
                         for st in fit['at_peak']])
        nu = np.array([st['nu'] for st in fit['at_peak']])
        dof = nu * (1.0 - 4.0 / nu.sum())
        own = (dof >= self.min_noise_dof) & np.isfinite(chi2) & (chi2 > 0)
        pool = own if own.any() else np.ones_like(own)
        common = chi2[pool].sum() / dof[pool].sum()
        if not (np.isfinite(common) and common > 0):
            raise ValueError(
                f"cannot estimate the noise from the fit (chi2 per dof = {common}); "
                "pass an explicit sigma")
        fallback = ~own
        if fallback.any():
            bad = [win['order'] for win, fb in zip(windows, fallback) if fb]
            warnings.warn(
                f"order(s) {bad}: too few noise degrees of freedom (< "
                f"{self.min_noise_dof}) or no residual to measure their own noise; "
                "using the common noise estimate for them", stacklevel=3)
        var = np.where(fallback, common, chi2 / np.where(fallback, 1.0, dof))
        return np.sqrt(var), fallback

    @staticmethod
    def _template_on_grid(template, order, wave):
        '''
        Resample one template order onto the observation's wavelength grid.

        A single-order template (e.g. a synthetic library spectrum covering the
        whole range) is reused for every order. Points outside the template's
        coverage are filled with the continuum (1.0) rather than clamped to the
        edge value, so an uncovered margin contributes no spurious lines.

        Parameters
        ----------
        template : Template
        order : int
            The observed order's index.
        wave : ndarray
            The observed order's wavelengths (Angstrom).

        Returns
        -------
        ndarray
            The template flux on ``wave`` (linear interpolation).
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
        high-pass filtered by ``Spectrum.preprocess``. Pass a float to override.
        The continuum must be removed *before* apodisation: dropping the DC
        Fourier mode removes a constant offset for free, but not a
        window-shaped one.

        Parameters
        ----------
        flux : array_like
        continuum : 'median' or float

        Returns
        -------
        ndarray
            ``flux`` minus the continuum level.
        '''
        flux = np.asarray(flux, float)
        level = np.median(flux) if continuum == 'median' else float(continuum)
        return flux - level


# =============================================================================
# The resampled-noise model
# =============================================================================
def _resampling_noise_psd(n, dv_native, dv, phases=None, n_phase=32):
    '''
    Noise spectrum (todcor.estimator.resampled_noise_psd) of one order's white
    native noise after Spectrum.interpolate onto a log grid of step
    ``dv``, finer or coarser than native.

    Each native pixel's impulse response depends on its sub-pixel phase
    relative to the analysis nodes, so the responses at ``n_phase`` phases
    (:func:`_impulse_responses`) are averaged with weights counting the
    order's native pixels at each phase. With unknown native positions the
    phases are taken as uniformly spread -- true for a linear-wavelength
    native grid, whose velocity step drifts across the order, but not for a
    commensurate log grid (e.g. one exactly half as fine), whose few fixed
    phases can change the noise power by ~15 per cent.

    This models a single resampling from a uniform native grid; earlier
    filtering or repeated resampling is not represented.

    Parameters
    ----------
    n : int
        Length of the analysis window (pixels).
    dv_native : float
        The order's native velocity step (km/s).
    dv : float
        The analysis step (km/s).
    phases : array_like, optional
        Sub-pixel phases in [0, 1) of the order's native pixels on the
        analysis grid (analysis pixels); None for uniformly spread.
    n_phase : int
        Number of phase bins probed.

    Returns
    -------
    ndarray
        Relative noise power per positive Fourier mode, length n // 2.
    '''
    # Rounded keys let near-identical steps (the same grid, epoch after
    # epoch) share one probe.
    responses = _impulse_responses(round(float(dv_native), 9), round(float(dv), 9), n_phase)
    weights = None
    if phases is not None:
        bins = np.floor(np.asarray(phases) * n_phase).astype(int) % n_phase
        weights = np.bincount(bins, minlength=n_phase)
    return resampled_noise_psd(responses, n, dv_native / dv, weights=weights)


@lru_cache(maxsize=256)
def _impulse_responses(dv_native, dv, n_phase=32, half=20):
    '''
    Analysis-grid responses of the Spectrum.interpolate resampler to a unit
    impulse in one native pixel, at ``n_phase`` sub-pixel phases.

    The native grid is synthetic: log-uniform with step ``dv_native``, with
    impulses ``2 * half`` pixels apart. Impulse k is evaluated at analysis
    nodes placed so that it sits (k + 0.5) / n_phase analysis pixels above the
    node below it, out to ``half`` native pixels either side -- far beyond the
    cubic spline's support (its cardinal function decays as 0.27 per pixel).
    One spline evaluation serves every phase: the nodes of all impulses are
    concatenated. The responses depend on the two steps only (a log grid is
    the same at every wavelength), so they are cached on them.

    Parameters
    ----------
    dv_native : float
        Native velocity step (km/s).
    dv : float
        Analysis velocity step (km/s).
    n_phase : int
        Number of phases.
    half : int
        Native pixels kept either side of each impulse.

    Returns
    -------
    tuple of ndarray
        n_phase read-only responses (analysis pixels).
    '''
    lam0 = 5000.0                            # Angstrom; the response is scale-free
    step_native, step = dv_native / dsp.C_KMS, dv / dsp.C_KMS
    hits = half + 2 * half * np.arange(n_phase)
    ln_native = np.log(lam0) + step_native * np.arange(2 * half * n_phase + 1)
    flux = np.zeros(ln_native.size)
    flux[hits] = 1.0

    reach = half * dv_native / dv            # analysis pixels within `half` native ones
    nodes = []
    for k, j in enumerate(hits):
        phase = (k + 0.5) / n_phase
        m = np.arange(np.floor(phase - reach) + 1, np.ceil(phase + reach))
        nodes.append(ln_native[j] + (m - phase) * step)
    response = resample_flux(np.exp(ln_native), flux, np.exp(np.concatenate(nodes)))
    out = np.split(response, np.cumsum([len(x) for x in nodes])[:-1])
    for h in out:
        h.flags.writeable = False
    return tuple(out)


def _native_phases(native_wave, wave, dv):
    '''
    Sub-pixel phase of each native pixel on a log analysis grid.

    Parameters
    ----------
    native_wave : array_like
        The order's native wavelengths (Angstrom).
    wave : ndarray
        The log analysis grid (Angstrom), step ``dv``.
    dv : float
        Analysis velocity step (km/s).

    Returns
    -------
    ndarray
        Phases in [0, 1) (analysis pixels) of the native pixels inside the
        analysis grid.
    '''
    x = dsp.C_KMS * np.log(np.asarray(native_wave, float) / wave[0]) / dv
    return np.mod(x[(x >= 0) & (x <= len(wave) - 1)], 1.0)
