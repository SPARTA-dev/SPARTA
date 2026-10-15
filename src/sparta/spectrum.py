"""Spectrum: one observed (multi-order) spectrum and its preprocessing.

``wv`` and ``sp`` are lists holding one 1-D wavelength and flux array per
order (orders may differ in length), plus metadata: time of observation,
barycentric velocity correction, target name and the grid bookkeeping in
``Info``. The methods resample (linear or log-lambda), trim, filter, apodise,
clip cosmics, apply the barycentric correction and select orders; ``orders`` /
``from_spectrum1d`` convert to and from specutils ``Spectrum1D``. The
preprocessing methods loop over the per-order functions of
``sparta.preprocess``.

Conventions: wavelengths in Angstrom, velocities in km/s, times in BJD (days).
Cross-correlation needs a log-lambda (velocity-even) grid, step
``log(1 + dv/c)`` -- see ``interpolate(method='log')``.
"""

import numpy as np
from astropy import constants as consts, units as u
from unicor import dsp

from sparta.preprocess import (
    bandpass, coarsest_native_step, remove_cosmics, resample_flux, resample_to_linear_grid,
    resample_to_log_grid, trim_margins,
)


class Spectrum:
    '''
    One observed spectrum: per-order wavelength and flux arrays plus metadata.

    The preprocessing methods work in place on ``wv`` / ``sp`` and return
    ``self``, so they chain.

    Attributes
    ----------
    wv, sp : list of ndarray
        Per-order wavelengths (Angstrom) and fluxes.
    bjd : float
        Time of observation (BJD, days).
    bcv : astropy.units.Quantity
        Barycentric velocity correction (km/s); 0 once applied.
    name : str
        Target name.
    n_ord : int
        Number of orders.
    Info : dict
        Grid bookkeeping: 'GridType' ('raw', 'linear' or 'log'),
        'GridDelta', 'GridUnits', 'WavelengthUnits', 'TargetName' and, once
        resampled, 'NativeStep' (km/s per order, see ``native_steps``).
    c : astropy.units.Quantity
        The speed of light (km/s).
    '''

    # =============================================================================
    # =============================================================================
    def __init__(self, wv=None, sp=None, wave_unit=1.0*u.Angstrom,
                 bjd=0.0, bcv=0.0, name='Anonymous'):
        '''
        Build a Spectrum from per-order wavelength and flux lists.

        The lists are stored as given (not copied).

        Parameters
        ----------
        wv : list of array_like, optional
            1-D wavelength arrays (Angstrom), one per order. Default: none.
        sp : list of array_like, optional
            1-D flux arrays, matching ``wv`` order by order. Default: none.
        wave_unit : astropy.units.Quantity
            Unit of the wavelengths (default Angstrom).
        bjd : float
            Barycentric Julian date of the observation (days).
        bcv : float or astropy.units.Quantity
            Barycentric velocity correction (km/s if a plain number).
        name : str
            Name of the target.

        Raises
        ------
        ValueError
            If ``wv`` and ``sp`` hold different numbers of orders.
        '''
        wv = [] if wv is None else wv
        sp = [] if sp is None else sp
        if len(wv) != len(sp):
            raise ValueError(f"wv has {len(wv)} orders but sp has {len(sp)}")
        self.c = consts.c.to('km/s') # The speed of light in km/sec
        self.Info = {'GridType': 'raw',
                     'WavelengthUnits': wave_unit,
                     'TargetName': name}
        self.wv = wv
        self.sp = sp
        self.bjd = bjd
        self.bcv = bcv
        self.name = name
        self.n_ord = len(self.sp)

        if type(bcv) is not u.quantity.Quantity:
            self.bcv = self.bcv*u.kilometer/u.second

    # =============================================================================
    # =============================================================================
    @property
    def orders(self):
        '''
        The spectrum as a specutils ``SpectrumList``, one ``Spectrum1D`` per order.

        This is the interchange representation for the specutils ecosystem. The
        working ``.wv`` / ``.sp`` lists remain the mutable store used by the
        preprocessing methods; ``orders`` is built from them on access.

        Returns
        -------
        specutils.SpectrumList
            Each order with a real ``SpectralAxis`` (Angstrom) and
            dimensionless flux. Per-order ``meta`` carries this Spectrum's
            ``Info`` plus ``bjd``/``bcv``/``name``; a recorded ``NativeStep``
            is split into the order's own value (km/s), so
            ``from_spectrum1d`` can restore it.
        '''
        from specutils import Spectrum as Spec1D, SpectrumList

        wunit = self.Info.get('WavelengthUnits', 1.0 * u.Angstrom)
        wunit = getattr(wunit, 'unit', wunit)
        native = self.native_steps() if 'NativeStep' in self.Info else None
        out = []
        for I, (w, s) in enumerate(zip(self.wv, self.sp)):
            meta = dict(self.Info)
            meta.update(bjd=self.bjd, bcv=self.bcv, name=self.name)
            if native is not None:
                meta['NativeStep'] = native[I]
            out.append(Spec1D(flux=np.asarray(s, float) * u.dimensionless_unscaled,
                              spectral_axis=np.asarray(w, float) * wunit,
                              meta=meta))
        return SpectrumList(out)

    # =============================================================================
    # =============================================================================
    @classmethod
    def from_spectrum1d(cls, spectra, bjd=0.0, bcv=0.0, name='Anonymous'):
        '''
        Build a ``Spectrum`` from one or more specutils ``Spectrum1D`` objects.

        The sampling bookkeeping that ``orders`` writes is restored, so the
        error estimates still count the native pixels: the grid description
        (``GridType``, ``GridDelta``, ``GridUnits``) when every order agrees on
        it (otherwise the grid is taken as 'raw', to be resampled), and the
        per-order ``NativeStep`` (km/s; an order without one counts its own
        grid as native).

        Parameters
        ----------
        spectra : specutils.Spectrum1D, list of them, or specutils.SpectrumList
            Each object becomes one order; its spectral axis is converted to
            Angstrom.
        bjd : float
            Time of observation (BJD, days), used unless the first object's
            ``meta`` has 'bjd'.
        bcv : float or astropy.units.Quantity
            Barycentric velocity correction (km/s), used unless the first
            object's ``meta`` has 'bcv'.
        name : str
            Target name, used unless the first object's ``meta`` has 'name'.

        Returns
        -------
        Spectrum
        '''
        from specutils import Spectrum as Spec1D

        if isinstance(spectra, Spec1D):
            spectra = [spectra]
        wv = [np.asarray(o.spectral_axis.to(u.Angstrom).value, float) for o in spectra]
        sp = [np.asarray(o.flux.value, float) for o in spectra]
        metas = [getattr(o, 'meta', None) or {} for o in spectra]
        meta = metas[0]
        spec = cls(wv=wv, sp=sp,
                   bjd=meta.get('bjd', bjd),
                   bcv=meta.get('bcv', bcv),
                   name=meta.get('name', name))

        grid_keys = ('GridType', 'GridDelta', 'GridUnits')
        if 'GridType' in meta and all(_same(m.get(k), meta.get(k))
                                      for m in metas[1:] for k in grid_keys):
            spec.Info.update({k: meta[k] for k in grid_keys if k in meta})
        if any('NativeStep' in m for m in metas):
            spec.Info['NativeStep'] = [float(m.get('NativeStep', dsp.dv_from_wave(w)))
                                       for m, w in zip(metas, wv)]
        return spec

# =============================================================================
# =============================================================================
    def native_steps(self):
        '''
        Velocity step (km/s) of each order *as originally sampled*.

        Resampling onto a finer grid adds samples but no information:
        neighbouring resampled pixels share the same noise. Error estimates
        (UNICOR's ML uncertainty, TODCOR's CRLB) therefore need the native
        sampling, which ``interpolate`` records (and coarsens, if a
        resampling is coarser than anything before it) in
        ``Info['NativeStep']``. A spectrum that was never resampled is taken
        to be honestly sampled on its current grid.

        Returns
        -------
        list of float
            One step per order (km/s): the recorded ``Info['NativeStep']``,
            else c * median(d(lambda)/lambda) of the order's current grid.

        Raises
        ------
        ValueError
            If ``Info['NativeStep']`` does not hold one finite, positive value
            per current order (e.g. orders were added or dropped without
            updating it).
        '''
        if 'NativeStep' not in self.Info:
            return [dsp.dv_from_wave(w) for w in self.wv]
        steps = [float(x) for x in self.Info['NativeStep']]
        if len(steps) != len(self.wv):
            raise ValueError(f"Info['NativeStep'] holds {len(steps)} values for "
                             f"{len(self.wv)} orders; it must have one per order")
        if not all(np.isfinite(x) and x > 0 for x in steps):
            raise ValueError("Info['NativeStep'] must be finite and positive (km/s), "
                             f"got {steps}")
        return steps

# =============================================================================
# =============================================================================
    def preprocess(self, n_trim=10, clean_margins=True, remove_nans=True,
                   delta=0.5, cosmic_sigma=3, continuum_scale_kms=2000.0,
                   noise_scale_kms=4.0, alpha=0.3):
        '''
        Apply the standard preprocessing chain, in place.

        In the recommended order: ``trim``, a linear-wavelength
        ``interpolate``, ``remove_cosmics``, ``filter_spectrum``
        (first-order Butterworth band-pass) and ``apply_cosine_bell``. The
        filter standardizes each order (zero mean, unit variance), so the
        flux is no longer continuum-normalized, and the error estimates of
        UNICOR and TODCOR are no longer calibrated afterwards (see
        ``CCF1d.cross_correlate``).

        Parameters
        ----------
        n_trim, clean_margins, remove_nans
            See ``trim``.
        delta : float
            Linear resampling step, in units of each order's median wavelength
            step (see ``interpolate``).
        cosmic_sigma : float
            Upward outlier threshold, in mad_std (see ``remove_cosmics``).
        continuum_scale_kms, noise_scale_kms : float
            The band-pass edges as velocity periods (km/s; see
            ``filter_spectrum``). The defaults, 2000 and 4 km/s, are the
            recommendation for HARPS (R = 120,000).
        alpha : float
            Tukey shape parameter (see ``apply_cosine_bell``).

        Returns
        -------
        Spectrum
            self.
        '''

        self.trim(n_trim=n_trim, clean_margins=clean_margins, remove_nans=remove_nans)
        self.interpolate(delta=delta)
        self.remove_cosmics(sigma_upper=cosmic_sigma)
        self.filter_spectrum(continuum_scale_kms=continuum_scale_kms,
                             noise_scale_kms=noise_scale_kms, order=1)
        self.apply_cosine_bell(alpha=alpha)

        return self

    # =============================================================================
    # =============================================================================
    def interpolate(self, delta=None, method='linear'):
        '''
        Resample every order onto an evenly spaced wavelength grid, in place.

        The flux is resampled with the cubic spline of ``resample_flux``. Each
        new grid lies strictly inside its order: the linear one starts one new
        step above the first wavelength, the log one one median input step
        above it. ``Info`` records the grid ('GridType', 'GridDelta',
        'GridUnits') and the native sampling ('NativeStep', see
        ``native_steps``).

        Parameters
        ----------
        delta : float or astropy.units.Quantity, optional
            For ``method='linear'``: a plain number (or a dimensionless
            Quantity) is the step in units of each order's *median*
            wavelength step (default 0.5); a length Quantity is the step
            itself (``0.25 * u.AA``), the same in every order. ``Info``
            records it as given, with 'GridUnits' 'median step' or
            'wavelength'. For ``method='log'``: the velocity step (km/s if a
            plain number; default 0.1 km/s); the ln(lambda) step is
            ln(1 + delta/c).
        method : {'linear', 'log'}
            Evenly spaced in wavelength, or in ln(lambda) (velocity-even, as
            the CCF requires).

        Returns
        -------
        Spectrum
            self, with ``wv`` / ``sp`` on the new grids.

        Raises
        ------
        ValueError
            For an unknown ``method``, or a ``method='linear'`` step that is
            not a single positive, finite number or length.
        '''
        if method not in ('linear', 'log'):
            raise ValueError(f"method must be 'linear' or 'log', not {method!r}")

        for I, w in enumerate(self.wv):
            if len(w) < 2:
                raise ValueError(f"order {I} has {len(w)} pixels and cannot be resampled; "
                                 "it was trimmed away or a template does not cover it "
                                 "(drop it with select_orders)")

        wnt = self.Info['WavelengthUnits']

        # Remember the sampling that sets the noise scale, so later error
        # estimates can count independent pixels (native_steps).
        previous = self.native_steps()

        if method == 'linear':
            if delta is None:
                delta = 0.5
            if not (np.ndim(delta) == 0 and np.isfinite(u.Quantity(delta).value)
                    and u.Quantity(delta).value > 0):
                raise ValueError(f"method='linear' needs one positive, finite step, not {delta!r}")
            if isinstance(delta, u.Quantity) and delta.unit.is_equivalent(u.AA):
                # An absolute step: the atom takes it as a multiple of each
                # order's own median step.
                step = delta.to_value(wnt)
                resampled = [resample_to_linear_grid(w, s, step / np.median(np.diff(w)))
                             for w, s in zip(self.wv, self.sp)]
                grid_units = 'wavelength'
            else:
                if isinstance(delta, u.Quantity):
                    if delta.unit.physical_type == 'speed':
                        raise ValueError(f"a velocity step ({delta}) belongs to method='log'; "
                                         "method='linear' takes a number (a multiple of each "
                                         "order's median step) or a length")
                    if not delta.unit.is_equivalent(u.dimensionless_unscaled):
                        raise ValueError(f"method='linear' takes a number or a length as its "
                                         f"step, not {delta}")
                    delta = delta.to_value(u.dimensionless_unscaled)
                delta = float(delta)
                resampled = [resample_to_linear_grid(w, s, delta)
                             for w, s in zip(self.wv, self.sp)]
                grid_units = 'median step'
        else:
            if delta is None:
                delta = 0.1 * u.kilometer / u.second
            elif type(delta) is not u.quantity.Quantity:
                delta = float(delta) * u.kilometer / u.second
            dv_kms = delta.to_value(u.kilometer / u.second)
            resampled = [resample_to_log_grid(w, s, dv_kms)
                         for w, s in zip(self.wv, self.sp)]
        wvI = [w for w, _ in resampled]
        spI = [s for _, s in resampled]

        self.Info['NativeStep'] = [coarsest_native_step(old, new)
                                   for old, new in zip(previous, wvI)]

        # Replace the data with the interpolated wavelengths
        self.wv, self.sp = wvI, spI

        # Document the type of grid that is currently
        # saved with the function.
        if method == 'log':
            self.Info['GridType'], self.Info['GridDelta'], self.Info['GridUnits'] = 'log', delta, 'velocity'
        else:
            self.Info['GridType'], self.Info['GridDelta'], self.Info['GridUnits'] = 'linear', delta, grid_units

        return self

    # =============================================================================
    # =============================================================================
    def trim(self, n_trim=None, n_trim_left=0, n_trim_right=0,
             clean_margins=False, remove_nans=True, margin_fraction=0.1,
             margin_window=15):
        '''
        Trim the edges of every order, in place.

        Echelle orders are often padded at their ends: either with exact
        zeros, or -- as in the NRES pipeline -- with unexposed, background-
        subtracted detector that reads zero plus noise. Those margins must go
        *before* the continuum is removed: filtering a jump from ~0 to the
        continuum level turns it into a step at the order edge.

        Parameters
        ----------
        n_trim : int, optional
            Pixels to remove from each end; overrides ``n_trim_left`` /
            ``n_trim_right``.
        n_trim_left, n_trim_right : int
            Pixels to remove from the blue / red end, counted inside the
            margins when ``clean_margins`` is set.
        clean_margins : bool
            Drop the padded margins: the leading and trailing runs whose
            running median (``margin_window`` pixels) is below
            ``margin_fraction`` of the order's median flux. This catches exact
            zero padding and noisy unexposed detector alike.
        remove_nans : bool
            Drop NaN pixels first.
        margin_fraction : float
            Flux level, as a fraction of the order's median, below which an
            edge pixel counts as padding.
        margin_window : int
            Width (pixels) of the running median that smooths the noise before
            the threshold is applied.

        Returns
        -------
        Spectrum
            self, with ``wv`` / ``sp`` trimmed.
        '''
        if n_trim is not None:
            n_trim_left, n_trim_right = n_trim, n_trim

        trimmed = [trim_margins(w, s, n_left=n_trim_left, n_right=n_trim_right,
                                clean_margins=clean_margins, remove_nans=remove_nans,
                                margin_fraction=margin_fraction, margin_window=margin_window)
                   for w, s in zip(self.wv, self.sp)]
        wvT = [w for w, _ in trimmed]
        spT = [s for _, s in trimmed]

        self.wv, self.sp = wvT, spT
        return self

    # =============================================================================
    # =============================================================================
    def filter_spectrum(self, continuum_scale_kms, noise_scale_kms, order):
        '''
        Band-pass filter every order with a Butterworth filter, in place.

        Removes the slowly varying instrumental response (variations broader
        than ``continuum_scale_kms``) and the pixel-scale noise (structure
        narrower than ``noise_scale_kms``): velocity scales, so every order,
        template or data, gets the same physical filter whatever its pixels.
        The filter runs forward and backward (zero phase) on the flux minus
        its 80th percentile, and each filtered order is standardized
        (``unicor.dsp.standardize``). The grid must be evenly spaced: the
        Nyquist frequency is taken from the first wavelength step. See
        ``sparta.preprocess.bandpass``.

        Parameters
        ----------
        continuum_scale_kms : float
            Lower band edge, as a velocity period (km/s).
        noise_scale_kms : float
            Upper band edge, as a velocity period (km/s); more than two grid
            steps.
        order : int
            Order of the Butterworth filter.

        Returns
        -------
        Spectrum
            self, with ``sp`` filtered.
        '''
        spF = [bandpass(w, s, continuum_scale_kms, noise_scale_kms, order=order)
               for w, s in zip(self.wv, self.sp)]
        self.sp = spF
        return self

    # =============================================================================
    # =============================================================================
    def apply_cosine_bell(self, alpha=0.1):
        '''
        Multiply every order by a cosine bell (Tukey window), in place.

        The taper brings the order's ends to zero, so it suits a filtered
        (zero-mean) spectrum.

        Parameters
        ----------
        alpha : float
            Shape parameter of the Tukey window: the tapered fraction of the
            order (0 = no taper, 1 = Hann window).

        Returns
        -------
        Spectrum
            self, with ``sp`` tapered.
        '''
        spC = [s * dsp.tukey_window(len(w), alpha) for w, s in zip(self.wv, self.sp)]
        self.sp = spC
        return self

# =============================================================================
# =============================================================================
    def remove_cosmics(self, sigma_upper=3):
        '''
        Replace upward outliers (cosmic-ray hits) in every order, in place.

        An outlier lies more than ``sigma_upper`` x mad_std above the median
        of its order (iterated, as in astropy's sigma_clip). Points that
        deviate below are never touched.

        A clipped pixel is replaced by linear interpolation (in wavelength)
        between its nearest unclipped neighbours, rather than dropped: dropping
        it would break the uniform (linear or log-lambda) grid that
        ``Info['GridType']`` declares and the CCF requires. NaN pixels are left
        as they are and never used as interpolation anchors. The wavelengths
        are unchanged.

        Parameters
        ----------
        sigma_upper : float
            Clipping threshold, in units of mad_std.

        Returns
        -------
        Spectrum
            self, with ``sp`` cleaned.
        '''

        spR = [remove_cosmics(w, s, sigma_upper=sigma_upper)
               for w, s in zip(self.wv, self.sp)]

        self.sp = spR
        return self

# =============================================================================
# =============================================================================
    def barycentric_correction(self):
        '''
        Apply the barycentric correction to every order's wavelengths, in place.

        The wavelengths are multiplied by (1 + bcv/c) -- always the 'cz'
        composition, whatever velocity convention the RVs are later read in
        -- and ``bcv`` is set to 0 km/s, so a second call does nothing.

        Returns
        -------
        Spectrum
            self.
        '''

        # The BERV is composed multiplicatively as lambda * (1 + bcv/c) -- the
        # cz convention -- whatever velocity_convention the RVs are later read
        # in: pipeline BERVs are defined to compose this way (1 + z_obs =
        # (1 + z_star)(1 + bcv/c)), so the factor is fixed by how bcv was
        # computed, not chosen here. Do not route it through the RV convention.
        bcv_kms = u.Quantity(self.bcv, u.km / u.s).to_value(u.km / u.s)
        wvBC = [np.squeeze(dsp.doppler_shift_wave(w, bcv_kms, convention='cz'))
                for w in self.wv]

        # Update the BCV to be zero, so that another activation
        # of the function wont mess up the results.
        self.wv = wvBC
        self.bcv = 0.0 * u.kilometer / u.second

        return self

    # =============================================================================
    # =============================================================================
    def select_orders(self, orders, remove=True):
        '''
        Keep or discard orders, in place.

        The remaining orders keep their original relative order (sorted by
        index), and ``n_ord`` and any recorded ``Info['NativeStep']`` are
        updated to match.

        Parameters
        ----------
        orders : sequence of int
            Order indices, 0 .. n_ord - 1.
        remove : bool
            If True, drop these orders; if False, keep only these.

        Returns
        -------
        Spectrum
            self.

        Raises
        ------
        ValueError
            If an index is outside 0 .. n_ord - 1.
        '''
        if np.max(orders) >= self.n_ord or np.min(orders) < 0:
            raise ValueError(f"order indices must lie in 0..{self.n_ord - 1}, got "
                             f"{np.min(orders)}..{np.max(orders)}")

        if remove:
            x = np.setdiff1d(np.arange(self.n_ord), orders)
        else:
            x = orders

        wv = [self.wv[I] for I in np.sort(x)]
        sp = [self.sp[I] for I in np.sort(x)]
        if 'NativeStep' in self.Info:
            native = self.native_steps()
            self.Info['NativeStep'] = [native[I] for I in np.sort(x)]

        self.wv = wv
        self.sp = sp
        self.n_ord = len(self.wv)

        return self


def _same(a, b):
    '''Whether two Info values are both absent (None) or compare equal.'''
    if a is None or b is None:
        return a is b
    return bool(np.all(a == b))
