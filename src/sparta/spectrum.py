"""Spectrum: one observed (multi-order) spectrum and its preprocessing.

``wv`` and ``sp`` are lists holding one 1-D wavelength and flux array per
order (orders may differ in length), plus metadata: time of observation,
barycentric velocity correction, target name and the grid bookkeeping in
``Info``. The methods resample (linear or log-lambda), trim, filter, apodise,
clip cosmics, apply the barycentric correction and select orders; ``orders`` /
``from_spectrum1d`` convert to and from specutils ``Spectrum1D``.

Conventions: wavelengths in Angstrom, velocities in km/s, times in BJD (days).
Cross-correlation needs a log-lambda (velocity-even) grid, step
``log(1 + dv/c)`` -- see ``interpolate(method='log')``.
"""

import numpy as np
from scipy import signal
from astropy.stats import sigma_clip, mad_std
from astropy import constants as consts, units as u
from specutils import Spectrum as Spec1D, SpectrumList
from specutils.manipulation import SplineInterpolatedResampler
from unicor import dsp


def resample_flux(wave, flux, new_wave, unit=u.Angstrom):
    '''
    Resample one order's flux with the cubic spline of ``Spectrum.interpolate``
    (specutils ``SplineInterpolatedResampler``). Other code that must model
    this resampling exactly (CCF2d's noise model) calls it too.

    :param: wave     - 1-D increasing wavelength array, in ``unit``.
    :param: flux     - 1-D flux array matching ``wave``.
    :param: new_wave - 1-D increasing wavelengths (``unit``) inside the span
                       of ``wave``; outside it the flux is NaN.
    :param: unit     - astropy unit of both wavelength arrays (default Angstrom).
    :return: ndarray, the flux at ``new_wave``.
    '''
    src = Spec1D(flux=np.asarray(flux, float) * u.dimensionless_unscaled,
                 spectral_axis=np.asarray(wave, float) * unit)
    return SplineInterpolatedResampler()(src, np.asarray(new_wave, float) * unit).flux.value


class Spectrum:

    # =============================================================================
    # =============================================================================
    def __init__(self, wv=[], sp=[], wave_unit=1.0*u.Angstrom,
                 bjd=0.0, bcv=0.0, name='Anonymous'):
        '''
        Build a Spectrum from per-order wavelength and flux lists.

        :param: wv - list of 1-D wavelength arrays (Angstrom), one per order.
        :param: sp - list of 1-D flux arrays, matching ``wv`` order by order.
        :param: wave_unit - astropy unit of the wavelengths (default Angstrom).
        :param: bjd - barycentric Julian date of the observation (days).
        :param: bcv - barycentric velocity correction (km/s).
        :param: name - str, the name of the target.

        Stores the mutable per-order lists ``self.wv`` / ``self.sp``, the
        grid/units bookkeeping ``self.Info``, the speed of light ``self.c``
        (km/s), and the order count ``self.n_ord``.
        '''
        assert len(wv) == len(sp), "Dimensions mismatch (wv vs. sp)"
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
        The spectrum as a specutils ``SpectrumList`` -- one ``Spectrum1D`` per
        order, with a real ``SpectralAxis`` (Angstrom) and (normalized,
        dimensionless) flux. Per-order ``meta`` carries this Spectrum's ``Info``
        plus ``bjd``/``bcv``/``name``; a recorded ``NativeStep`` is split into
        the order's own value (km/s), so ``from_spectrum1d`` can restore it.

        This is the interchange representation for the specutils ecosystem. The
        working ``.wv`` / ``.sp`` lists remain the mutable store used by the
        preprocessing methods; ``orders`` is built from them on access.
        '''
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
        Build a ``Spectrum`` from one or more specutils ``Spectrum1D`` objects
        (a single object, a list, or a ``SpectrumList``). Each becomes one
        order. ``bjd``/``bcv``/``name`` are taken from the first object's
        ``meta`` when present, else from the arguments.

        The sampling bookkeeping that ``orders`` writes is restored, so the
        error estimates still count the native pixels: the grid description
        (``GridType``, ``GridDelta``, ``GridUnits``) when every order agrees on
        it (otherwise the grid is taken as 'raw', to be resampled), and the
        per-order ``NativeStep`` (km/s; an order without one counts its own
        grid as native).
        '''
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

        :return: list of floats (km/s), one per order: the recorded
                 ``Info['NativeStep']``, else c * median(d(lambda)/lambda) of
                 each order's current grid.
        :raises ValueError: if ``Info['NativeStep']`` does not hold one finite,
                            positive value per current order (e.g. orders
                            were added or dropped without updating it).
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
                   delta=0.5, cosmic_sigma=3, filter_lowcut=3, filter_highcut=0.15,
                   alpha=0.3):
        '''
        Apply the standard preprocessing chain, in the recommended order:
        ``trim``, a linear ``interpolate``, ``remove_cosmics``,
        ``filter_spectrum`` (first-order Butterworth band-pass) and
        ``apply_cosine_bell``.

        :param: n_trim, clean_margins, remove_nans - see ``trim``.
        :param: delta          - linear resampling step, in units of each
                                 order's median wavelength step (see
                                 ``interpolate``).
        :param: cosmic_sigma   - upward outlier threshold, in mad_std (see
                                 ``remove_cosmics``).
        :param: filter_lowcut, filter_highcut - the band-pass edges (see
                                 ``filter_spectrum``).
        :param: alpha          - Tukey shape parameter (see
                                 ``apply_cosine_bell``).
        :return: self.
        '''

        self.trim(n_trim=n_trim, clean_margins=clean_margins, remove_nans=remove_nans)
        self.interpolate(delta=delta)
        self.remove_cosmics(sigma_upper=cosmic_sigma)
        self.filter_spectrum(lowcut=filter_lowcut, highcut=filter_highcut, order=1)
        self.apply_cosine_bell(alpha=alpha)

        return self

    # =============================================================================
    # =============================================================================
    def interpolate(self, delta=None, method='linear'):
        '''
        This function resamples a list of wavelength and spectra
        so that the wavelength values will be evenly spaced.

        :param: self.wv - list of 1D wavelength arrays.
        :param: self.sp - list of 1D flux arrays, correspond the wv arrays.
        :param: delta   - positive float.
                          If method is 'linear', the step of the evenly
                          spaced array is taken to be delta*(median delta lambda).
                          In this case delta is a unitless parameter.
                          If method is 'log', then the array is resampled
                          to be logarithmically evenly-spaced. The step is
                          taken to be log(1+delta/c), where c is the
                          speed of light in km/sec, and so is delta.
        :param: method  - string. 'linear' to generate an evenly spaced
                          wavelength vector. 'log' to generate a
                          logarithmically evenly-spaced vector.
        :return: self.wv - list of evenly spaced wavelegth arrays.
        :return: self.sp - list of flux values, interpolated at wvI.

        Notes: The input, wv and sp, are data from a single observation.
               Each entry in the list represents a different order.
               The step size of the interpolated wavelength array is set
               to be f times the minimal (non zero, positive) wavelength
               step in the input data.-
        '''
        if method not in ('linear', 'log'):
            raise ValueError(f"method must be 'linear' or 'log', not {method!r}")

        # Initialize data
        wvI, spI, wnt = [], [], self.Info['WavelengthUnits']

        # Remember the sampling that sets the noise scale, so later error
        # estimates can count independent pixels (native_steps).
        previous = self.native_steps()

        # Set the interpolation deltas
        if method == 'linear':
            if delta is None:
                delta = 0.5 * wnt
            elif type(delta) is not u.quantity.Quantity:
                delta = float(delta) * wnt
            deltawv = ((delta/wnt).decompose()).value

        if method == 'log':
            if delta is None:
                delta = 0.1 * u.kilometer / u.second
            elif type(delta) is not u.quantity.Quantity:
                delta = float(delta) * u.kilometer / u.second
            betavel = ((delta/self.c).decompose()).value

        # Resample the flux onto the new grid with specutils (cubic spline).
        wunit = getattr(wnt, 'unit', u.Angstrom)

        # Interpolate per order
        for I, w in enumerate(self.wv):

            # Calculate the median wavelength difference.
            medianDiff = np.median(np.diff(w))

            # If the grid is interpolated in wavelength:
            if method == 'linear':
                dlam = deltawv*medianDiff
                wvI.append(np.arange(min(w) + abs(dlam), max(w) - abs(dlam), abs(dlam)))

            # If the grid is interpolated in velocity
            elif method == 'log':
                dlam = np.log(1+betavel)
                wvI.append(np.exp(
                           np.arange(np.log(min(w)+medianDiff),
                                     np.log(max(w)-medianDiff),
                                     dlam)))

            # Resample the flux onto the new grid. The target grid lies strictly
            # inside the input span (built with a one-median-step margin), so no
            # extrapolation occurs.
            spI.append(resample_flux(w, self.sp[I], wvI[I], unit=wunit))

        # Resampling onto a finer grid adds no independent pixels; resampling
        # onto a coarser one leaves fewer. So the coarsest sampling seen so far
        # is the native one. dv_from_wave takes the median d(lambda)/lambda:
        # a linear grid's velocity step falls from the blue to the red end of
        # the order, and this is its mid-order value.
        self.Info['NativeStep'] = [max(old, dsp.dv_from_wave(new))
                                   for old, new in zip(previous, wvI)]

        # Replace the data with the interpolated wavelengths
        self.wv, self.sp = wvI, spI

        # Document the type of grid that is currently
        # saved with the function.
        if method == 'log':
            self.Info['GridType'], self.Info['GridDelta'], self.Info['GridUnits'] = 'log', delta, 'velocity'
        else:
            self.Info['GridType'], self.Info['GridDelta'], self.Info['GridUnits'] = 'linear', delta, 'wavelength'

        return self

    # =============================================================================
    # =============================================================================
    def trim(self, n_trim=None, n_trim_left=0, n_trim_right=0,
             clean_margins=False, remove_nans=True):
        '''
        In some cases the spectra is zero-padded at the edges, which may affect
        the filtering stage. This routine trims the edges of the data.

        :param: self.wv - list of 1D wavelength arrays.
        :param: self.sp - list of 1D flux arrays, correspond the wv arrays.
        :param: n_trim  - Integer. Number of points to remove from each side
                          of the spectrum; overrides n_trim_left/n_trim_right.
        :param: n_trim_left, n_trim_right - Integers. Points to remove from
                          the blue / red end.
        :param: clean_margins - Boolean. Trim counting from the first and last
                          non-zero flux, i.e. drop the zero padding.
        :param: remove_nans - Boolean. Drop NaN pixels first.
        :return: self, with self.wv / self.sp trimmed.
        '''
        # Initialize data
        wvT, spT = [], []

        if n_trim is not None:
            n_trim_left, n_trim_right = n_trim, n_trim

        # Interpolate per order
        for I, w in enumerate(self.wv):
            s = self.sp[I]

            if remove_nans:
                w = w[~np.isnan(s)]
                s = s[~np.isnan(s)]

            if clean_margins:
                non_zero_loc = np.nonzero(self.sp[I])
                first_non_zero = np.min(non_zero_loc)
                last_non_zero = np.max(non_zero_loc)
            else:
                first_non_zero = 0
                last_non_zero = len(w)

            s = s[(first_non_zero+n_trim_left):(last_non_zero-n_trim_right)]
            w = w[(first_non_zero+n_trim_left):(last_non_zero-n_trim_right)]

            spT.append(s)
            wvT.append(w)

        self.wv, self.sp = wvT, spT
        return self

    # =============================================================================
    # =============================================================================
    def filter_spectrum(self, lowcut, highcut, order):
        '''
        This function applies a Butterworth fitler to the input
        spectrum, removes the low-pass instrumental repsponse and the high-pass
        instrumental noise.

        :param: self.sp - list of 1D flux arrays, correspond to the wv arrays.
        :param: lowcut  - float. Stopband freq for low-pass filter. Given in
                          units of the minimal frequency (max(w)-min(w))**(-1)
        :param: highcut - float. Stopband freq for the high-pass filter. Given
                          in units of the Nyquist frequency.
        :param: order -   integer. The order of the Butterworth filter.

        :return: spF - list of filtered flux values at wv.
        '''
        # Initialize data
        spF = []

        # Filter the spectrum per order
        for I, w in enumerate(self.wv):
            # the bandpass values are normalized to the Nyquist frequency.
            # here we assume that the spectrum in evenly sampled, therefore:
            nyq = 0.5/np.abs(w[1]-w[0])

            # The minimal frequency is set by the full range of the datapoints
            # here we use express the minimal frequency in terms of the Nyquist
            df = (max(w)-min(w))**(-1)/nyq

            # The highpass frequency is already given in terms of the Nyquist
            # the lowpass filter is provided in terms of the minimal frequency.
            # it is therefore expressed by the Nyquist frequency.
            low = lowcut*df

            # Define the Butterworth filter
            b, a = signal.butter(order, [low, highcut], btype='band')

            # Winsorize the data
            s = signal.filtfilt(b, a, self.sp[I]-np.percentile(self.sp[I], 80),
                                padlen=10*max(len(b), len(a)))

            s = (s-np.mean(s))/np.std(s)

            # Filter the data
            spF.append(s)

        self.sp = spF
        return self

    # =============================================================================
    # =============================================================================
    def apply_cosine_bell(self, alpha=0.1):
        '''
        This function applies a cosine bell (Tukey Window) to the spectrum,

        :param:  self.wv - list of 1D wavelength arrays (evenly sampled).
        :param:  self.sp - list of 1D flux arrays, correspond to the wv arrays.
        :param:  alpha -   (optional) float. Shape parameter of the Tukey win.

        :return: spC - list of flux values at wv,
                      after the cosine bell was applied
        '''
        # Initialize data
        spC = []

        # Interpolate per order
        for I, w in enumerate(self.wv):
            window = signal.windows.tukey(len(w), alpha)
            spC.append(
                    np.multiply(self.sp[I], window))

        self.sp = spC
        return self

# =============================================================================
# =============================================================================
    def remove_cosmics(self, sigma_upper=3):
        '''
        This function replaces positive outliers, i.e., points that
        significantly deviate above the measured spectra. Significant deviation
        is defined as points that lie more than sigma_upper x mad_std above the
        median of the order (iterated, as in astropy's sigma_clip). Points that
        deviate below are never touched.

        A clipped pixel is replaced by linear interpolation (in wavelength)
        between its nearest unclipped neighbours, rather than dropped: dropping
        it would break the uniform (linear or log-lambda) grid that
        ``Info['GridType']`` declares and the CCF requires. NaN pixels are left
        as they are and never used as interpolation anchors.

        :param: self.wv - list of 1D wavelength arrays (Angstrom); unchanged.
        :param: self.sp - list of 1D flux arrays, correspond to the wv arrays.
        :param: sigma_upper - float. Number of sigma beyond which points will
                              be replaced. (Optional. default=3).

        :return: self, with self.sp holding the cleaned fluxes.
        '''

        spR = []
        for w, s in zip(self.wv, self.sp):
            w = np.asarray(w, float)
            s = np.array(s, float)
            clipped = np.ma.getmaskarray(sigma_clip(s, sigma_lower=np.inf,
                                                    sigma_upper=sigma_upper,
                                                    masked=True, cenfunc='median',
                                                    stdfunc=mad_std))
            finite = np.isfinite(s)
            cosmic = clipped & finite
            anchors = ~clipped & finite
            s[cosmic] = np.interp(w[cosmic], w[anchors], s[anchors])
            spR.append(s)

        self.sp = spR
        return self

# =============================================================================
# =============================================================================
    def barycentric_correction(self):
        '''
        Apply a barycentric correction to the wavelength vectors
        of each order. Once it is done, the bcv value in the structure
        is set to be 0.0 km/s.
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
        This routine enables the user to keep or discard orders from the observed
        data. Orders are numbered according to their index (0-> #ord-1),

        :param orders: Indices of the required orders
        :param remove: Boolean. If True, these orders will be removed.
                       otherwise, these will be the orders that will
                       be kept.
        :return: Same structure, with the orders arranged.
        '''
        assert (np.max(orders) < self.n_ord), "Invalid order numbers"
        assert (np.min(orders) >= 0), "Invalid order numbers"

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
    '''True if two Info values are both absent or compare equal.'''
    if a is None or b is None:
        return a is b
    return bool(np.all(a == b))
