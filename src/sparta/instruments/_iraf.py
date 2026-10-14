"""IRAF multispec world-coordinate evaluation (one echelle aperture per image line)."""
import re

import numpy as np
from numpy.polynomial import chebyshev, legendre


def _iraf_multispec_wavelengths(header):
    '''
    Evaluate an IRAF "multispec" world coordinate system, one aperture per
    image line.

    Implements sec. 5 of the IRAF spectral WCS description
    (noao/onedspec/doc/specwcs.hlp, https://github.com/iraf-community/iraf):
    ``specN = ap beam dtype w1 dw nw z aplow aphigh [functions]`` with
    linear (dtype 0), log-linear (1) and non-linear (2) dispersion; the
    non-linear form is ``w = sum_i wt_i * (w0_i + W_i(p)) / (1 + z)`` with
    Chebyshev (ftype 1) or Legendre (ftype 2) ``W_i``. specutils' 'iraf'
    reader ignores the weight and zero-point terms and sums of functions,
    which IRAF writes after shift-corrected dispersion solutions, and also
    the LTV/LTM logical-to-physical mapping of image sections.

    The Doppler factor divides the *wavelength* in every case, including the
    log-linear one: ``w = 10 ** (w1 + dw (p - 1)) / (1 + z)``. That is what
    IRAF's multispec driver computes (sys/mwcs/wfmspec.x, wf_msp_fwd) and what
    specwcs.hlp states in prose; its displayed log-linear equation, which puts
    ``(1 + z)`` inside the exponent, disagrees with both.

    Parameters
    ----------
    header : astropy.io.fits.Header
        Header with WAT2_nnn cards and NAXIS1.

    Returns
    -------
    dict
        Physical image line N (1-based) -> float64 wavelengths of the image
        columns whose physical pixel lies within the ``nw`` valid ones (a
        leading run of columns), in the units of WAT1 (Angstrom for TRES).

    Raises
    ------
    ValueError
        If the header carries no multispec WAT2 attributes or they are malformed.
    NotImplementedError
        For spline / sampled dispersion functions (ftype 3-6), an
        uncalibrated aperture (dtype -1), or an image section whose valid
        pixels do not start at its first column.
    '''
    keys = sorted(k for k in header if re.fullmatch(r'WAT2_\d{3}', k))
    # IRAF splits the attribute string into 68-character card values; astropy
    # strips trailing blanks, which may fall between two fields.
    wat = ''.join(f"{header[k]:68s}" for k in keys)
    specs = re.findall(r'spec(\d+)\s*=\s*"([^"]*)"', wat)
    if not specs:
        raise ValueError("no IRAF multispec 'specN' attributes in the WAT2 header cards")
    # Logical -> physical pixel (specwcs.hlp: p = (l - LTV1) / LTM1_1).
    pix = (np.arange(1, header['NAXIS1'] + 1) - header.get('LTV1', 0.0)) / header.get('LTM1_1', 1.0)

    waves = {}
    for line, attribute in specs:
        f = [float(x) for x in attribute.split()]
        if len(f) < 9:
            raise ValueError(f"malformed multispec attribute spec{line}: {attribute!r}")
        dtype, w1, dw, nw, z = int(f[2]), f[3], f[4], int(f[5]), f[6]
        if dtype == 0:
            wave = (w1 + dw * (pix - 1)) / (1 + z)
        elif dtype == 1:
            wave = 10 ** (w1 + dw * (pix - 1)) / (1 + z)
        elif dtype == 2:
            wave = _multispec_nonlinear(f[9:], pix, line) / (1 + z)
        else:
            raise NotImplementedError(f"spec{line} has dispersion type {dtype}: "
                                      "the aperture is not wavelength calibrated")
        # nw counts physical pixels; in an image section (LTV1 != 0) the
        # valid columns are those that map onto physical pixels 1..nw.
        valid = (pix > 0.5) & (pix < nw + 0.5)
        n_valid = int(np.count_nonzero(valid))
        if not valid[:n_valid].all():
            raise NotImplementedError(f"spec{line}: the valid physical pixels 1..{nw} "
                                      "do not start at the first image column")
        waves[int(line)] = wave[:n_valid]
    return waves


def _multispec_nonlinear(fields, pix, line):
    '''
    Sum of weighted IRAF dispersion functions, ``sum wt_i * (w0_i + W_i(p))``.

    Parameters
    ----------
    fields : list of float
        The function fields of a multispec attribute (after the first nine).
    pix : ndarray
        Physical pixel coordinates (1-based).
    line : str
        Aperture line number, for error messages.

    Returns
    -------
    ndarray
        Wavelength before the Doppler factor, per pixel.
    '''
    wave = np.zeros_like(pix)
    i = 0
    while i < len(fields):
        weight, zero_point, ftype = fields[i], fields[i + 1], int(fields[i + 2])
        if ftype not in (1, 2):
            raise NotImplementedError(f"spec{line}: IRAF dispersion function type {ftype} "
                                      "(spline / pixel array) is not supported")
        order, pmin, pmax = int(fields[i + 3]), fields[i + 4], fields[i + 5]
        coeffs = fields[i + 6:i + 6 + order]
        if len(coeffs) != order:
            raise ValueError(f"spec{line}: expected {order} coefficients, found {len(coeffs)}")
        x = (pix - (pmax + pmin) / 2) / ((pmax - pmin) / 2)
        series = chebyshev.chebval(x, coeffs) if ftype == 1 else legendre.legval(x, coeffs)
        wave += weight * (zero_point + series)
        i += 6 + order
    if i == 0:
        raise ValueError(f"spec{line} is non-linear (dtype 2) but lists no dispersion function")
    return wave
