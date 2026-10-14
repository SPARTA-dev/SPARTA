"""ESO Phase-3 Science Data Product (SDP) helpers: reference frame, air/vacuum, BCV."""


def _eso_sdp_frames(hdul):
    '''
    Reference frame and air/vacuum flag of an ESO Phase-3 SDP 1-D spectrum.

    ESO Science Data Products Standard (GEN-SPE-ESO-33000-5335, issue 8,
    https://www.eso.org/sci/observing/phase3/p3sdpstd.pdf): SPECSYS is
    mandatory for spectra (sec. 5.6), and the UCD of the WAVE column is
    'em.wl;obs.atmos' for air and 'em.wl' for vacuum wavelengths.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        Open SDP file (primary header + SPECTRUM binary table in HDU 1).

    Returns
    -------
    (specsys, wavelength_frame) : (str, str or None)
        ``wavelength_frame`` is 'air', 'vacuum', or None if TUCD is absent.

    Raises
    ------
    ValueError
        If SPECSYS is missing.
    '''
    specsys = hdul[0].header.get('SPECSYS', hdul[1].header.get('SPECSYS'))
    if specsys is None:
        raise ValueError("ESO SDP spectrum has no SPECSYS keyword; cannot tell whether "
                         "the wavelengths are barycentric or topocentric")
    names = [name.upper() for name in hdul[1].columns.names]
    tucd = hdul[1].header.get(f"TUCD{names.index('WAVE') + 1}", '').strip()
    frame = {'em.wl;obs.atmos': 'air', 'em.wl': 'vacuum'}.get(tucd)
    return specsys.strip(), frame


def _bcv_for_specsys(header, specsys, bcv_keyword):
    '''
    Barycentric velocity correction (km/s) still to be applied, given SPECSYS.

    Parameters
    ----------
    header : astropy.io.fits.Header
        Header holding ``bcv_keyword``.
    specsys : str
        'BARYCENT' (already corrected -> 0) or 'TOPOCENT' (read the keyword).
    bcv_keyword : str
        Header keyword with the barycentric correction in km/s.

    Returns
    -------
    float
        Velocity correction in km/s.

    Raises
    ------
    ValueError
        For any other SPECSYS.
    '''
    if specsys == 'BARYCENT':
        return 0.0
    if specsys == 'TOPOCENT':
        return float(header[bcv_keyword])
    raise ValueError(f"unsupported SPECSYS {specsys!r}: expected 'BARYCENT' or 'TOPOCENT'")
