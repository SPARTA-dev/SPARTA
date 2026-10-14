"""TODCOR: two-dimensional correlation for double-lined (SB2) spectra.

The SPARTA-object layer over the standalone ``todcor`` package: ``CCF2d``
takes a ``Spectrum`` and two ``Template`` objects, treats each spectral order as
one TODCOR window, pools them, and reports the velocities, amplitudes and CRLB
uncertainties. The two-component counterpart of UNICOR's ``CCF1d``.

The method itself -- sufficient statistics, profiled amplitudes, CRLB, the
forward model -- and its grid/FFT/broadening primitives are array-level and
live in the ``todcor`` and ``unicor`` packages (``import todcor``).
"""

from .CCF2d import CCF2d

__all__ = ['CCF2d']
