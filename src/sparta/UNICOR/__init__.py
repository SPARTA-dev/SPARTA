"""UNICOR: SPARTA-object layer for one-dimensional cross-correlation RVs.

``Spectrum`` and ``Template`` hold (multi-order) spectra; ``CCF1d`` correlates
one against the other. The array-level engine -- CCF, order combination, RV,
uncertainty, bisector -- is the standalone ``unicor`` package.
"""

from .Spectrum import Spectrum
from .Template import Template
from .CCF1d import CCF1d
