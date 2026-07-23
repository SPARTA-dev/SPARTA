"""TIRAVEL: template-independent template construction.

The third correlation engine, alongside UNICOR (`CCF1d`, one template) and
TODCOR (`CCF2d`, two templates). Those two *require* a template; TIRAVEL
*builds* one out of the observations themselves -- Zucker & Mazeh (2006,
MNRAS 371, 1513).

Given K observed spectra and a set of trial radial velocities, every epoch is
shifted to its rest frame and resampled onto one common grid. The K x K
correlation matrix of the aligned spectra has a largest eigenvalue
``lambda_max``, and

    rho = (lambda_max - 1) / (K - 1)

measures how well a *single* principal component describes the whole set --
i.e. how well the spectra are aligned. Its eigenvector gives the optimal
weights for co-adding the epochs into the common template.

Scope
-----
This is the **fixed-velocity** half of the method: the velocities are supplied
and ``rho`` is evaluated once, yielding a template that UNICOR or TODCOR can
then measure against. The full method maximises ``rho`` over the K velocities;
``rho`` is exposed as the objective function for that step.

Note that TIRAVEL velocities are **relative** -- only their differences enter
the correlation matrix, so the template's rest frame inherits whatever zero
point the supplied velocities carry.

Layout
------
``TIRAVEL``      - the SPARTA-object class (Spectrum list + velocities -> Template).
``tiravel_ops``  - the measure: alignment, correlation matrix, weights, co-addition.
"""

from .TIRAVEL import TIRAVEL
from .tiravel_ops import (
    align_spectra, alignment, coadd, common_grid, correlation_matrix,
    normalize_rows, rest_frame_wavelengths,
)

__all__ = [
    'TIRAVEL',
    'rest_frame_wavelengths', 'common_grid', 'align_spectra',
    'normalize_rows', 'correlation_matrix', 'alignment', 'coadd',
]
