"""Tests for TIRAVEL (:mod:`sparta.tiravel`, :mod:`sparta.tiravel_ops`).

Two layers:

* the pure measure (``tiravel_ops``) -- de-shifting, the common grid, the
  correlation matrix, and the alignment/weights;
* the SPARTA-object class ``TIRAVEL``, driven with ``Spectrum`` objects.

The scientifically important ones are ``test_rho_peaks_at_the_true_velocity``
(rho is a usable objective for the optimizer that comes next) and
``test_rho_is_invariant_to_a_global_velocity_offset`` (the zero-point
degeneracy is real and must be documented, not fixed).

Self-contained: no external data.
"""

import numpy as np
import pytest

import unicor as ops
from sparta.tiravel import TIRAVEL
from sparta.tiravel_ops import (
    align_spectra, alignment, coadd, common_grid, correlation_matrix, normalize_rows,
    rest_frame_wavelengths,
)
from sparta.spectrum import Spectrum

C = ops.C_KMS


def _template(wave, lo=4400.0, hi=4600.0, n_lines=60, fwhm_kms=12.0, seed=1):
    """A realistic forest of Gaussian absorption lines on a unit continuum.

    The line list is fixed by ``seed`` so a template built over the same range
    is reproducible across calls (the epochs must share one rest template).
    """
    rng = np.random.default_rng(seed)
    centres = rng.uniform(lo + 5.0, hi - 5.0, n_lines)
    depths = rng.uniform(0.1, 0.6, n_lines)
    flux = np.ones_like(wave)
    for c, d in zip(centres, depths):
        sigma = (fwhm_kms / 2.3548) * c / C
        flux *= 1.0 - d * np.exp(-0.5 * ((wave - c) / sigma) ** 2)
    return flux


def _epochs(velocities, noise=0.0, seed=7, lo=4400.0, hi=4600.0, dv=1.0):
    """Noisy, Doppler-shifted observations of one common template."""
    rng = np.random.default_rng(seed)
    wave_t = np.arange(lo - 20.0, hi + 20.0, 0.02)
    rest = _template(wave_t, lo=lo, hi=hi)

    grid, _ = ops.log_lambda_grid(lo, hi, dv)
    waves, fluxes = [], []
    for v in velocities:
        # Observed: the rest template redshifted by v.
        flux = ops.resample(wave_t * (1.0 + v / C), rest, grid, fill=1.0)
        if noise:
            flux = flux + rng.normal(0.0, noise, size=flux.shape)
        waves.append(grid.copy())
        fluxes.append(flux)
    return waves, fluxes


def _rho_at(velocities, trial, noise=0.0, seed=7):
    """rho of the set built at `velocities`, evaluated at the shifts `trial`."""
    waves, fluxes = _epochs(velocities, noise=noise, seed=seed)
    grid = common_grid(waves, trial, margin=5)
    aligned = align_spectra(waves, fluxes, trial, grid)
    return alignment(correlation_matrix(normalize_rows([aligned])))['rho']


# =============================================================================
# The measure
# =============================================================================
def test_rest_frame_is_the_exact_inverse_of_a_redshift():
    wave = np.linspace(4000.0, 5000.0, 11)
    v = 137.0
    assert np.allclose(rest_frame_wavelengths(wave * (1.0 + v / C), v), wave)


def test_common_grid_is_inside_every_deshifted_epoch():
    velocities = [-80.0, 0.0, 120.0]
    waves, _ = _epochs(velocities)
    grid = common_grid(waves, velocities, margin=3)
    for w, v in zip(waves, velocities):
        rest = rest_frame_wavelengths(w, v)
        assert grid.min() >= rest.min() and grid.max() <= rest.max()


def test_common_grid_raises_when_epochs_do_not_overlap():
    waves, _ = _epochs([0.0, 0.0])
    # A velocity difference (~10% of c) that shifts one epoch clear of the
    # other's 4400-4600 range, so the de-shifted grids share nothing.
    with pytest.raises(ValueError, match='do not overlap'):
        common_grid(waves, [0.0, 30000.0])


def test_correlation_matrix_is_a_correlation_matrix():
    waves, fluxes = _epochs([-40.0, 0.0, 25.0, 60.0], noise=0.01)
    velocities = [-40.0, 0.0, 25.0, 60.0]
    grid = common_grid(waves, velocities, margin=5)
    R = correlation_matrix(normalize_rows([align_spectra(waves, fluxes,
                                                         velocities, grid)]))
    assert np.allclose(np.diag(R), 1.0)
    assert np.allclose(R, R.T)
    assert R.min() >= -1.0 and R.max() <= 1.0


def test_correlation_matrix_matches_explicit_pearson():
    waves, fluxes = _epochs([-30.0, 0.0, 45.0], noise=0.02)
    velocities = [-30.0, 0.0, 45.0]
    grid = common_grid(waves, velocities, margin=5)
    M = normalize_rows([align_spectra(waves, fluxes, velocities, grid)])
    R = correlation_matrix(M)
    expected = np.corrcoef(M)
    assert np.allclose(R, expected, atol=1e-10)


def test_rho_is_one_for_identical_aligned_spectra():
    velocities = [-50.0, 0.0, 33.0, 90.0]
    assert _rho_at(velocities, velocities) == pytest.approx(1.0, abs=1e-6)


def test_rho_degrades_with_noise():
    velocities = [-50.0, 0.0, 33.0, 90.0]
    clean = _rho_at(velocities, velocities)
    noisy = _rho_at(velocities, velocities, noise=0.05)
    assert 0.0 < noisy < clean


def test_rho_peaks_at_the_true_velocity():
    # The property the future optimizer will rely on: rho, scanned over one
    # epoch's velocity with the others held at truth, maximizes at the truth.
    truth = [-50.0, 0.0, 33.0, 90.0]
    offsets = np.arange(-20.0, 20.1, 2.5)
    scan = [_rho_at(truth, [truth[0] + d] + truth[1:], noise=0.02)
            for d in offsets]
    assert offsets[int(np.argmax(scan))] == pytest.approx(0.0, abs=2.5)
    # ...and the peak is a real peak, not a plateau.
    assert max(scan) > scan[0] + 0.01 and max(scan) > scan[-1] + 0.01


def test_rho_is_invariant_to_a_global_velocity_offset():
    # Only velocity *differences* enter R, so TIRAVEL velocities are relative.
    truth = [-50.0, 0.0, 33.0, 90.0]
    base = _rho_at(truth, truth, noise=0.02)
    shifted = _rho_at(truth, [v + 17.0 for v in truth], noise=0.02)
    assert shifted == pytest.approx(base, abs=1e-3)


def test_alignment_picks_the_principal_eigenpair():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((6, 300))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    R = X @ X.T
    np.fill_diagonal(R, 1.0)

    out = alignment(R)
    w, v = np.linalg.eigh(R)
    assert out['lambda_max'] == pytest.approx(w[-1])
    # The weights are the eigenvector of lambda_max, up to the fixed sign.
    assert abs(abs(np.dot(out['weights'], v[:, -1])) - 1.0) < 1e-10
    assert np.sum(out['weights']) >= 0                      # sign convention
    assert out['weights'] @ out['weights'] == pytest.approx(1.0)
    assert out['eigenvalues'].sum() == pytest.approx(len(R))   # trace(R) = K
    assert np.all(np.diff(out['eigenvalues']) <= 1e-12)        # descending
    assert 1.0 <= out['n_eff'] <= len(R)


def test_alignment_rejects_a_single_epoch():
    with pytest.raises(ValueError, match='at least 2 epochs'):
        alignment(np.ones((1, 1)))


def test_weights_favour_the_higher_snr_epochs():
    # Four epochs, two of them much noisier: the clean pair must carry the
    # larger co-addition weight.
    velocities = [0.0, 0.0, 0.0, 0.0]
    waves, fluxes = _epochs(velocities, noise=0.005, seed=11)
    rng = np.random.default_rng(5)
    for i in (2, 3):
        fluxes[i] = fluxes[i] + rng.normal(0.0, 0.08, size=fluxes[i].shape)

    grid = common_grid(waves, velocities, margin=5)
    M = normalize_rows([align_spectra(waves, fluxes, velocities, grid)])
    w = alignment(correlation_matrix(M))['weights']
    assert min(w[0], w[1]) > max(w[2], w[3])


def test_coadd_beats_every_single_epoch():
    # The whole point: the weighted co-addition is a better estimate of the
    # unknown template than any one noisy epoch.
    velocities = [-60.0, -20.0, 15.0, 55.0, 95.0]
    waves, fluxes = _epochs(velocities, noise=0.06, seed=17)
    grid = common_grid(waves, velocities, margin=5)
    M = normalize_rows([align_spectra(waves, fluxes, velocities, grid)])
    template = coadd(M, alignment(correlation_matrix(M))['weights'])

    truth = ops.normalize(_template(grid))
    got = float(np.dot(ops.normalize(template), truth))
    singles = [abs(float(np.dot(ops.normalize(row), truth))) for row in M]
    assert abs(got) > max(singles)


# =============================================================================
# The SPARTA-object class
# =============================================================================
def _spectra(velocities, noise=0.02, n_ord=1, seed=7):
    ranges = [(4400.0, 4600.0), (4600.0, 4800.0)][:n_ord]
    per_order = [_epochs(velocities, noise=noise, seed=seed + k, lo=lo, hi=hi)
                 for k, (lo, hi) in enumerate(ranges)]
    return [Spectrum(wv=[per_order[k][0][i] for k in range(n_ord)],
                     sp=[per_order[k][1][i] for k in range(n_ord)],
                     bjd=float(i), name='TIRAVEL-sim')
            for i in range(len(velocities))]


def test_build_template_recovers_the_common_template():
    velocities = [-60.0, -20.0, 15.0, 55.0, 95.0]
    tir = TIRAVEL().BuildTemplate(_spectra(velocities), velocities)

    assert tir.rho > 0.9
    assert tir.n_ord == 1
    assert tir.correlation.shape == (5, 5)
    assert tir.weights.shape == (5,)

    grid = tir.template.model.wv[0]
    truth = ops.normalize(_template(np.asarray(grid, float)))
    got = ops.normalize(np.asarray(tir.template.model.sp[0], float))
    assert abs(float(np.dot(got, truth))) > 0.98


def test_build_template_is_multi_order():
    velocities = [-40.0, 0.0, 30.0, 70.0]
    tir = TIRAVEL().BuildTemplate(_spectra(velocities, n_ord=2), velocities)

    assert tir.n_ord == 2
    assert len(tir.template.model.wv) == 2
    # Both orders share one weight vector -- they are views of one alignment.
    assert tir.correlation.shape == (4, 4)
    for I in range(2):
        assert len(tir.template.model.sp[I]) == len(tir.template.model.wv[I])


def test_build_template_diagnostics_are_consistent():
    velocities = [-40.0, 0.0, 30.0, 70.0]
    tir = TIRAVEL().BuildTemplate(_spectra(velocities), velocities)
    d = tir.diagnostics

    K = len(velocities)
    assert d['lambda_max'] == pytest.approx(tir.rho * (K - 1) + 1.0)
    assert d['explained'] == pytest.approx(d['lambda_max'] / K)
    assert 1.0 <= d['n_eff'] <= K


def test_build_template_validates_its_inputs():
    velocities = [-40.0, 0.0, 30.0]
    specs = _spectra(velocities)

    with pytest.raises(ValueError, match='3 spectra but 2 velocities'):
        TIRAVEL().BuildTemplate(specs, velocities[:2])
    with pytest.raises(ValueError, match='at least 2 epochs'):
        TIRAVEL().BuildTemplate(specs[:1], velocities[:1])
    with pytest.raises(ValueError, match='NaN/inf'):
        TIRAVEL().BuildTemplate(specs, [np.nan] + velocities[1:])


def test_build_template_zero_point_shifts_the_template():
    # Adding a constant to every velocity leaves rho and the weights alone and
    # moves the template's rest frame by that amount -- the documented
    # degeneracy. The template built at v+offset, cross-correlated against the
    # one built at v, must peak at exactly -offset.
    from sparta import CCF1d
    from sparta.template import Template

    velocities = [-60.0, -20.0, 15.0, 55.0]
    specs = _spectra(velocities, noise=0.01)
    base = TIRAVEL().BuildTemplate(specs, velocities)
    offset = 20.0
    moved = TIRAVEL().BuildTemplate(specs, [v + offset for v in velocities])

    assert moved.rho == pytest.approx(base.rho, abs=1e-6)
    assert np.allclose(np.abs(moved.weights), np.abs(base.weights), atol=1e-6)

    # A CCF of the two templates recovers a shift of exactly `offset` between
    # their rest frames (the sign follows the CCF convention).
    base_spec = Spectrum(wv=[np.asarray(base.template.model.wv[0], float)],
                         sp=[np.asarray(base.template.model.sp[0], float)])
    ccf = CCF1d().CrossCorrelateSpec(
        spec_in=base_spec, template_in=moved.template, dv=0.5, VelBound=60)
    rv, _ = ccf.subpixel_CCF(ccf.Corr['vel'], ccf.Corr['corr'][0])
    assert abs(rv) == pytest.approx(offset, abs=2.0)


def test_observations_driver():
    from sparta.timeseries import TimeSeries
    from sparta.observations import Observations

    velocities = [-60.0, -20.0, 15.0, 55.0]
    specs = _spectra(velocities)
    obs = Observations(time_series=TimeSeries(size=len(specs),
                                              times=list(range(len(specs))),
                                              vals=specs))
    obs.build_tiravel_template(velocities)

    assert obs.tiravel.rho > 0.9
    assert obs.tiravel_template is obs.tiravel.template
    assert len(obs.tiravel_template.model.wv[0]) > 100


def test_build_template_refuses_a_nan_pixel():
    # With linear resampling one NaN used to zero that epoch's weight silently.
    velocities = [-40.0, 0.0, 30.0]
    specs = _spectra(velocities)
    specs[1].sp[0][50] = np.nan
    with pytest.raises(ValueError, match="epoch 1, order 0"):
        TIRAVEL().BuildTemplate(specs, velocities, kind='linear')


def test_common_grid_refuses_a_negative_margin():
    waves, _ = _epochs([0.0, 10.0])
    with pytest.raises(ValueError, match="margin"):
        common_grid(waves, [0.0, 10.0], margin=-5)
