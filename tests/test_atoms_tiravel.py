"""TIRAVEL as a composition of the array atom :func:`sparta.tiravel_ops.build_tiravel`.

Three layers, as in ``tests/test_atoms_unicor.py``:

* ``build_tiravel`` itself, on plain per-epoch, per-order arrays: what it
  returns, what it refuses, and that it is the chain of the finer
  ``tiravel_ops`` steps;
* ``TIRAVEL.build_template`` calls it exactly once -- spied through the name
  ``sparta.tiravel`` imports;
* the atom, on the arrays of the byte-identity baseline's spectra
  (``tests/_atoms_baseline.py``, cubic and linear cases), rebuilds every
  attribute of the object exactly (``np.array_equal``, not a tolerance).

Velocities in km/s, wavelengths in Angstrom. Self-contained: no external data.
"""
import dataclasses
import pathlib
import sys

import numpy as np
import pytest

from sparta import tiravel as tiravel_module
from sparta.tiravel import TIRAVEL
from sparta.tiravel_ops import (
    TiravelResult, align_spectra, alignment, build_tiravel, coadd, common_grid,
    correlation_matrix, normalize_rows,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _atoms_baseline import _tiravel_epochs  # noqa: E402

VELOCITIES = np.array([-31.0, -12.5, 4.0, 18.2, 27.7, 40.1])
CASES = {"cubic": {}, "linear_ref2_m5": dict(kind="linear", reference=2, margin=5)}


def _arrays(spectra):
    return [s.wv for s in spectra], [s.sp for s in spectra]


@pytest.fixture(scope="module")
def spectra():
    return _tiravel_epochs(VELOCITIES)


# =============================================================================
# The atom
# =============================================================================
def test_build_tiravel_returns_a_frozen_result(spectra):
    waves, fluxes = _arrays(spectra)
    res = build_tiravel(waves, fluxes, VELOCITIES)
    assert isinstance(res, TiravelResult)
    K = len(VELOCITIES)
    assert res.weights.shape == (K,) and res.correlation.shape == (K, K)
    assert np.isclose(np.linalg.norm(res.weights), 1.0)
    assert len(res.grids) == len(res.template_flux) == len(res.aligned) == 2
    for g, t, a in zip(res.grids, res.template_flux, res.aligned):
        assert t.shape == g.shape and a.shape == (K, g.size)
    assert set(res.diagnostics) == {"lambda_max", "eigenvalues", "explained", "n_eff"}
    assert 0.9 < res.rho <= 1.0
    assert res.velocities.dtype == float and np.array_equal(res.velocities, VELOCITIES)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.rho = 0.0


def test_build_tiravel_is_the_chain_of_the_finer_steps(spectra):
    waves, fluxes = _arrays(spectra)
    res = build_tiravel(waves, fluxes, VELOCITIES, orders=[1, 0], margin=7, kind="linear")
    grids, aligned = [], []
    for I in (1, 0):
        w = [s.wv[I] for s in spectra]
        f = [s.sp[I] for s in spectra]
        grids.append(common_grid(w, VELOCITIES, margin=7))
        aligned.append(align_spectra(w, f, VELOCITIES, grids[-1], kind="linear"))
    normalized = normalize_rows(aligned)
    R = correlation_matrix(normalized)
    stats = alignment(R)
    cut = np.cumsum([0] + [g.size for g in grids])
    for got, want in zip(res.grids, grids):
        assert np.array_equal(got, want)
    for I, (a, b) in enumerate(zip(cut[:-1], cut[1:])):
        assert np.array_equal(res.aligned[I], normalized[:, a:b])
        assert np.array_equal(res.template_flux[I], coadd(normalized[:, a:b], stats["weights"]))
    assert np.array_equal(res.correlation, R) and np.array_equal(res.weights, stats["weights"])
    assert res.rho == stats["rho"]


def test_build_tiravel_leaves_its_inputs_alone(spectra):
    waves, fluxes = _arrays(spectra)
    before = [[f.copy() for f in per_epoch] for per_epoch in fluxes]
    build_tiravel(waves, fluxes, VELOCITIES)
    for per_epoch, saved in zip(fluxes, before):
        for f, s in zip(per_epoch, saved):
            assert np.array_equal(f, s)


@pytest.mark.parametrize("mutate, match", [
    (lambda w, f, v: (w, f, v[:-1]), "got 6 spectra but 5 velocities"),
    (lambda w, f, v: (w[:1], f[:1], v[:1]), "at least 2 epochs"),
    (lambda w, f, v: (w, f[:-1], v), "one flux list per epoch"),
    (lambda w, f, v: (w, f, np.r_[np.nan, v[1:]]), "velocities contain NaN/inf"),
    (lambda w, f, v: ([w[0][:1]] + w[1:], f, v), "same number of orders"),
    (lambda w, f, v: (w, [f[0][:1]] + f[1:], v), "one flux array per order"),
])
def test_build_tiravel_validates_its_inputs(spectra, mutate, match):
    waves, fluxes = _arrays(spectra)
    with pytest.raises(ValueError, match=match):
        build_tiravel(*mutate(waves, fluxes, VELOCITIES))


def test_build_tiravel_names_the_epoch_and_order_of_a_nan_pixel(spectra):
    waves, fluxes = _arrays(spectra)
    fluxes = [list(f) for f in fluxes]
    fluxes[3][1] = fluxes[3][1].copy()
    fluxes[3][1][100] = np.nan
    with pytest.raises(ValueError, match=r"^epoch 3, order 1: flux contains NaN/inf"):
        build_tiravel(waves, fluxes, VELOCITIES)


# =============================================================================
# TIRAVEL.build_template calls the atom once
# =============================================================================
def test_build_template_calls_build_tiravel_once(monkeypatch, spectra):
    calls = []
    real = tiravel_module.build_tiravel

    def wrapper(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(tiravel_module, "build_tiravel", wrapper)
    tir = TIRAVEL()
    tir.default_margin = 4
    tir.build_template(spectra, VELOCITIES, orders=[1], reference=3, kind="linear")
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert kwargs == dict(orders=[1], reference=3, margin=4, kind="linear")
    assert all(w is s.wv for w, s in zip(args[0], spectra))


# =============================================================================
# The atom rebuilds the object exactly
# =============================================================================
@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("keep_aligned", [True, False])
def test_the_atom_rebuilds_tiravel_exactly(spectra, case, keep_aligned):
    trial = VELOCITIES + 0.3                       # the baseline's trial velocities
    kwargs = CASES[case]
    obj = TIRAVEL().build_template(spectra, trial, keep_aligned=keep_aligned, **kwargs)
    res = build_tiravel(*_arrays(spectra), trial, margin=kwargs.get("margin", 10),
                        **{k: v for k, v in kwargs.items() if k != "margin"})

    assert obj.rho == res.rho
    for name in ("weights", "correlation", "velocities"):
        assert np.array_equal(getattr(obj, name), getattr(res, name)), name
    assert obj.diagnostics.keys() == res.diagnostics.keys()
    for key in res.diagnostics:
        assert np.array_equal(obj.diagnostics[key], res.diagnostics[key]), key
    assert obj.n_ord == len(res.grids) == len(obj.template.model.wv)
    for I, grid in enumerate(res.grids):
        assert np.array_equal(obj.grids[I], grid)
        assert np.array_equal(obj.template.model.wv[I], grid)
        assert np.array_equal(obj.template.model.sp[I], res.template_flux[I])
        if keep_aligned:
            assert np.array_equal(obj.aligned[I], res.aligned[I])
    assert obj.aligned is None or keep_aligned
    assert obj.template.model.name == "tiravel"
