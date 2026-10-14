"""The instrument registry: completeness, metadata consistency, generated docs, and the
declared frames of the verified products against the bundled real files."""
import pathlib
import pkgutil
import warnings

import numpy as np
import pytest
from astropy import constants

import sparta.instruments as instruments
from sparta.instruments.base import SPECSYS_VALUES, WAVELENGTH_FRAMES, Product
from unicor.dsp import dv_from_wave

ROOT = pathlib.Path(__file__).resolve().parents[2]
C_KMS = constants.c.to_value("km/s")

# Every verified product must be exercised on real archive files bundled here.
REAL_FILES = {
    "HARPS": sorted((ROOT / "examples" / "data" / "TOI193").glob("*.fits")),
    "NRES": sorted((ROOT / "examples" / "data" / "TOI677").glob("*.fits")),
}
needs_data = pytest.mark.skipif(not all(REAL_FILES.values()), reason="bundled example data not present")


def test_docs_file_is_the_rendered_registry():
    docs = (ROOT / "docs" / "instruments.md").read_text(encoding="utf-8")
    assert docs == instruments.render_markdown(), \
        "docs/instruments.md is stale: regenerate with `python -m sparta.instruments > docs/instruments.md`"


def test_every_public_module_is_registered():
    public = {m.name for m in pkgutil.iter_modules(instruments.__path__)
              if not m.name.startswith("_") and m.name != "base"}
    assert public == {m.__name__.rsplit(".", 1)[1] for m in instruments.MODULES}
    for module in instruments.MODULES:
        assert module.PRODUCTS
        for product in module.PRODUCTS:
            assert instruments.REGISTRY[product.survey] is product


def test_duplicate_survey_keys_raise():
    with pytest.raises(ValueError, match="duplicate survey key 'HARPS'"):
        instruments._build_registry([instruments.harps, instruments.harps])


def test_unknown_survey_lists_known_ones():
    with pytest.raises(ValueError, match=r"unknown survey 'KEK'; known surveys: APOGEE, .*HARPS"):
        instruments.get("KEK")


@pytest.mark.parametrize("product", instruments.REGISTRY.values(), ids=lambda p: p.survey)
def test_metadata_is_consistent(product):
    assert product.wavelength_frame in WAVELENGTH_FRAMES
    assert product.specsys in SPECSYS_VALUES
    assert product.time_system
    instrument, rec = product.instrument, product.recommended
    native = instrument.native_step_kms
    if native is not None and rec.dv_kms is not None:
        assert rec.dv_kms <= native
    if native is not None and instrument.resolving_power is not None:
        # The native sampling must Nyquist-sample a resolution element.
        assert C_KMS / instrument.resolving_power >= 2 * native
    if rec.window_A is not None:
        assert not product.multi_order and rec.window_A[0] < rec.window_A[1]
    if not product.verified:
        assert product.survey not in REAL_FILES


def test_invalid_frames_are_refused_at_definition():
    good = instruments.get("HARPS")
    fields = {f: getattr(good, f) for f in Product.__dataclass_fields__}
    with pytest.raises(ValueError, match="wavelength_frame"):
        Product(**{**fields, "wavelength_frame": "vac"})
    with pytest.raises(ValueError, match="specsys"):
        Product(**{**fields, "specsys": "LSRK"})


def test_every_verified_product_has_real_files():
    assert {p.survey for p in instruments.REGISTRY.values() if p.verified} == set(REAL_FILES)


@pytest.mark.parametrize("product", [p for p in instruments.REGISTRY.values() if not p.verified],
                         ids=lambda p: p.survey)
def test_unverified_loaders_warn(product, tmp_path):
    with pytest.warns(UserWarning, match=f"loader not verified against archive files: {product.survey}"):
        with pytest.raises(FileNotFoundError):
            product.load(tmp_path / "missing.fits", min_wv=5000.0, max_wv=5100.0)


@needs_data
@pytest.mark.parametrize("survey", list(REAL_FILES))
def test_bundled_files_match_declared_frames(survey):
    product = instruments.get(survey)
    window = product.recommended.window_A or (5150.0, 5300.0)
    for path in REAL_FILES[survey]:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)        # verified: no warning
            spec = product.load(path, min_wv=window[0], max_wv=window[1])
        assert (spec.specsys, spec.wavelength_frame, spec.time_system) == \
            (product.specsys, product.wavelength_frame, product.time_system)
        assert spec.survey == survey and spec.path == str(path)
        assert (len(spec.wave) > 1) == product.multi_order


@needs_data
def test_nres_native_step_matches_the_bundled_grid():
    native = instruments.get("NRES").instrument.native_step_kms
    for path in REAL_FILES["NRES"]:
        spec = instruments.get("NRES").load(path, min_wv=5150.0, max_wv=5300.0)
        measured = np.median([dv_from_wave(w) for w in spec.wave])
        assert abs(measured - native) <= 0.1 * native

