"""docs/api.md is the rendered docstrings of the public surface, kept current."""
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _generator():
    spec = importlib.util.spec_from_file_location("_generate_api", ROOT / "docs" / "_generate_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_api_reference_is_the_rendered_docstrings():
    docs = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    assert docs == _generator().render(), \
        "docs/api.md is stale: regenerate with `python docs/_generate_api.py --write`"


def test_api_reference_covers_the_top_level_surface():
    import sparta
    import todcor
    import unicor

    docs = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    for name in sparta.__all__:
        heading = f"`sparta.{name}`" if name[0].isupper() else f"### sparta.{name}"
        assert heading in docs, name
    for package in (unicor, todcor):
        for name in package.__all__:
            if callable(getattr(package, name)):
                assert f"{name}(" in docs, f"{package.__name__}.{name}"
