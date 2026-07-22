"""Tests for the hardened PHOENIX download path (no network required)."""
import pytest

from sparta.UNICOR.Template import Template


def test_download_disabled_raises_without_network():
    # A template whose files are not cached locally must raise (not silently
    # reach the network) when allow_download=False.
    with pytest.raises(FileNotFoundError):
        Template(temp=9999, log_g=9.99, metal=0, alpha=0,
                 min_val=5000, max_val=5100, allow_download=False)


def test_download_url_is_https_and_correct():
    t = Template.__new__(Template)
    t.server_url = "https://phoenix.astro.physik.uni-goettingen.de/data/"
    p, fname = t.create_PHOENIX_fname(5500, 4.5, 0.0, 0.0)
    url = t.server_url + p.strip("/") + "/" + fname
    assert url.startswith("https://")
    assert "/data/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/" in url
    assert url.endswith("lte05500-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")
