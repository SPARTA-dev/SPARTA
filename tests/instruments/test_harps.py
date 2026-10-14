"""HARPS-CERES loader on a synthetic CERES cube (the ESO SDP product is pinned on the
bundled TOI193 files in test_readspec.py and test_golden_instruments.py)."""
from fits_builders import check_ceres


def test_harps_ceres_picks_order_covering_window(tmp_path):
    check_ceres("HARPS-CERES", "vacuum", tmp_path)
