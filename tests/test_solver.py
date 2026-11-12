"""Tests for PlateSolver that do not require astrometry.net index files.

Full integration tests (actually solving against index files) are skipped
unless WILDS_ASTROMETRY_CACHE is set to a directory containing the required
Series 5200 index files.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from wilds.solving.solver import PlateSolver, ScaleHint, SolveResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NEEDS_INDEX = pytest.mark.skipif(
    not os.environ.get("WILDS_ASTROMETRY_CACHE"),
    reason="WILDS_ASTROMETRY_CACHE not set — skipping live solver tests",
)


def _mock_solver_context(has_match: bool = True):
    """Return a context-manager mock that simulates astrometry.Solver."""
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [512, 512]
    wcs.wcs.cdelt = [-0.0003, 0.0003]  # ~1 arcsec/px
    wcs.wcs.crval = [83.8, -5.4]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    match = MagicMock()
    match.astropy_wcs.return_value = wcs
    match.logodds = 122.0  # → "high" confidence

    solution = MagicMock()
    solution.has_match.return_value = has_match
    solution.best_match.return_value = match

    inner = MagicMock()
    inner.__enter__ = MagicMock(return_value=inner)
    inner.__exit__ = MagicMock(return_value=False)
    inner.solve.return_value = solution

    return inner


# ---------------------------------------------------------------------------
# Construction / context-manager protocol
# ---------------------------------------------------------------------------


def test_solve_raises_outside_context():
    with patch("astrometry.series_5200") as mock_series:
        mock_series.index_files.return_value = []
        solver = PlateSolver(cache_dir="/tmp", scales={0})

    stars = np.array([[10.0, 20.0], [30.0, 40.0]])
    with pytest.raises(RuntimeError, match="context manager"):
        solver.solve(stars, image_width=256, image_height=256)


def test_enter_exit_cleans_up():
    with patch("astrometry.series_5200") as mock_series, patch(
        "astrometry.Solver", return_value=_mock_solver_context()
    ):
        mock_series.index_files.return_value = []
        solver = PlateSolver(cache_dir="/tmp", scales={0})
        with solver:
            assert solver._solver is not None
        assert solver._solver is None


# ---------------------------------------------------------------------------
# Stars array normalisation
# ---------------------------------------------------------------------------


def _run_solve_with_stars(stars: np.ndarray) -> SolveResult | None:
    mock_inner = _mock_solver_context(has_match=True)
    with patch("astrometry.series_5200") as mock_series, patch(
        "astrometry.Solver", return_value=mock_inner
    ):
        mock_series.index_files.return_value = []
        with PlateSolver(cache_dir="/tmp", scales={0}) as solver:
            return solver.solve(stars, image_width=1024, image_height=1024)


def test_accepts_plain_xy_array():
    stars = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    result = _run_solve_with_stars(stars)
    assert result is not None


def test_accepts_structured_array_from_sep():
    dtype = np.dtype([("x", float), ("y", float), ("flux", float)])
    stars = np.array([(10.0, 20.0, 500.0), (30.0, 40.0, 300.0)], dtype=dtype)
    result = _run_solve_with_stars(stars)
    assert result is not None


# ---------------------------------------------------------------------------
# Return values
# ---------------------------------------------------------------------------


def test_returns_none_on_no_match():
    mock_inner = _mock_solver_context(has_match=False)
    with patch("astrometry.series_5200") as mock_series, patch(
        "astrometry.Solver", return_value=mock_inner
    ):
        mock_series.index_files.return_value = []
        stars = np.array([[10.0, 20.0]])
        with PlateSolver(cache_dir="/tmp", scales={0}) as solver:
            result = solver.solve(stars, image_width=256, image_height=256)
    assert result is None


def test_result_has_wcs_and_pixel_scale():
    stars = np.array([[10.0, 20.0], [30.0, 40.0]])
    result = _run_solve_with_stars(stars)
    assert result is not None
    assert result.wcs is not None
    assert result.pixel_scale_arcsec > 0


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------


async def test_solve_async_returns_result():
    mock_inner = _mock_solver_context(has_match=True)
    with patch("astrometry.series_5200") as mock_series, patch(
        "astrometry.Solver", return_value=mock_inner
    ):
        mock_series.index_files.return_value = []
        stars = np.array([[10.0, 20.0], [30.0, 40.0]])
        with PlateSolver(cache_dir="/tmp", scales={0}) as solver:
            result = await solver.solve_async(stars, image_width=256, image_height=256)
    assert result is not None


# ---------------------------------------------------------------------------
# Live integration test (requires index files)
# ---------------------------------------------------------------------------


@_NEEDS_INDEX
def test_live_solve():
    """Solve a synthetic field — requires index files in WILDS_ASTROMETRY_CACHE."""
    cache = os.environ["WILDS_ASTROMETRY_CACHE"]
    from tests.conftest import make_synthetic_image
    from wilds.solving.extractor import extract_sources

    image, _ = make_synthetic_image(shape=(512, 512), n_stars=30, star_flux=5000.0)
    sources = extract_sources(image)

    with PlateSolver(cache_dir=cache, scales={0, 1, 2, 3}) as solver:
        result = solver.solve(
            sources,
            image_width=512,
            image_height=512,
            scale_hint=ScaleHint(0.5, 2.0),
        )
    # Synthetic image has no real sky coords so no match expected, but the
    # call must complete without error.
    assert result is None or isinstance(result, SolveResult)


# ---------------------------------------------------------------------------
# Orientation: which way is north in the image, and is it mirrored
# ---------------------------------------------------------------------------


def _wcs(angle_ccw: float, parity: int = 1, scale_arcsec: float = 0.3):
    """A WCS for an image (x right, y down) in which north is ``angle_ccw`` degrees counter-
    clockwise from the top, mirrored when ``parity`` is -1."""
    import math

    from astropy.wcs import WCS

    s = scale_arcsec / 3600.0
    a = math.radians(angle_ccw)
    n = (-math.sin(a), -math.cos(a))  # north's direction in pixels (x right, y down)
    e = (parity * -math.cos(a), parity * math.sin(a))  # east: 90 degrees counter-clockwise
    # pixel offset = M (east, north) arcsec;  world offset = M^-1 pixel offset
    m = np.array([[e[0], n[0]], [e[1], n[1]]]) / s
    inv = np.linalg.inv(m)
    w = WCS(naxis=2)
    w.wcs.crpix = [500.0, 400.0]
    w.wcs.crval = [250.4, 36.4]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cd = inv  # degrees per pixel: rows are (RA, Dec)
    return w


@pytest.mark.parametrize("angle", [0.0, 31.0, -62.0, 90.0, 135.0, -170.0])
@pytest.mark.parametrize("parity", [1, -1])
def test_wcs_orientation_gives_north_angle_and_parity(angle, parity):
    from wilds.solving import wcs_orientation

    got_angle, got_parity = wcs_orientation(_wcs(angle, parity), 500.0, 400.0)
    assert got_parity == parity
    assert (got_angle - angle + 180.0) % 360.0 - 180.0 == pytest.approx(0.0, abs=0.01)


def test_the_solve_result_carries_the_orientation():
    from wilds.solving.solver import wcs_orientation

    mock_inner = _mock_solver_context(has_match=True)
    solution_wcs = _wcs(31.0, -1)
    mock_inner.solve.return_value.best_match.return_value.astropy_wcs.return_value = solution_wcs
    with patch("astrometry.series_5200") as mock_series, patch(
        "astrometry.Solver", return_value=mock_inner
    ):
        mock_series.index_files.return_value = []
        with PlateSolver(cache_dir="/tmp", scales={0}) as solver:
            result = solver.solve(np.array([[10.0, 20.0]]), image_width=1000, image_height=800)
    assert result is not None
    assert (result.north_deg, result.parity) == pytest.approx(
        wcs_orientation(solution_wcs, 500.0, 400.0)
    )
    assert result.parity == -1 and result.north_deg == pytest.approx(31.0, abs=0.01)


def test_local_index_files_are_chosen_by_scale_and_nothing_is_downloaded(tmp_path):
    folder = tmp_path / "5200"
    folder.mkdir()
    names = (
        "index-5200-10.fits",
        "index-5201-10.fits",
        "index-5201-09.fits",
        "index-5203-10.fits",
        "notes.txt",
    )
    for name in names:
        (folder / name).write_bytes(b"")
    with patch("astrometry.series_5200") as mock_series:
        solver = PlateSolver(cache_dir=tmp_path, scales={1, 0}, download=False)
        mock_series.index_files.assert_not_called()
    assert [f.name for f in solver._index_files] == [
        "index-5200-10.fits",
        "index-5201-09.fits",
        "index-5201-10.fits",
    ]
