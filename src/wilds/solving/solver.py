import asyncio
import functools
import logging
import math
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import astrometry
import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

logger = logging.getLogger(__name__)

_LOGODDS_THRESHOLD = math.log(1e12)

Confidence = Literal["poor", "good", "high", "absolute"]

# Logodds breakpoints for confidence levels.
# Minimum accepted solution is log(1e12) ≈ 27.6.
# The reference test solve scored 122 — well into "high".
_CONFIDENCE_THRESHOLDS: list[tuple[float, Confidence]] = [
    (math.log(1e20), "poor"),       # 27.6 – 46.1  : accepted but marginal
    (math.log(1e40), "good"),       # 46.1 – 92.1  : solid match
    (math.log(1e80), "high"),       # 92.1 – 184.2 : very confident
    (math.inf,       "absolute"),   # > 184.2       : essentially certain
]


def _confidence_from_logodds(logodds: float) -> Confidence:
    for threshold, label in _CONFIDENCE_THRESHOLDS:
        if logodds < threshold:
            return label
    return "absolute"


@dataclass(frozen=True)
class ScaleHint:
    """Plate-scale bounds for the solver.

    For the WILDS guide camera (ZWO ASI294MM, 4.63 µm pixels) at LDT f/6.1
    the native plate scale is ~0.037 arcsec/px at the main focus; the actual
    value at the guide camera port depends on the relay optics used.
    """

    lower_arcsec_per_pixel: float
    upper_arcsec_per_pixel: float


@dataclass(frozen=True)
class PositionHint:
    """Approximate sky position to constrain the search radius.

    Use mount encoder values when available — dramatically reduces solve time.
    """

    ra_deg: float
    dec_deg: float
    radius_deg: float = 2.0


@dataclass(frozen=True)
class SolveResult:
    """Result of a successful plate solve."""

    wcs: WCS
    ra_deg: float               # sky coords of the image centre
    dec_deg: float
    pixel_scale_arcsec: float
    logodds: float              # log-odds of the winning match (higher = more confident)
    confidence: Confidence      # qualitative label derived from logodds


class PlateSolver:
    """Plate solver wrapping the astrometry.net C library (in-process, no subprocess).

    Construct once at startup, then call :meth:`solve` or :meth:`solve_async`
    repeatedly — index files are loaded into memory on :meth:`__enter__` and
    held for the lifetime of the context.

    Parameters
    ----------
    cache_dir:
        Directory for astrometry.net Series 5200 index files.  Downloaded
        automatically on first use if not already present.
    scales:
        Set of Series 5200 scale indices to load.  For a typical guide-camera
        FOV at a 4 m telescope, scales ``{0, 1, 2, 3}`` cover ~0.5–5 arcsec/px.

    Examples
    --------
    ::

        from wilds.solving import PlateSolver, ScaleHint, PositionHint, extract_sources

        with PlateSolver(cache_dir="~/.astrometry", scales={0, 1, 2, 3}) as solver:
            sources = extract_sources(frame)
            result = solver.solve(
                sources,
                image_width=frame.shape[1],
                image_height=frame.shape[0],
                scale_hint=ScaleHint(0.5, 2.0),
                position_hint=PositionHint(ra_deg=83.8, dec_deg=-5.4),
            )
            if result is not None:
                wcs = result.wcs
    """

    def __init__(
        self,
        cache_dir: str | Path,
        scales: set[int],
        *,
        download: bool = True,
    ) -> None:
        resolved = Path(cache_dir).expanduser().resolve()
        if download:
            self._index_files = astrometry.series_5200.index_files(
                cache_directory=str(resolved),
                scales=scales,
            )
        else:
            # Use only files already present — no network access.
            # series_5200 stores files under <cache_dir>/5200/.
            self._index_files = sorted((resolved / "5200").glob("*.fits"))
            logger.info("Found %d local index file(s) in %s", len(self._index_files), resolved / "5200")
        self._solver: astrometry.Solver | None = None

    def __enter__(self) -> "PlateSolver":
        self._solver = astrometry.Solver(self._index_files)
        self._solver.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        if self._solver is not None:
            self._solver.__exit__(exc_type, exc_val, exc_tb)
            self._solver = None

    def solve(
        self,
        stars: np.ndarray,
        image_width: int,
        image_height: int,
        *,
        scale_hint: ScaleHint | None = None,
        position_hint: PositionHint | None = None,
    ) -> SolveResult | None:
        """Plate-solve a list of star pixel positions.

        Parameters
        ----------
        stars:
            Source positions — either a structured array with ``x``/``y`` fields
            (direct output of :func:`~wilds.solving.extractor.extract_sources`)
            or a plain ``(N, 2)`` array of ``[x, y]`` pairs.
        image_width, image_height:
            Sensor dimensions in pixels; used to compute the sky position of
            the image centre in the returned :class:`SolveResult`.
        scale_hint:
            Plate-scale bounds.  Strongly recommended — dramatically narrows
            the search.
        position_hint:
            Approximate sky position (e.g. from mount encoders).

        Returns
        -------
        SolveResult or None
            ``None`` if no match was found.  Check the log (``wilds.solving.solver``,
            WARNING level) for the best log-odds reached during the attempt —
            useful for diagnosing why the solve failed.
        """
        if self._solver is None:
            raise RuntimeError("PlateSolver must be used as a context manager")

        if stars.dtype.names is not None:
            xy = np.column_stack([stars["x"], stars["y"]])
        else:
            xy = np.asarray(stars, dtype=float)

        size_hint = (
            astrometry.SizeHint(
                lower_arcsec_per_pixel=scale_hint.lower_arcsec_per_pixel,
                upper_arcsec_per_pixel=scale_hint.upper_arcsec_per_pixel,
            )
            if scale_hint is not None
            else None
        )

        pos_hint = (
            astrometry.PositionHint(
                ra_deg=position_hint.ra_deg,
                dec_deg=position_hint.dec_deg,
                radius_deg=position_hint.radius_deg,
            )
            if position_hint is not None
            else None
        )

        # Track the best log-odds seen during the solve so we can report it on failure.
        peak_logodds: list[float] = [float("-inf")]

        def _track_logodds(logodds_list: list[float]) -> astrometry.Action:
            if logodds_list:
                peak_logodds[0] = max(peak_logodds[0], max(logodds_list))
            return astrometry.Action.CONTINUE

        solution = self._solver.solve(
            stars=xy,
            size_hint=size_hint,
            position_hint=pos_hint,
            solution_parameters=astrometry.SolutionParameters(
                sip_order=3,
                output_logodds_threshold=_LOGODDS_THRESHOLD,
                logodds_callback=_track_logodds,
            ),
        )

        if not solution.has_match():
            logger.warning(
                "No solution. stars=%d  scale=%s  position=%s  "
                "peak_logodds=%.1f  threshold=%.1f",
                len(xy),
                f"{scale_hint.lower_arcsec_per_pixel:.3f}–{scale_hint.upper_arcsec_per_pixel:.3f}\""
                if scale_hint else "none",
                f"RA {position_hint.ra_deg:.3f}° Dec {position_hint.dec_deg:.3f}° r={position_hint.radius_deg}°"
                if position_hint else "none",
                peak_logodds[0],
                _LOGODDS_THRESHOLD,
            )
            return None

        match = solution.best_match()
        wcs = match.astropy_wcs()

        cx, cy = image_width / 2.0, image_height / 2.0
        ra, dec = wcs.all_pix2world([[cx, cy]], 0)[0]

        scales_deg = proj_plane_pixel_scales(wcs)
        pixel_scale = float(np.mean(scales_deg)) * 3600.0  # → arcsec/px

        return SolveResult(
            wcs=wcs,
            ra_deg=float(ra),
            dec_deg=float(dec),
            pixel_scale_arcsec=pixel_scale,
            logodds=match.logodds,
            confidence=_confidence_from_logodds(match.logodds),
        )

    async def solve_async(
        self,
        stars: np.ndarray,
        image_width: int,
        image_height: int,
        *,
        scale_hint: ScaleHint | None = None,
        position_hint: PositionHint | None = None,
    ) -> SolveResult | None:
        """Async wrapper — runs :meth:`solve` in the default thread executor.

        The underlying C solver is CPU-bound and not async-native; running it
        in an executor keeps the event loop free during a solve.
        """
        loop = asyncio.get_running_loop()
        fn = functools.partial(
            self.solve,
            stars,
            image_width,
            image_height,
            scale_hint=scale_hint,
            position_hint=position_hint,
        )
        return await loop.run_in_executor(None, fn)
