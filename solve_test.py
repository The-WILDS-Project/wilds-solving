#!/usr/bin/env python3
"""
Test wilds-solving against real MASTW acquisition frames.

Index files (~360 MB for scales 0-2) are downloaded to ~/.astrometry on
first run.  Set WILDS_ASTROMETRY_CACHE to override the cache directory.

Usage:
    uv run python solve_test.py
"""
import logging
import os
import re
import warnings
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from wilds.solving import PlateSolver, PositionHint, ScaleHint, extract_sources

_SEARCH_ROOTS = [
    Path("/Storage/mast-share/MAST/mastw"),
    Path("/Storage/mast-share/MAST/mast00"),
]

FITS_FILES: list[Path] = sorted(
    p
    for root in _SEARCH_ROOTS
    if root.exists()
    for p in root.rglob("*solver=*.fits")
)

CACHE_DIR = os.environ.get("WILDS_ASTROMETRY_CACHE", "~/.astrometry")
SCALES = {0, 1, 2}
MAX_STARS = 50      # astrometry.net is fast and reliable with a small bright-star list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_filename_coords(path: Path) -> tuple[float, float]:
    """Extract RA (deg), Dec (deg) from ``target=RA_hours,Dec_deg`` in filename."""
    m = re.search(r"target=([0-9.]+),([0-9.-]+)", path.name)
    if m is None:
        raise ValueError(f"No target= found in: {path.name}")
    return float(m.group(1)) * 15.0, float(m.group(2))  # hours → deg, deg


def load_frame(path: Path) -> tuple[np.ndarray, WCS | None]:
    with fits.open(path) as hdul:
        data = hdul[0].data.copy()
        hdr = hdul[0].header
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wcs = WCS(hdr) if "CTYPE1" in hdr else None
    if wcs is not None and not wcs.has_celestial:
        wcs = None
    return data, wcs


def center_coords(wcs: WCS, width: int, height: int) -> tuple[float, float]:
    ra, dec = wcs.all_pix2world([[width / 2.0, height / 2.0]], 0)[0]
    return float(ra), float(dec)


def pixel_scale_arcsec(wcs: WCS) -> float:
    return float(np.mean(proj_plane_pixel_scales(wcs))) * 3600.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    with PlateSolver(cache_dir=CACHE_DIR, scales=SCALES, download=False) as solver:
        for fits_path_str in FITS_FILES:
            path = Path(fits_path_str)
            print(f"\n{'=' * 72}")
            print(f"File : {path.name}")

            data, ref_wcs = load_frame(path)
            h, w = data.shape

            # --- position and scale hints ---
            if ref_wcs is not None:
                hint_ra, hint_dec = center_coords(ref_wcs, w, h)
                ref_scale = pixel_scale_arcsec(ref_wcs)
                hint_source = "WCS (CRVAL centre)"
                scale_hint = ScaleHint(ref_scale * 0.85, ref_scale * 1.15)
            else:
                hint_ra, hint_dec = parse_filename_coords(path)
                ref_scale = None
                hint_source = "filename (target=)"
                scale_hint = ScaleHint(0.20, 0.35)  # rough prior for this camera

            print(f"Hint : {hint_source}  →  RA {hint_ra:.5f}°  Dec {hint_dec:+.5f}°")
            if ref_scale is not None:
                print(f"       ref scale {ref_scale:.4f} arcsec/px")

            # --- source extraction ---
            print("Extracting sources...", end=" ", flush=True)
            sources = extract_sources(data, threshold=3.0)
            # keep the brightest MAX_STARS only — matches original solver behaviour
            if len(sources) > MAX_STARS:
                idx = np.argsort(sources["flux"])[::-1][:MAX_STARS]
                sources = sources[idx]
            print(f"{len(sources)} used")

            # --- solve ---
            print("Solving...", end=" ", flush=True)
            result = solver.solve(
                sources,
                image_width=w,
                image_height=h,
                scale_hint=scale_hint,
                position_hint=PositionHint(ra_deg=hint_ra, dec_deg=hint_dec, radius_deg=1.0),
            )

            if result is None:
                print("NO SOLUTION")
                continue

            print(f"RA {result.ra_deg:.5f}°  Dec {result.dec_deg:+.5f}°  "
                  f"scale {result.pixel_scale_arcsec:.4f} arcsec/px  "
                  f"logodds {result.logodds:.1f}  [{result.confidence}]")

            if ref_wcs is not None:
                ref_ra, ref_dec = center_coords(ref_wcs, w, h)
                d_ra  = (result.ra_deg  - ref_ra)  * 3600.0 * np.cos(np.radians(hint_dec))
                d_dec = (result.dec_deg - ref_dec) * 3600.0
                sep   = np.hypot(d_ra, d_dec)
                d_scale = result.pixel_scale_arcsec - ref_scale
                print(f"Ref  : RA {ref_ra:.5f}°  Dec {ref_dec:+.5f}°  "
                      f"scale {ref_scale:.4f} arcsec/px")
                print(f"Δ    : {sep:.2f}\"  (ΔRA {d_ra:+.2f}\"  ΔDec {d_dec:+.2f}\"  "
                      f"Δscale {d_scale:+.5f} arcsec/px)")


if __name__ == "__main__":
    main()
