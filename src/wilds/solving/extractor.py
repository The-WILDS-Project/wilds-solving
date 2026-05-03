import numpy as np
import sep


def extract_sources(
    image: np.ndarray,
    *,
    threshold: float = 3.0,
    min_area: int = 5,
) -> np.ndarray:
    """Extract and centroid sources in *image* using SEP (SourceExtractor in Python).

    Parameters
    ----------
    image:
        2-D array of pixel values (any numeric dtype).
    threshold:
        Detection threshold in units of the per-pixel background RMS.
    min_area:
        Minimum number of connected pixels above threshold to constitute a source.

    Returns
    -------
    np.ndarray
        Structured array with ``x``, ``y``, ``flux``, and other SEP fields.
        Pass directly to :meth:`~wilds.solving.solver.PlateSolver.solve`.
    """
    data = np.asarray(image, dtype=np.float64)
    bkg = sep.Background(data)
    return sep.extract(data - bkg, threshold, err=bkg.globalrms, minarea=min_area)
