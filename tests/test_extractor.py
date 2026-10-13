import numpy as np

from wilds.solving.extractor import extract_sources

from .conftest import make_synthetic_image


def test_returns_structured_array(synthetic_image):
    image, _ = synthetic_image
    sources = extract_sources(image)
    assert sources.dtype.names is not None
    assert "x" in sources.dtype.names
    assert "y" in sources.dtype.names
    assert "flux" in sources.dtype.names


def test_finds_most_injected_stars(synthetic_image):
    image, positions = synthetic_image
    sources = extract_sources(image, threshold=5.0)
    assert len(sources) >= len(positions) * 0.8


def test_positions_accurate():
    """Single bright star at a known pixel position should be recovered within 1 px."""
    shape = (128, 128)
    rng = np.random.default_rng(0)
    image = rng.normal(100.0, 3.0, shape).astype(np.float64)
    x0, y0 = 64.3, 48.7
    sigma = 2.0
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    image += 8000.0 * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma**2))

    sources = extract_sources(image, threshold=5.0)
    assert len(sources) >= 1

    dists = np.hypot(sources["x"] - x0, sources["y"] - y0)
    assert dists.min() < 1.0


def test_min_area_filters_noise_peaks():
    """Very small min_area should detect more sources than a large one."""
    image, _ = make_synthetic_image(sky_noise=8.0, seed=7)
    sources_strict = extract_sources(image, threshold=3.0, min_area=20)
    sources_loose = extract_sources(image, threshold=3.0, min_area=2)
    assert len(sources_loose) >= len(sources_strict)


def test_accepts_integer_image(synthetic_image):
    image, _ = synthetic_image
    sources = extract_sources(image.astype(np.uint16))
    assert len(sources) > 0
