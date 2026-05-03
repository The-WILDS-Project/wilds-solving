import numpy as np
import pytest


def make_synthetic_image(
    shape: tuple[int, int] = (256, 256),
    n_stars: int = 15,
    sky: float = 100.0,
    sky_noise: float = 5.0,
    star_flux: float = 2000.0,
    fwhm: float = 3.0,
    seed: int = 42,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Return a synthetic image and the injected star positions."""
    rng = np.random.default_rng(seed)
    image = rng.normal(sky, sky_noise, shape)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    positions: list[tuple[float, float]] = []
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    margin = 20
    for _ in range(n_stars):
        x = rng.uniform(margin, shape[1] - margin)
        y = rng.uniform(margin, shape[0] - margin)
        image = image + star_flux * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
        positions.append((float(x), float(y)))
    return image, positions


@pytest.fixture
def synthetic_image() -> tuple[np.ndarray, list[tuple[float, float]]]:
    return make_synthetic_image()
