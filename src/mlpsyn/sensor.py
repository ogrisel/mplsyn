"""Sensor and atmospheric post-processing for rendered images.

These functions take a rendered ``uint8`` RGB image and return a degraded
version, emulating what a real sensor or the atmosphere would add:

- atmospheric path: haze / thin fog, then cloud cover with soft shadows;
- optics: optical blur;
- detector: Gaussian read noise and Poisson (photon) shot noise.

They operate on the final image, so they compose with any material, object or
scene. :func:`apply_sensor` owns the whole chain and enforces the physical
order (**atmosphere before noise**); :data:`EFFECTS` registers the one-argument
variants used by the per-effect examples.

Noise is deliberately mild and bounded: :data:`GAUSSIAN_SIGMA` and
:data:`POISSON_SCALE` are tuned to give a comparable standard deviation, and
user-supplied values are clamped so no image is ever heavily noised.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

__all__ = [
    "EFFECTS",
    "GAUSSIAN_SIGMA",
    "MAX_GAUSSIAN_SIGMA",
    "MIN_POISSON_SCALE",
    "POISSON_SCALE",
    "apply_sensor",
    "atmosphere",
    "blur",
    "cloud_alpha",
    "clouds",
    "gaussian_noise",
    "haze",
    "poisson_noise",
]

#: Default Gaussian read-noise standard deviation, in 0–255 units.
GAUSSIAN_SIGMA = 8.0
#: Hard cap on Gaussian read noise: never noisy enough to hide the scene.
MAX_GAUSSIAN_SIGMA = 14.0
#: Default Poisson full-well count; higher means less shot noise.
POISSON_SCALE = 600.0
#: Lower bound on the full-well count (i.e. upper bound on shot noise).
MIN_POISSON_SCALE = 220.0


def _to_float(image: np.ndarray) -> np.ndarray:
    """Convert a uint8 image to float32 in [0, 1]."""
    return image.astype(np.float64) / 255.0


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Clip a float image in [0, 1] back to uint8."""
    return np.clip(image * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


def _box_blur(array: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur for float arrays (approximates a Gaussian)."""
    if radius <= 0:
        return array
    k = 2 * radius + 1
    for axis in (0, 1):
        moved = np.moveaxis(array, axis, -1)
        pad = np.pad(moved, ((radius, radius), (0, 0)), mode="edge")
        cumulative = np.cumsum(pad, axis=0)
        cumulative = np.concatenate([np.zeros((1, cumulative.shape[1])), cumulative], axis=0)
        moved = (cumulative[k:] - cumulative[:-k]) / k
        array = np.moveaxis(moved, -1, axis)
    return array


def gaussian_noise(image: np.ndarray, rng, sigma: float = GAUSSIAN_SIGMA) -> np.ndarray:
    """Additive Gaussian read noise (``sigma`` in 0–255 units, clamped)."""
    sigma = float(np.clip(sigma, 0.0, MAX_GAUSSIAN_SIGMA))
    noisy = _to_float(image) + rng.normal(0.0, sigma / 255.0, image.shape)
    return _to_uint8(noisy)


def poisson_noise(image: np.ndarray, rng, scale: float = POISSON_SCALE) -> np.ndarray:
    """Poisson photon noise: simulate counts at a given ``scale`` (full-well).

    ``scale`` is clamped to at least :data:`MIN_POISSON_SCALE` so the shot noise
    stays mild.
    """
    scale = float(max(scale, MIN_POISSON_SCALE))
    counts = rng.poisson(np.clip(_to_float(image), 0.0, 1.0) * scale)
    return _to_uint8(counts / scale)


def blur(image: np.ndarray, rng, radius: float = 1.2) -> np.ndarray:
    """Optical blur (PIL Gaussian)."""
    return np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius)))


def haze(image: np.ndarray, rng, strength: float = 0.3) -> np.ndarray:
    """Thin fog: blend toward a pale sky colour, stronger near the top."""
    height = image.shape[0]
    vertical = np.linspace(1.2, 0.6, height)[:, None, None]
    fog = np.array([0.82, 0.84, 0.87])
    blended = _to_float(image) * (1.0 - strength * vertical) + fog * (strength * vertical)
    return _to_uint8(blended)


def _sample_opacity(rng) -> float:
    """Per-scene cloud opacity.

    75 % of cloudy scenes get an **opaque core** (opacity ~1); the remaining
    25 % stay **semi-transparent**, so the ground shows through.
    """
    if rng.random() < 0.75:
        return float(rng.uniform(0.95, 1.0))
    return float(rng.uniform(0.45, 0.6))


def cloud_alpha(shape, rng, coverage: float = 0.4, opacity: float | None = None) -> np.ndarray:
    """Soft cloud-occlusion field (0 = clear, 1 = fully clouded) for ``shape``.

    ``opacity`` caps how white a cloud gets. When it is ``None`` (the default) a
    per-scene value is drawn: most scenes keep a fully opaque core, a quarter are
    left semi-transparent (see :func:`_sample_opacity`).
    """
    height, width = shape[:2]
    if opacity is None:
        opacity = _sample_opacity(rng)
    field = np.zeros((height, width))
    total = 0.0
    for cells, weight in [(4, 1.0), (8, 0.6), (16, 0.35), (32, 0.2)]:
        low = rng.random((cells, cells))
        up = np.asarray(
            Image.fromarray((low * 255).astype(np.uint8)).resize((width, height), Image.BICUBIC),
            dtype=np.float64,
        )
        field += weight * up / 255.0
        total += weight
    field /= total
    field = _box_blur(field, max(2, height // 90))
    low_p, high_p = np.percentile(field, [100 - 80 * coverage, 100 - 30 * coverage])
    alpha = np.clip((field - low_p) / (high_p - low_p + 1e-6), 0.0, 1.0)
    return alpha**1.3 * float(np.clip(opacity, 0.05, 1.0))


def clouds(
    image: np.ndarray,
    rng,
    coverage: float = 0.4,
    shadow: float = 0.2,
    opacity: float | None = None,
    return_mask: bool = False,
):
    """Soft cloud cover with an offset shadow on the ground.

    With ``return_mask=True`` also returns the soft occlusion field (the label
    mask for cloud segmentation).
    """
    height = image.shape[0]
    alpha = cloud_alpha(image.shape, rng, coverage, opacity)
    shifted = np.roll(alpha, shift=max(2, int(0.05 * height)), axis=1)
    shifted = np.roll(shifted, shift=max(2, int(0.035 * height)), axis=0)
    out = _to_float(image) * (1.0 - shadow * shifted[..., None])
    white = np.array([1.0, 0.99, 0.97])
    out = out * (1.0 - alpha[..., None]) + white * alpha[..., None]
    result = _to_uint8(out)
    if return_mask:
        return result, alpha
    return result


def atmosphere(
    image: np.ndarray,
    rng,
    haze_strength: float = 0.0,
    cloud_coverage: float = 0.0,
    shadow: float = 0.2,
    cloud_opacity: float | None = None,
) -> np.ndarray:
    """Atmospheric path: haze first (lower layer), then cloud occlusion."""
    out = image
    if haze_strength > 0:
        out = haze(out, rng, strength=haze_strength)
    if cloud_coverage > 0:
        out = clouds(out, rng, coverage=cloud_coverage, shadow=shadow, opacity=cloud_opacity)
    return out


def _clean(image: np.ndarray, rng) -> np.ndarray:
    """Identity sensor effect (no degradation)."""
    return image.copy()


def apply_sensor(
    image: np.ndarray,
    rng,
    kind: str | None = None,
    blur_radius: float = 0.0,
    haze_strength: float = 0.0,
    cloud_coverage: float = 0.0,
    shadow: float = 0.2,
    cloud_opacity: float | None = None,
    sigma: float = GAUSSIAN_SIGMA,
    scale: float = POISSON_SCALE,
) -> np.ndarray:
    """Apply the full degradation chain in physical order.

    Order is **atmosphere → optics → detector**:

    1. haze / cloud occlusion (the light path through the atmosphere);
    2. optical blur (the instrument point-spread function);
    3. sensor noise (Gaussian or Poisson).

    Noise is therefore never added before the atmospheric effects. All noise
    magnitudes are bounded (see :data:`MAX_GAUSSIAN_SIGMA` and
    :data:`MIN_POISSON_SCALE`).
    """
    out = atmosphere(image, rng, haze_strength, cloud_coverage, shadow, cloud_opacity)
    if blur_radius > 0:
        out = blur(out, rng, radius=blur_radius)
    if kind == "gaussian":
        out = gaussian_noise(out, rng, sigma=sigma)
    elif kind == "poisson":
        out = poisson_noise(out, rng, scale=scale)
    return out


EFFECTS = {
    "clean": _clean,
    "gaussian": lambda image, rng: gaussian_noise(image, rng),
    "poisson": lambda image, rng: poisson_noise(image, rng),
    "blur": lambda image, rng: blur(image, rng),
    "haze": lambda image, rng: haze(image, rng),
    "clouds": lambda image, rng: clouds(image, rng),
}