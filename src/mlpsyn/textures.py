"""Texture generation from simple matplotlib drawing primitives.

This is the foundational layer of :mod:`mlpsyn`: it renders abstract textures
without any domain semantics. Scenes are deterministic given a seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import colormaps, patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

IMAGE_SIZE = 600
MAX_SHAPES = 10_000
_DPI = 100


def jitter(rng, color, spread: float = 0.08):
    """Vary ``color`` mostly in luminance, keeping its hue stable.

    A small per-channel term adds just enough chroma variation without turning
    neutral materials into coloured confetti.
    """
    rgb = np.asarray(to_rgb(color), dtype=float)
    rgb = rgb * (1.0 + rng.uniform(-spread, spread))
    rgb = rgb + rng.uniform(-spread * 0.2, spread * 0.2, size=3)
    return tuple(np.clip(rgb, 0.0, 1.0))


def blob(rng, cx, cy, radius, n_vertices=8):
    """A closed irregular polygon around ``(cx, cy)``."""
    angles = np.sort(rng.uniform(0.0, 2 * np.pi, size=n_vertices))
    radii = radius * rng.uniform(0.45, 1.0, size=n_vertices)
    return np.column_stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)])


_jitter = jitter
_blob = blob


class ShapeBudget:
    """Tracks and caps the number of matplotlib artists added to a scene.

    Every texture generator draws through a budget so that a single image can
    never exceed ``MAX_SHAPES`` primitives. Adding beyond the cap raises a
    ``ValueError``; callers can check :attr:`full` to stop early instead.
    """

    def __init__(self, max_shapes: int = MAX_SHAPES):
        if max_shapes <= 0:
            raise ValueError(f"max_shapes must be positive, got {max_shapes}")
        self.max_shapes = max_shapes
        self.count = 0

    @property
    def remaining(self) -> int:
        return self.max_shapes - self.count

    @property
    def full(self) -> bool:
        return self.count >= self.max_shapes

    def _reserve(self) -> None:
        if self.full:
            raise ValueError(
                f"scene exceeds the hard cap of MAX_SHAPES={self.max_shapes} "
                "matplotlib shapes per image"
            )
        self.count += 1

    def add_patch(self, ax, patch):
        self._reserve()
        ax.add_patch(patch)
        return patch

    def add_artist(self, artist):
        """Register an already-created artist (e.g. a scatter collection)."""
        self._reserve()
        return artist

    def add_line(self, ax, x, y, **kwargs):
        self._reserve()
        (line,) = ax.plot(x, y, **kwargs)
        return line

    def add_line_collection(self, ax, segments, **kwargs):
        """Add many 2-point segments as one LineCollection (much faster)."""
        self._reserve()
        collection = LineCollection(segments, **kwargs)
        ax.add_collection(collection)
        return collection


def render_scene(size: int, background, draw, seed=None, max_shapes: int = MAX_SHAPES):
    """Render a square scene by letting ``draw(ax, rng, budget)`` populate it.

    Parameters
    ----------
    size : int
        Width and height of the square output image in pixels.
    background : matplotlib color or numpy.ndarray
        Background colour, or an RGB image to use as the background.
    draw : callable
        Called as ``draw(ax, rng, budget)`` and should add artists to ``ax``
        using the provided :class:`ShapeBudget`.
    seed : int or None, default=None
        Seed for the random generator; makes the output deterministic.
    max_shapes : int, default=MAX_SHAPES
        Hard cap on the number of artists the scene may contain.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(size, size, 3)`` and dtype ``uint8``.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")

    rng = np.random.default_rng(seed)
    fig = Figure(figsize=(size / _DPI, size / _DPI), dpi=_DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    if isinstance(background, np.ndarray):
        fig.patch.set_facecolor("black")
        ax.imshow(background, extent=(0, 1, 0, 1), zorder=0, interpolation="nearest")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        fig.patch.set_facecolor(background)
        ax.set_facecolor(background)

    budget = ShapeBudget(max_shapes)
    draw(ax, rng, budget)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return rgba[..., :3].copy()


def _random_polygon(rng, cx, cy, radius, n_vertices):
    """An irregular closed polygon around a centre."""
    angles = np.sort(rng.uniform(0, 2 * np.pi, size=n_vertices))
    radii = radius * rng.uniform(0.4, 1.0, size=n_vertices)
    return np.column_stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)])


def _add_shape(ax, rng, color, budget):
    """Draw one random material speckle (polygon or line)."""
    kind = rng.integers(0, 5)
    cx, cy = rng.uniform(0.0, 1.0, size=2)
    angle = rng.uniform(0, 360)

    if kind == 0:
        w, h = rng.uniform(0.02, 0.35, size=2)
        artist = patches.Rectangle(
            (cx - w / 2, cy - h / 2), w, h, angle=angle, facecolor=color, edgecolor="none"
        )
        budget.add_patch(ax, artist)
    elif kind == 1:
        radius = rng.uniform(0.01, 0.2)
        artist = patches.Circle((cx, cy), radius, facecolor=color, edgecolor="none")
        budget.add_patch(ax, artist)
    elif kind == 2:
        radius = rng.uniform(0.02, 0.25)
        vertices = _random_polygon(rng, cx, cy, radius, rng.integers(3, 9))
        artist = patches.Polygon(vertices, closed=True, facecolor=color, edgecolor="none")
        budget.add_patch(ax, artist)
    elif kind == 3:
        width = rng.uniform(0.05, 0.6)
        height = rng.uniform(0.02, 0.15)
        artist = patches.Ellipse(
            (cx, cy),
            width,
            height,
            angle=angle,
            facecolor=color,
            edgecolor="none",
        )
        budget.add_patch(ax, artist)
    else:
        x0, y0, x1, y1 = rng.uniform(0.0, 1.0, size=4)
        budget.add_line(
            ax,
            [x0, x1],
            [y0, y1],
            color=color,
            linewidth=rng.uniform(0.5, 4.0),
            solid_capstyle="round",
        )


def generate_texture(
    size: int = IMAGE_SIZE,
    n_shapes: int = 2_000,
    seed: int | None = None,
    cmap: str = "terrain",
    background: str | None = None,
):
    """Render a texture of ``size`` x ``size`` pixels using ``n_shapes`` shapes.

    Parameters
    ----------
    size : int, default=600
        Width and height of the square output image in pixels.
    n_shapes : int, default=2000
        Number of matplotlib primitives to draw. Must not exceed ``MAX_SHAPES``.
    seed : int or None, default=None
        Seed for the random generator; makes the output deterministic.
    cmap : str, default="terrain"
        Name of a matplotlib colormap used to sample shape colours.
    background : str or None, default=None
        Background colour. Defaults to the darkest colour of ``cmap``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(size, size, 3)`` and dtype ``uint8``.
    """
    if n_shapes < 0:
        raise ValueError(f"n_shapes must be non-negative, got {n_shapes}")
    if n_shapes > MAX_SHAPES:
        raise ValueError(
            f"n_shapes={n_shapes} exceeds the hard cap of MAX_SHAPES={MAX_SHAPES} "
            "matplotlib shapes per image"
        )

    colormap = colormaps[cmap]
    if background is None:
        background = colormap(0.0)

    def draw(ax, rng, budget):
        for color in colormap(rng.random(n_shapes)):
            if budget.full:
                break
            _add_shape(ax, rng, color, budget)

    return render_scene(size, background, draw, seed=seed)


def save_texture(image, path: str | Path) -> Path:
    """Save an RGB array produced by :func:`generate_texture` as a PNG file."""
    from PIL import Image

    path = Path(path)
    Image.fromarray(image).save(path)
    return path