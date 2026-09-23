"""Low-level material textures, reused as fills by objects and backgrounds.

A *material* is a small, dense texture describing a surface: grainy asphalt,
grass, foliage, soil, water, bark, roof covering, ... Materials sit below
objects and backgrounds in the stack. Objects fill their geometry with a
material via :func:`fill_patch`, and scene grounds are built from materials too.

Each material exposes a function ``material_x(size=600, seed=None) -> ndarray``
and is registered in :data:`MATERIALS`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from matplotlib import patches
from matplotlib.colors import to_rgb
from matplotlib.transforms import Affine2D, TransformedPatchPath

from mlpsyn.textures import IMAGE_SIZE, _blob, _jitter, render_scene

__all__ = [
    "MATERIALS",
    "MATERIAL_DESCRIPTIONS",
    "asphalt",
    "bark",
    "concrete",
    "crop",
    "fill_patch",
    "foliage",
    "generate_material",
    "grass",
    "gravel",
    "mud",
    "rock",
    "roof",
    "sand",
    "snow",
    "soil",
    "water",
    "material_textures",
]


def _area(size, per_shape):
    return max(1, int(size * size / per_shape))


def _color_noise(rng, base, spread, n):
    rgb = np.asarray(to_rgb(base), dtype=float)
    factors = 1.0 + rng.uniform(-spread, spread, size=(n, 1))
    chroma = rng.uniform(-spread * 0.05, spread * 0.05, size=(n, 3))
    return np.clip(rgb * factors + chroma, 0.0, 1.0)


def _speckle(ax, rng, budget, n, base, spread=0.2, s=3.0, marker=".", alpha=1.0):
    points = rng.random((n, 2))
    colors = _color_noise(rng, base, spread, n)
    collection = ax.scatter(
        points[:, 0],
        points[:, 1],
        s=s,
        c=colors,
        marker=marker,
        linewidths=0,
        alpha=alpha,
    )
    budget.add_artist(collection)


def _mottle(ax, rng, budget, n, base, spread=0.1, s=200.0, alpha=0.08):
    """Soft tonal mottling from many large, faint marks (no hard discs)."""
    points = rng.random((n, 2))
    colors = _color_noise(rng, base, spread, n)
    collection = ax.scatter(
        points[:, 0], points[:, 1], s=s, c=colors, marker="o", linewidths=0, alpha=alpha
    )
    budget.add_artist(collection)


def _cover(ax, rng, budget, n, base, spread, rmin, rmax, alpha=1.0):
    for _ in range(n):
        if budget.full:
            break
        cx, cy = rng.uniform(0.0, 1.0, size=2)
        budget.add_patch(
            ax,
            patches.Circle(
                (cx, cy),
                rng.uniform(rmin, rmax),
                facecolor=_jitter(rng, base, spread),
                edgecolor="none",
                alpha=alpha,
            ),
        )


def _lines(ax, rng, budget, n, base, spread, vertical=False, width=1.0):
    for _ in range(n):
        if budget.full:
            break
        position = rng.uniform(0.0, 1.0)
        offset = rng.uniform(0.0, 1.0)
        length = rng.uniform(0.05, 0.5)
        if vertical:
            x = [position, position]
            y = [offset, offset + length]
        else:
            x = [offset, offset + length]
            y = [position, position]
        budget.add_line(ax, x, y, color=_jitter(rng, base, spread), linewidth=width)


def _ripples(ax, rng, budget, n, base, spread, amplitude=0.008, width=0.6):
    """Short wavy water streaks, so water does not look banded by straight lines."""
    for _ in range(n):
        if budget.full:
            break
        y = rng.uniform(0.0, 1.0)
        x0 = rng.uniform(-0.1, 0.6)
        length = rng.uniform(0.15, 0.5)
        phase = rng.uniform(0.0, 2 * np.pi)
        t = np.linspace(0.0, 1.0, 16)
        x = x0 + length * t
        yy = y + amplitude * np.sin(2 * np.pi * 2.0 * t + phase)
        budget.add_line(ax, x, yy, color=_jitter(rng, base, spread), linewidth=width)


def fill_patch(ax, patch, texture, budget=None, zorder=2, rotation=0.0, center=None, alpha=1.0):
    """Fill ``patch`` with a material ``texture`` array, clipped to its shape.

    The material is drawn as an image over the whole axes and clipped by the
    patch path, so the same texture field is shared by every object using it.
    When ``center`` and ``rotation`` are given, the texture field is rotated
    about ``center`` so it stays aligned with a rotated object. ``alpha`` lets a
    fill blend with what is beneath it (used for soft ground patches).
    """
    patch.set_facecolor("none")
    patch.set_zorder(zorder)
    ax.add_patch(patch)
    transform = ax.transData
    if center is not None and rotation:
        transform = Affine2D().rotate_deg_around(center[0], center[1], rotation) + ax.transData
    image = ax.imshow(
        texture,
        extent=(0, 1, 0, 1),
        interpolation="nearest",
        zorder=zorder,
        transform=transform,
        alpha=alpha,
    )
    image.set_clip_path(TransformedPatchPath(patch))
    if budget is not None:
        budget.add_artist(image)
    return image


def _cross_lines(ax, rng, budget, angle_deg, spacing, color, width=0.8):
    """Parallel lines at ``angle_deg`` spanning the whole unit square."""
    theta = np.deg2rad(angle_deg)
    direction = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-np.sin(theta), np.cos(theta)])
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    offsets = corners @ normal
    for offset in np.arange(offsets.min() + spacing, offsets.max(), spacing):
        center = offset * normal
        p0, p1 = center - 1.5 * direction, center + 1.5 * direction
        budget.add_line(
            ax, [p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=width
        )


def _cracks(ax, rng, budget, n, base, spread, width=0.6):
    """Short, bent cracks (random walk) instead of long straight scratches."""
    for _ in range(n):
        if budget.full:
            break
        x, y = rng.uniform(0.0, 1.0, size=2)
        angle = rng.uniform(0.0, 2 * np.pi)
        points = [(x, y)]
        for _ in range(int(rng.integers(3, 6))):
            angle += rng.uniform(-0.6, 0.6)
            length = rng.uniform(0.02, 0.06)
            x += length * np.cos(angle)
            y += length * np.sin(angle)
            points.append((x, y))
        points = np.asarray(points)
        budget.add_line(
            ax, points[:, 0], points[:, 1], color=_jitter(rng, base, spread), linewidth=width
        )


def grass(size: int = IMAGE_SIZE, seed: int | None = None):
    """Grainy green grass with tonal patches."""

    def draw(ax, rng, budget):
        _mottle(ax, rng, budget, _area(size, 300), "#87a24e", spread=0.10, s=340, alpha=0.10)
        _mottle(ax, rng, budget, _area(size, 520), "#5f7d33", spread=0.10, s=380, alpha=0.07)
        _speckle(ax, rng, budget, _area(size, 70), "#5f7d33", spread=0.35, s=3.5)
        # short blade ticks break up the flat fill
        _speckle(ax, rng, budget, _area(size, 55), "#557a2c", spread=0.30, s=6.0, marker="|")
        _speckle(ax, rng, budget, _area(size, 90), "#9cb35e", spread=0.22, s=5.0, marker="|")

    return render_scene(size, "#6f8f3f", draw, seed=seed)


def foliage(size: int = IMAGE_SIZE, seed: int | None = None):
    """Clumpy leaf canopy with highlights and shadowed gaps."""

    def draw(ax, rng, budget):
        _cover(ax, rng, budget, _area(size, 260), "#2f6b2c", 0.2, 0.012, 0.035)
        _cover(ax, rng, budget, _area(size, 1100), "#3f7a2f", 0.2, 0.004, 0.012)
        _cover(ax, rng, budget, _area(size, 2200), "#6aa63f", 0.15, 0.002, 0.006, alpha=0.7)

    return render_scene(size, "#1c3d1a", draw, seed=seed)


def asphalt(size: int = IMAGE_SIZE, seed: int | None = None):
    """Grainy asphalt with aggregate speckle and a few cracks."""

    def draw(ax, rng, budget):
        _speckle(ax, rng, budget, _area(size, 35), "#6f6f74", spread=0.35, s=2.0)
        _speckle(ax, rng, budget, _area(size, 120), "#2a2a2d", spread=0.25, s=3.0)
        _cracks(ax, rng, budget, int(rng.integers(3, 8)), "#1f1f22", 0.1, width=0.6)

    return render_scene(size, "#3c3c40", draw, seed=seed)


def concrete(size: int = IMAGE_SIZE, seed: int | None = None):
    """Light mottled concrete with expansion joints and potholes."""

    def draw(ax, rng, budget):
        _mottle(ax, rng, budget, _area(size, 500), "#b0ada6", spread=0.05, s=260, alpha=0.08)
        for _ in range(int(rng.integers(2, 5))):
            angle = rng.uniform(0.0, 180.0)
            _cross_lines(
                ax,
                rng,
                budget,
                angle,
                rng.uniform(0.18, 0.35),
                _jitter(rng, "#8f8c85", 0.05),
                width=rng.uniform(0.6, 1.2),
            )
        for _ in range(int(rng.integers(3, 8))):
            cx, cy = rng.uniform(0.0, 1.0, size=2)
            budget.add_patch(
                ax,
                patches.Circle(
                    (cx, cy),
                    rng.uniform(0.004, 0.012),
                    facecolor=_jitter(rng, "#9a978f", 0.08),
                    edgecolor="none",
                ),
            )
        _speckle(ax, rng, budget, _area(size, 110), "#9a978f", spread=0.2, s=2.0)

    return render_scene(size, "#bcb9b2", draw, seed=seed)


def soil(size: int = IMAGE_SIZE, seed: int | None = None):
    """Granular bare soil with clods."""

    def draw(ax, rng, budget):
        _mottle(ax, rng, budget, _area(size, 400), "#8a6a43", spread=0.12, s=280, alpha=0.12)
        _mottle(ax, rng, budget, _area(size, 700), "#6b4f30", spread=0.10, s=340, alpha=0.10)
        _speckle(ax, rng, budget, _area(size, 60), "#6b4f30", spread=0.3, s=5.0)
        _speckle(ax, rng, budget, _area(size, 200), "#9a7a52", spread=0.2, s=3.0)
        # small irregular clods and a few dry cracks
        for _ in range(_area(size, 220)):
            if budget.full:
                break
            cx, cy = rng.uniform(0.0, 1.0, size=2)
            radius = rng.uniform(0.004, 0.010)
            budget.add_patch(
                ax,
                patches.Polygon(
                    _blob(rng, cx, cy, radius, n_vertices=int(rng.integers(5, 8))),
                    closed=True,
                    facecolor=_jitter(rng, "#8a6a43", 0.18),
                    edgecolor="none",
                    alpha=0.6,
                ),
            )
        _cracks(ax, rng, budget, int(rng.integers(2, 6)), "#5f452a", 0.12, width=0.5)

    return render_scene(size, "#7a5c3a", draw, seed=seed)


def sand(size: int = IMAGE_SIZE, seed: int | None = None):
    """Fine desert sand with coherent wind ripples."""

    def draw(ax, rng, budget):
        def ripple_family(theta, spacing, amplitude, wavelength, color, alpha, width):
            direction = np.array([np.cos(theta), np.sin(theta)])
            normal = np.array([-np.sin(theta), np.cos(theta)])
            corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
            offsets = corners @ normal
            t = np.linspace(-1.2, 1.8, 60)
            base = np.outer(t, direction)
            phase = rng.uniform(0.0, 2 * np.pi)
            for offset in np.arange(offsets.min() + spacing, offsets.max(), spacing):
                wobble = amplitude * np.sin(t / wavelength * 2 * np.pi + phase + offset * 6)
                points = base + np.outer(wobble, normal) + offset * normal
                budget.add_line(
                    ax,
                    points[:, 0],
                    points[:, 1],
                    color=color,
                    linewidth=width,
                    alpha=alpha,
                )

        theta = rng.uniform(0.0, np.pi)
        ripple_family(
            theta, rng.uniform(0.05, 0.09), rng.uniform(0.008, 0.02),
            rng.uniform(0.15, 0.25), _jitter(rng, "#c9a866", 0.06), 1.0, rng.uniform(0.8, 1.4),
        )
        # fainter cross-ripples give a wind-sculpted, non-repetitive look
        ripple_family(
            theta + np.deg2rad(rng.uniform(65, 115)), rng.uniform(0.12, 0.2),
            rng.uniform(0.004, 0.01), rng.uniform(0.2, 0.35),
            _jitter(rng, "#b89a5c", 0.06), 0.35, 0.7,
        )
        _speckle(ax, rng, budget, _area(size, 90), "#c9a866", spread=0.10, s=2.0)
        _speckle(ax, rng, budget, _area(size, 150), "#b8995a", spread=0.12, s=1.8)

    return render_scene(size, "#d8b978", draw, seed=seed)


def water(size: int = IMAGE_SIZE, seed: int | None = None):
    """Deep water with soft tonal variation and gentle wavy ripples."""

    def draw(ax, rng, budget):
        _mottle(ax, rng, budget, _area(size, 400), "#356089", spread=0.06, s=320, alpha=0.1)
        _ripples(ax, rng, budget, _area(size, 2500), "#7fa8cc", 0.08, amplitude=0.008, width=0.6)

    return render_scene(size, "#3f6f9c", draw, seed=seed)


def crop(size: int = IMAGE_SIZE, seed: int | None = None):
    """A cultivated crop surface: grainy, with soft plough bands (no hard rows)."""

    def draw(ax, rng, budget):
        _mottle(ax, rng, budget, _area(size, 350), "#a8b24f", spread=0.12, s=300, alpha=0.16)
        _mottle(ax, rng, budget, _area(size, 600), "#7f9a3f", spread=0.10, s=320, alpha=0.08)
        for _ in range(int(rng.integers(8, 14))):
            y = rng.uniform(0.0, 1.0)
            x0 = rng.uniform(0.0, 0.4)
            length = rng.uniform(0.5, 1.0)
            budget.add_line(
                ax,
                [x0, x0 + length],
                [y, y + rng.uniform(-0.02, 0.02)],
                color=_jitter(rng, "#7f9a3f", 0.12),
                linewidth=rng.uniform(1.5, 4.0),
                alpha=0.25,
            )
        _speckle(ax, rng, budget, _area(size, 80), "#9aa84f", spread=0.25, s=3.0)
        _speckle(ax, rng, budget, _area(size, 120), "#b7c266", spread=0.18, s=4.0, marker="|")

    return render_scene(size, "#8faa4a", draw, seed=seed)


def gravel(size: int = IMAGE_SIZE, seed: int | None = None):
    """Loose gravel: dense small rounded stones on a darker base."""

    def draw(ax, rng, budget):
        _speckle(ax, rng, budget, _area(size, 80), "#6b6259", spread=0.2, s=4.0)
        colors = ["#9a9186", "#8a8177", "#a8a096", "#7d746a", "#b0a89c"]
        n_stones = max(120, int(size * size / 120))
        for _ in range(n_stones):
            if budget.full:
                break
            cx, cy = rng.uniform(0.0, 1.0, size=2)
            budget.add_patch(
                ax,
                patches.Circle(
                    (cx, cy),
                    rng.uniform(0.006, 0.016),
                    facecolor=_jitter(rng, colors[int(rng.integers(len(colors)))], 0.08),
                    edgecolor=_jitter(rng, "#5f574e", 0.1),
                    linewidth=0.3,
                ),
            )

    return render_scene(size, "#7d746a", draw, seed=seed)


def mud(size: int = IMAGE_SIZE, seed: int | None = None):
    """Wet mud: dark clods, cracks and faint puddle sheen."""

    def draw(ax, rng, budget):
        _mottle(ax, rng, budget, _area(size, 350), "#4a3826", spread=0.10, s=320, alpha=0.18)
        _cover(ax, rng, budget, _area(size, 900), "#5f4a30", 0.12, 0.006, 0.018)
        _cracks(ax, rng, budget, _area(size, 1200), "#2f2418", 0.1, width=0.5)
        for _ in range(int(rng.integers(4, 9))):
            cx, cy = rng.uniform(0.0, 1.0, size=2)
            width = rng.uniform(0.02, 0.05)
            budget.add_patch(
                ax,
                patches.Ellipse(
                    (cx, cy),
                    width * 2 * rng.uniform(0.8, 1.4),
                    width * 2,
                    facecolor=_jitter(rng, "#7a6a52", 0.04),
                    edgecolor="none",
                    alpha=0.4,
                ),
            )
        _speckle(ax, rng, budget, _area(size, 150), "#3a2c1c", spread=0.12, s=2.5)

    return render_scene(size, "#554430", draw, seed=seed)


def rock(size: int = IMAGE_SIZE, seed: int | None = None):
    """Rough rock face: overlapping broken slabs with subtle lit facets."""

    def draw(ax, rng, budget):
        _mottle(ax, rng, budget, _area(size, 240), "#6b5f4d", spread=0.12, s=320, alpha=0.18)
        base_colors = ["#8a7d68", "#7d7160", "#948671", "#6f6353", "#877a66", "#9b8d78"]
        n_slabs = max(220, int(size * size / 900))
        for _ in range(n_slabs):
            if budget.full:
                break
            cx, cy = rng.uniform(-0.05, 1.05, size=2)
            radius = rng.uniform(0.035, 0.11)
            slab = _blob(rng, cx, cy, radius, n_vertices=int(rng.integers(6, 10)))
            color = base_colors[int(rng.integers(len(base_colors)))]
            budget.add_patch(
                ax,
                patches.Polygon(
                    slab,
                    closed=True,
                    facecolor=_jitter(rng, color, 0.06),
                    edgecolor=_jitter(rng, "#574c3d", 0.08),
                    linewidth=0.5,
                ),
            )
            budget.add_patch(
                ax,
                patches.Polygon(
                    _blob(rng, cx - 0.18 * radius, cy + 0.18 * radius, radius * 0.45, n_vertices=6),
                    closed=True,
                    facecolor=_jitter(rng, "#8f8170", 0.05),
                    edgecolor="none",
                ),
            )
            budget.add_patch(
                ax,
                patches.Polygon(
                    _blob(rng, cx + 0.22 * radius, cy - 0.22 * radius, radius * 0.45, n_vertices=6),
                    closed=True,
                    facecolor=_jitter(rng, "#665a49", 0.06),
                    edgecolor="none",
                ),
            )
        _speckle(ax, rng, budget, _area(size, 90), "#6b5d49", spread=0.2, s=2.5)
        _speckle(ax, rng, budget, _area(size, 140), "#a99b86", spread=0.15, s=1.8)

    return render_scene(size, "#7d7160", draw, seed=seed)


def snow(size: int = IMAGE_SIZE, seed: int | None = None):
    """Snow field with faint blue shading and cracks."""

    def draw(ax, rng, budget):
        _mottle(ax, rng, budget, _area(size, 500), "#dbe6ef", spread=0.03, s=300, alpha=0.12)
        _mottle(ax, rng, budget, _area(size, 420), "#b9cbdb", spread=0.05, s=420, alpha=0.10)
        _lines(ax, rng, budget, _area(size, 2200), "#c3d5e2", 0.06, width=0.5)
        _speckle(ax, rng, budget, _area(size, 200), "#ffffff", spread=0.02, s=2.0)

    return render_scene(size, "#eef2f6", draw, seed=seed)


def bark(size: int = IMAGE_SIZE, seed: int | None = None):
    """Vertical bark streaks for trunks."""

    def draw(ax, rng, budget):
        _lines(ax, rng, budget, _area(size, 500), "#3f2f1c", 0.2, vertical=True, width=1.2)
        _cover(ax, rng, budget, _area(size, 2500), "#6b5231", 0.15, 0.004, 0.012)

    return render_scene(size, "#5a4326", draw, seed=seed)


def roof(size: int = IMAGE_SIZE, seed: int | None = None):
    """Roof covering: panels, seams or shingles, varying with the seed."""

    def draw(ax, rng, budget):
        style = int(rng.integers(0, 3))
        angle = rng.uniform(0.0, 180.0)
        seam = _jitter(rng, "#6f6f6f", 0.08)
        if style == 0:
            spacing = rng.uniform(0.08, 0.14)
            _cross_lines(ax, rng, budget, angle, spacing, seam, width=0.8)
            _cross_lines(ax, rng, budget, angle + 90, spacing, seam, width=0.8)
        elif style == 1:
            spacing = rng.uniform(0.05, 0.1)
            _cross_lines(ax, rng, budget, angle, spacing, seam, width=0.7)
        else:
            spacing = rng.uniform(0.05, 0.09)
            for row in range(int(1.0 / spacing)):
                y = row * spacing
                x_shift = (row % 2) * spacing / 2
                for x in np.arange(-spacing, 1.0, spacing):
                    if budget.full:
                        break
                    budget.add_patch(
                        ax,
                        patches.Rectangle(
                            (x + x_shift, y),
                            spacing * 0.9,
                            spacing * 0.7,
                            facecolor=_jitter(rng, "#8a8a8a", 0.12),
                            edgecolor=_jitter(rng, seam, 0.05),
                            linewidth=0.3,
                        ),
                    )
        _speckle(ax, rng, budget, _area(size, 300), "#767676", spread=0.2, s=2.0)

    return render_scene(size, "#8a8a8a", draw, seed=seed)


MATERIALS = {
    "grass": grass,
    "foliage": foliage,
    "asphalt": asphalt,
    "concrete": concrete,
    "soil": soil,
    "sand": sand,
    "water": water,
    "crop": crop,
    "gravel": gravel,
    "mud": mud,
    "rock": rock,
    "snow": snow,
    "bark": bark,
    "roof": roof,
}

MATERIAL_DESCRIPTIONS = {
    "grass": "Grainy green grass with tonal patches.",
    "foliage": "Clumpy leaf canopy with highlights and shadowed gaps.",
    "asphalt": "Grainy asphalt with aggregate speckle and a few cracks.",
    "concrete": "Light mottled concrete / pavement.",
    "soil": "Granular bare soil with clods.",
    "sand": "Fine desert sand with faint wind ripples.",
    "water": "Deep water with soft tonal variation and light streaks.",
    "crop": "A cultivated crop surface: parallel rows plus grain.",
    "gravel": "Loose gravel: dense small rounded stones.",
    "mud": "Wet mud with clods, cracks and faint puddle sheen.",
    "rock": "Rough rock face with angular fragments and grit.",
    "snow": "Snow field with faint blue shading and cracks.",
    "bark": "Vertical bark streaks for trunks.",
    "roof": "Grey roof covering: panel seams plus grit.",
}


def generate_material(name: str, size: int = IMAGE_SIZE, seed: int | None = None):
    """Render a named material texture from :data:`MATERIALS`."""
    try:
        generator = MATERIALS[name]
    except KeyError:
        known = ", ".join(sorted(MATERIALS))
        raise KeyError(f"unknown material {name!r}; known materials: {known}") from None
    return generator(size=size, seed=seed)


def material_textures(size: int, seed: int | None, names):
    """Render several material textures at once, sharing one seed offset.

    Small textures (used by the coverage grids) are cached across scenes: the
    material functions are pure, so the same ``(name, size, seed)`` can be
    reused by several scene types in one notebook run.
    """
    seed = 0 if seed is None else seed
    if size <= 256:
        return {name: _material_cached(name, size, seed + i) for i, name in enumerate(names)}
    return {name: MATERIALS[name](size=size, seed=seed + i) for i, name in enumerate(names)}


@lru_cache(maxsize=512)
def _material_cached(name: str, size: int, seed: int):
    return MATERIALS[name](size=size, seed=seed)