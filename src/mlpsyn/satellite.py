"""Satellite-image texture generators.

Each function fills a square image with a plausible land-cover texture built
only from matplotlib drawing primitives. Textures are grouped by environment:

- vegetation: :func:`dense_forest`, :func:`sparse_woodland`, :func:`grassland`
- agriculture: :func:`agricultural_fields`, :func:`crop_rows`
- built-up: :func:`urban_buildings`, :func:`road_network`
- water: :func:`water_river`
- arid: :func:`desert`
- cryosphere: :func:`snow_ice`

All generators share the signature ``f(size=IMAGE_SIZE, seed=None)`` and return
an RGB ``uint8`` array of shape ``(size, size, 3)``. Shape counts scale with the
image area and always respect the ``MAX_SHAPES`` hard cap.
"""

from __future__ import annotations

import numpy as np
from matplotlib import patches

from mlpsyn.materials import MATERIALS, fill_patch
from mlpsyn.textures import IMAGE_SIZE, _blob, _jitter, render_scene

__all__ = [
    "TEXTURES",
    "TEXTURE_DESCRIPTIONS",
    "agricultural_fields",
    "crop_rows",
    "dense_forest",
    "desert",
    "generate_satellite_texture",
    "grassland",
    "road_network",
    "snow_ice",
    "sparse_woodland",
    "urban_buildings",
    "water_river",
]


def _area_count(size: int, area_per_shape: float, minimum: int = 1) -> int:
    """Number of shapes so that density is roughly constant across sizes."""
    return max(minimum, int(round(size * size / area_per_shape)))


def _parallel_lines(ax, budget, rect, angle_deg, spacing, color, linewidth=0.8, clip=None):
    """Draw parallel lines at ``angle_deg`` across ``rect=(x, y, w, h)``."""
    x, y, w, h = rect
    theta = np.deg2rad(angle_deg)
    direction = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-np.sin(theta), np.cos(theta)])
    corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
    offsets = corners @ normal
    for offset in np.arange(offsets.min() + spacing, offsets.max(), spacing):
        if budget.full:
            break
        center = offset * normal
        p0, p1 = center - 2 * direction, center + 2 * direction
        line = budget.add_line(
            ax,
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color=color,
            linewidth=linewidth,
            solid_capstyle="butt",
        )
        if clip is not None:
            line.set_clip_path(clip)


def _split_rects(rng, rect, n_splits):
    """Recursively split a rectangle into axis-aligned parcels."""
    rects = [rect]
    for _ in range(n_splits):
        index = max(range(len(rects)), key=lambda i: rects[i][2] * rects[i][3])
        x, y, w, h = rects.pop(index)
        frac = rng.uniform(0.35, 0.65)
        if w >= h:
            rects.append((x, y, w * frac, h))
            rects.append((x + w * frac, y, w * (1 - frac), h))
        else:
            rects.append((x, y, w, h * frac))
            rects.append((x, y + h * frac, w, h * (1 - frac)))
    return rects


def dense_forest(size: int = IMAGE_SIZE, seed: int | None = None):
    """Canopy texture: dense overlapping tree crowns with soft shadows."""
    background = MATERIALS["foliage"](size, seed)

    def draw(ax, rng, budget):
        for _ in range(_area_count(size, 220)):
            if budget.full:
                break
            cx, cy = rng.uniform(0.0, 1.0, size=2)
            radius = rng.uniform(0.006, 0.018)
            budget.add_patch(
                ax,
                patches.Ellipse(
                    (cx + 0.004, cy - 0.004),
                    2.2 * radius,
                    1.8 * radius,
                    facecolor=_jitter(rng, "#0b1c0d", 0.03),
                    edgecolor="none",
                ),
            )
            budget.add_patch(
                ax,
                patches.Circle(
                    (cx, cy), radius, facecolor=_jitter(rng, "#2f6b2c", 0.12), edgecolor="none"
                ),
            )
            if not budget.full and rng.random() < 0.35:
                budget.add_patch(
                    ax,
                    patches.Circle(
                        (cx + radius * 0.3, cy + radius * 0.3),
                        radius * 0.55,
                        facecolor=_jitter(rng, "#4f8f3f", 0.1),
                        edgecolor="none",
                    ),
                )

    return render_scene(size, background, draw, seed=seed)


def sparse_woodland(size: int = IMAGE_SIZE, seed: int | None = None):
    """Dry savanna: scattered trees and shrubs over bare ground."""
    background = MATERIALS["sand"](size, seed)

    def draw(ax, rng, budget):
        for _ in range(_area_count(size, 1400)):
            if budget.full:
                break
            cx, cy = rng.uniform(0.0, 1.0, size=2)
            radius = rng.uniform(0.01, 0.03)
            budget.add_patch(
                ax,
                patches.Ellipse(
                    (cx + 0.006, cy - 0.006),
                    1.5 * radius,
                    1.2 * radius,
                    facecolor=(0.15, 0.15, 0.08, 0.30),
                    edgecolor="none",
                ),
            )
            budget.add_patch(
                ax,
                patches.Polygon(
                    _blob(rng, cx, cy, radius),
                    closed=True,
                    facecolor=_jitter(rng, "#4a5d23", 0.12),
                    edgecolor="none",
                ),
            )
        for _ in range(_area_count(size, 900)):
            if budget.full:
                break
            cx, cy = rng.uniform(0.0, 1.0, size=2)
            radius = rng.uniform(0.004, 0.012)
            budget.add_patch(
                ax,
                patches.Circle(
                    (cx, cy), radius, facecolor=_jitter(rng, "#6d7a35", 0.15), edgecolor="none"
                ),
            )
        for _ in range(_area_count(size, 3500)):
            if budget.full:
                break
            cx, cy = rng.uniform(0.0, 1.0, size=2)
            budget.add_patch(
                ax,
                patches.Ellipse(
                    (cx, cy),
                    rng.uniform(0.02, 0.06),
                    rng.uniform(0.008, 0.025),
                    angle=rng.uniform(0, 180),
                    facecolor=_jitter(rng, "#b09a63", 0.05),
                    edgecolor="none",
                ),
            )

    return render_scene(size, background, draw, seed=seed)


def grassland(size: int = IMAGE_SIZE, seed: int | None = None):
    """Meadow: fine grass blades over mottled green patches."""
    background = MATERIALS["grass"](size, seed)

    def draw(ax, rng, budget):
        for _ in range(_area_count(size, 1100)):
            if budget.full:
                break
            cx, cy = rng.uniform(0.0, 1.0, size=2)
            budget.add_patch(
                ax,
                patches.Ellipse(
                    (cx, cy),
                    rng.uniform(0.03, 0.12),
                    rng.uniform(0.02, 0.08),
                    angle=rng.uniform(0, 180),
                    facecolor=_jitter(rng, "#87a24e", 0.1),
                    edgecolor="none",
                ),
            )
        for _ in range(_area_count(size, 260)):
            if budget.full:
                break
            x, y = rng.uniform(0.0, 1.0, size=2)
            length = rng.uniform(0.005, 0.02)
            angle = rng.uniform(np.pi * 0.3, np.pi * 0.7)
            budget.add_line(
                ax,
                [x, x + length * np.cos(angle)],
                [y, y + length * np.sin(angle)],
                color=_jitter(rng, "#5f7d33", 0.15),
                linewidth=rng.uniform(0.3, 0.8),
            )
        for _ in range(_area_count(size, 2200)):
            if budget.full:
                break
            x, y = rng.uniform(0.0, 1.0, size=2)
            budget.add_patch(
                ax,
                patches.Circle(
                    (x, y),
                    rng.uniform(0.0015, 0.004),
                    facecolor=_jitter(rng, "#e8e2a0", 0.1),
                    edgecolor="none",
                ),
            )

    return render_scene(size, background, draw, seed=seed)


_CROP_COLORS = [
    "#6f8f3f",
    "#88a84e",
    "#a8b24f",
    "#c2b45a",
    "#8a9a3a",
    "#b58a4b",
    "#7a9b4a",
    "#c9c07a",
]


def agricultural_fields(size: int = IMAGE_SIZE, seed: int | None = None):
    """Parcel mosaic: irregular fields in different crop colours and row directions."""
    background = MATERIALS["soil"](size, seed)
    textures = {name: MATERIALS[name](size, seed) for name in ("crop", "grass", "soil")}

    def draw(ax, rng, budget):
        n_splits = int(np.clip(_area_count(size, 30000), 4, 40))
        for x, y, w, h in _split_rects(rng, (0.0, 0.0, 1.0, 1.0), n_splits):
            if budget.full:
                break
            key = str(rng.choice(["crop", "crop", "grass", "soil"]))
            color = _jitter(rng, _CROP_COLORS[rng.integers(len(_CROP_COLORS))], 0.05)
            patch = patches.Rectangle(
                (x, y), w, h, facecolor=color, edgecolor="#5f4a2e", linewidth=0.6
            )
            fill_patch(ax, patch, textures[key], budget=budget, zorder=2)
            if budget.full or rng.random() > 0.75:
                continue
            spacing = rng.uniform(0.006, 0.014)
            row_color = _jitter(rng, color, 0.35)
            if rng.random() < 0.5:
                for yy in np.arange(y + spacing, y + h, spacing):
                    if budget.full:
                        break
                    line = budget.add_line(
                        ax, [x, x + w], [yy, yy], color=row_color, linewidth=0.5, zorder=3
                    )
                    line.set_clip_path(patch)
            else:
                for xx in np.arange(x + spacing, x + w, spacing):
                    if budget.full:
                        break
                    line = budget.add_line(
                        ax, [xx, xx], [y, y + h], color=row_color, linewidth=0.5, zorder=3
                    )
                    line.set_clip_path(patch)

    return render_scene(size, background, draw, seed=seed)


def crop_rows(size: int = IMAGE_SIZE, seed: int | None = None):
    """Vineyard / orchard: blocks of parallel crop rows with individual plants."""
    background = MATERIALS["soil"](size, seed)

    def draw(ax, rng, budget):
        for x, y, w, h in _split_rects(rng, (0.0, 0.0, 1.0, 1.0), 2):
            block = budget.add_patch(
                ax,
                patches.Rectangle(
                    (x, y), w, h, facecolor=_jitter(rng, "#9c7b4f", 0.06), edgecolor="none"
                ),
            )
            angle = float(rng.choice([0.0, 90.0, 30.0, -30.0]))
            spacing = rng.uniform(0.012, 0.02)
            row_color = _jitter(rng, "#4f6b2a", 0.1)
            _parallel_lines(ax, budget, (x, y, w, h), angle, spacing, row_color, clip=block)

            theta = np.deg2rad(angle)
            direction = np.array([np.cos(theta), np.sin(theta)])
            normal = np.array([-np.sin(theta), np.cos(theta)])
            corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
            offsets = corners @ normal
            for offset in np.arange(offsets.min() + spacing / 2, offsets.max(), spacing):
                if budget.full:
                    break
                for t in np.arange(-1.0, 1.0, 0.04):
                    if budget.full or rng.random() > 0.5:
                        continue
                    point = offset * normal + t * direction
                    if not (x <= point[0] <= x + w and y <= point[1] <= y + h):
                        continue
                    budget.add_patch(
                        ax,
                        patches.Circle(
                            (point[0], point[1]),
                            rng.uniform(0.003, 0.006),
                            facecolor=_jitter(rng, "#3f6b25", 0.12),
                            edgecolor="none",
                        ),
                    )

    return render_scene(size, background, draw, seed=seed)


def urban_buildings(size: int = IMAGE_SIZE, seed: int | None = None):
    """Built-up area: a street grid filled with building footprints."""
    background = MATERIALS["concrete"](size, seed)
    roof_colors = ["#9a9a9a", "#b0a89a", "#a86f5a", "#7f8fa0", "#c0b8a8", "#8a8a8a"]

    def draw(ax, rng, budget):
        n = int(np.clip(round(size / 50), 6, 14))
        step = 1.0 / n
        for i in range(n + 1):
            budget.add_line(ax, [i * step, i * step], [0, 1], color="#4f4f4f", linewidth=1.5)
            budget.add_line(ax, [0, 1], [i * step, i * step], color="#4f4f4f", linewidth=1.5)
        for ix in range(n):
            for iy in range(n):
                if budget.full:
                    break
                x0, y0 = ix * step, iy * step
                inner = step * rng.uniform(0.62, 0.82)
                ox = x0 + rng.uniform(0.05, 0.12) * step
                oy = y0 + rng.uniform(0.05, 0.12) * step
                if rng.random() < 0.12:
                    budget.add_patch(
                        ax,
                        patches.Rectangle(
                            (ox, oy),
                            inner,
                            inner,
                            facecolor=_jitter(rng, "#4f7a3a", 0.1),
                            edgecolor="none",
                        ),
                    )
                    continue
                n_sub = int(rng.integers(1, 4))
                for k in range(n_sub):
                    if budget.full:
                        break
                    bw = inner / n_sub * rng.uniform(0.7, 0.95)
                    bh = inner * rng.uniform(0.6, 0.95)
                    bx = ox + k * inner / n_sub
                    roof = roof_colors[rng.integers(len(roof_colors))]
                    budget.add_patch(
                        ax,
                        patches.Rectangle(
                            (bx, oy),
                            bw,
                            bh,
                            facecolor=_jitter(rng, roof, 0.06),
                            edgecolor="#5a5a5a",
                            linewidth=0.4,
                        ),
                    )

    return render_scene(size, background, draw, seed=seed)


def road_network(size: int = IMAGE_SIZE, seed: int | None = None):
    """Rural/urban road network over mixed terrain."""
    background = MATERIALS["grass"](size, seed)

    def draw(ax, rng, budget):
        for _ in range(_area_count(size, 12000)):
            if budget.full:
                break
            x, y = rng.uniform(0.0, 1.0, size=2)
            budget.add_patch(
                ax,
                patches.Ellipse(
                    (x, y),
                    rng.uniform(0.05, 0.2),
                    rng.uniform(0.03, 0.12),
                    angle=rng.uniform(0, 180),
                    facecolor=_jitter(rng, "#8a9a5a", 0.08),
                    edgecolor="none",
                ),
            )
        for _ in range(int(rng.integers(2, 5))):
            if budget.full:
                break
            n_pts = 6
            xs = np.linspace(-0.05, 1.05, n_pts)
            ys = np.linspace(rng.uniform(0, 1), rng.uniform(0, 1), n_pts)
            ys = np.convolve(ys, np.ones(3) / 3, mode="same")
            budget.add_line(
                ax,
                xs,
                ys,
                color="#3f3f3f",
                linewidth=rng.uniform(3.0, 6.0),
                solid_capstyle="round",
            )
        n = int(np.clip(round(size / 90), 4, 9))
        step = 1.0 / n
        for i in range(1, n):
            jitter = rng.uniform(-0.01, 0.01)
            budget.add_line(
                ax, [0, 1], [i * step + jitter, i * step + jitter], color="#5a5a5a", linewidth=1.4
            )
            budget.add_line(
                ax, [i * step + jitter, i * step + jitter], [0, 1], color="#5a5a5a", linewidth=1.4
            )
        for _ in range(int(rng.integers(1, 4))):
            if budget.full:
                break
            x, y = rng.uniform(0.1, 0.9, size=2)
            budget.add_patch(
                ax,
                patches.Circle(
                    (x, y),
                    rng.uniform(0.008, 0.015),
                    facecolor="#4f4f4f",
                    edgecolor="#d8d8d8",
                    linewidth=0.8,
                ),
            )

    return render_scene(size, background, draw, seed=seed)


def water_river(size: int = IMAGE_SIZE, seed: int | None = None):
    """Floodplain: a meandering river with sandy banks, a lake and tributaries."""
    background = MATERIALS["grass"](size, seed)

    def draw(ax, rng, budget):
        xs = np.linspace(-0.05, 1.05, 60)
        base = rng.uniform(0.35, 0.65)
        ys = base + 0.22 * np.sin(
            xs * 2 * np.pi * rng.uniform(0.5, 1.2) + rng.uniform(0, 2 * np.pi)
        )
        ys = np.convolve(ys, np.ones(5) / 5, mode="same")
        half_width = 0.03 + 0.02 * np.abs(np.sin(xs * 2 * np.pi * rng.uniform(0.5, 1.0)))

        def ribbon(width_scale):
            upper = np.column_stack([xs, ys + half_width * width_scale])
            lower = np.column_stack([xs, ys - half_width * width_scale])[::-1]
            return np.vstack([upper, lower])

        budget.add_patch(
            ax,
            patches.Polygon(
                ribbon(1.8), closed=True, facecolor=_jitter(rng, "#c8b27a", 0.04), edgecolor="none"
            ),
        )
        budget.add_patch(
            ax,
            patches.Polygon(
                ribbon(1.0), closed=True, facecolor=_jitter(rng, "#3f6f9c", 0.05), edgecolor="none"
            ),
        )
        for _ in range(int(rng.integers(1, 3))):
            if budget.full:
                break
            cx, cy = rng.uniform(0.15, 0.85, size=2)
            budget.add_patch(
                ax,
                patches.Ellipse(
                    (cx, cy),
                    rng.uniform(0.08, 0.18),
                    rng.uniform(0.05, 0.12),
                    angle=rng.uniform(0, 180),
                    facecolor=_jitter(rng, "#4a7aa8", 0.05),
                    edgecolor="none",
                ),
            )
        for _ in range(int(rng.integers(2, 5))):
            if budget.full:
                break
            x0 = rng.choice([0.0, 1.0])
            y0 = rng.uniform(0.0, 1.0)
            x1, y1 = rng.uniform(0.2, 0.8, size=2)
            budget.add_line(
                ax,
                [x0, x1],
                [y0, y1],
                color=_jitter(rng, "#4a7aa8", 0.05),
                linewidth=rng.uniform(1.5, 3.5),
                solid_capstyle="round",
            )

    return render_scene(size, background, draw, seed=seed)


def desert(size: int = IMAGE_SIZE, seed: int | None = None):
    """Arid ground: wind ripples over sand with scattered rocks."""
    background = MATERIALS["sand"](size, seed)

    def draw(ax, rng, budget):
        for _ in range(_area_count(size, 280)):
            if budget.full:
                break
            x, y = rng.uniform(0.0, 1.0, size=2)
            length = rng.uniform(0.02, 0.08)
            angle = rng.uniform(0.0, np.pi)
            x1 = x + length * np.cos(angle)
            y1 = y + length * np.sin(angle)
            budget.add_line(
                ax,
                [x, (x + x1) / 2, x1],
                [y, (y + y1) / 2 + rng.uniform(-0.004, 0.004), y1],
                color=_jitter(rng, "#c9a866", 0.05),
                linewidth=rng.uniform(0.4, 0.9),
            )
        for _ in range(_area_count(size, 1400)):
            if budget.full:
                break
            x, y = rng.uniform(0.0, 1.0, size=2)
            budget.add_patch(
                ax,
                patches.Polygon(
                    _blob(rng, x, y, rng.uniform(0.004, 0.012), n_vertices=6),
                    closed=True,
                    facecolor=_jitter(rng, "#8a6b45", 0.1),
                    edgecolor="none",
                ),
            )
        for _ in range(_area_count(size, 6000)):
            if budget.full:
                break
            x, y = rng.uniform(0.0, 1.0, size=2)
            budget.add_patch(
                ax,
                patches.Circle(
                    (x, y),
                    rng.uniform(0.002, 0.005),
                    facecolor=_jitter(rng, "#7f7a4a", 0.12),
                    edgecolor="none",
                ),
            )

    return render_scene(size, background, draw, seed=seed)


def snow_ice(size: int = IMAGE_SIZE, seed: int | None = None):
    """Cryosphere: snow field with ice cracks and exposed rock."""
    background = MATERIALS["snow"](size, seed)

    def draw(ax, rng, budget):
        for _ in range(_area_count(size, 1100)):
            if budget.full:
                break
            x, y = rng.uniform(0.0, 1.0, size=2)
            budget.add_patch(
                ax,
                patches.Ellipse(
                    (x, y),
                    rng.uniform(0.04, 0.14),
                    rng.uniform(0.02, 0.08),
                    angle=rng.uniform(0, 180),
                    facecolor=_jitter(rng, "#dbe6ef", 0.03),
                    edgecolor="none",
                ),
            )
        for _ in range(_area_count(size, 450)):
            if budget.full:
                break
            x, y = rng.uniform(0.0, 1.0, size=2)
            length = rng.uniform(0.03, 0.12)
            angle = rng.uniform(0.0, np.pi)
            x1 = x + length * np.cos(angle)
            y1 = y + length * np.sin(angle)
            budget.add_line(
                ax,
                [x, x1],
                [y, y1],
                color=_jitter(rng, "#b9cfdd", 0.04),
                linewidth=rng.uniform(0.3, 0.8),
            )
        for _ in range(_area_count(size, 1800)):
            if budget.full:
                break
            x, y = rng.uniform(0.0, 1.0, size=2)
            budget.add_patch(
                ax,
                patches.Polygon(
                    _blob(rng, x, y, rng.uniform(0.005, 0.02), n_vertices=7),
                    closed=True,
                    facecolor=_jitter(rng, "#8a8578", 0.12),
                    edgecolor="none",
                ),
            )

    return render_scene(size, background, draw, seed=seed)


TEXTURES = {
    "dense_forest": dense_forest,
    "sparse_woodland": sparse_woodland,
    "grassland": grassland,
    "agricultural_fields": agricultural_fields,
    "crop_rows": crop_rows,
    "urban_buildings": urban_buildings,
    "road_network": road_network,
    "water_river": water_river,
    "desert": desert,
    "snow_ice": snow_ice,
}

TEXTURE_DESCRIPTIONS = {
    "dense_forest": "Dense canopy: overlapping tree crowns with soft shadows.",
    "sparse_woodland": "Dry savanna: scattered trees and shrubs over bare ground.",
    "grassland": "Meadow: fine grass blades over mottled green patches.",
    "agricultural_fields": "Parcel mosaic: irregular fields with crop rows.",
    "crop_rows": "Vineyard/orchard: parallel crop rows with individual plants.",
    "urban_buildings": "Built-up area: a street grid filled with building footprints.",
    "road_network": "Roads over mixed terrain, with intersections and roundabouts.",
    "water_river": "Meandering river with sandy banks, a lake and tributaries.",
    "desert": "Arid ground: wind ripples over sand with scattered rocks.",
    "snow_ice": "Snow field with ice cracks and exposed rock.",
}


def generate_satellite_texture(
    name: str, size: int = IMAGE_SIZE, seed: int | None = None
):
    """Render a named satellite texture from :data:`TEXTURES`."""
    try:
        generator = TEXTURES[name]
    except KeyError:
        known = ", ".join(sorted(TEXTURES))
        raise KeyError(f"unknown texture {name!r}; known textures: {known}") from None
    return generator(size=size, seed=seed)
