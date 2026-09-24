"""Object generators: textured geometric shapes assembled into semantic objects.

Objects sit above the material layer. Their geometry (roof, canopy, water, crop
surface, road, ...) is filled with a *material texture* from
:mod:`mlpsyn.materials`, so objects inherit realistic grain instead of flat
colours.

Each point object has the signature::

    draw_x(ax, rng, budget, cx, cy, size, materials=None, rotation=None, **kwargs)
    -> (x0, y0, x1, y1)

The returned bounding box is normalised and axis-aligned. When ``rotation`` is
left to ``None`` the object picks a random orientation, so scenes have no
axis-aligned prior. Point generators are collected in :data:`OBJECTS`; linear
features use :func:`draw_road`.
"""

from __future__ import annotations

import numpy as np
from matplotlib import patches
from matplotlib.colors import to_rgb
from matplotlib.transforms import TransformedPatchPath

from mlpsyn._mode import mask_mode
from mlpsyn.materials import fill_patch
from mlpsyn.textures import _jitter

__all__ = [
    "OBJECTS",
    "draw_boat",
    "draw_building",
    "draw_bush",
    "draw_car",
    "draw_conifer",
    "draw_field_plot",
    "draw_field_polygon",
    "draw_greenhouse",
    "draw_hedgerow",
    "draw_object",
    "draw_orchard",
    "draw_parking_lot",
    "draw_plaza",
    "draw_pond",
    "draw_pool",
    "draw_river",
    "draw_road",
    "draw_rock",
    "draw_silo",
    "draw_tree",
    "draw_truck",
]

_SHADOW = (0.05, 0.08, 0.05, 0.42)
_ROOF_COLORS = [
    "#9a9a9a",
    "#b0a89a",
    "#a86f5a",
    "#7f8fa0",
    "#c0b8a8",
    "#8a8a8a",
    "#6f7d6a",
    "#b58a6a",
    "#7a7f88",
    "#a0947a",
    "#8fa08a",
    "#9c8f9c",
]
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


def _smooth_blob(rng, cx, cy, radius, n_vertices=18, roughness=0.18):
    """An organic, non-spiky closed polygon around ``(cx, cy)``."""
    angles = np.linspace(0.0, 2 * np.pi, n_vertices, endpoint=False)
    radii = radius * (1.0 - roughness * rng.random(n_vertices))
    return np.column_stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)])


def _rotate(points, cx, cy, angle_deg):
    """Rotate 2D ``points`` by ``angle_deg`` degrees around ``(cx, cy)``."""
    theta = np.deg2rad(angle_deg)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.asarray(points) @ rotation.T + np.array([cx, cy])


def _clip_box(x0, y0, x1, y1):
    """Clip a normalised box to the unit square and return it as a tuple."""
    return (
        float(np.clip(x0, 0.0, 1.0)),
        float(np.clip(y0, 0.0, 1.0)),
        float(np.clip(x1, 0.0, 1.0)),
        float(np.clip(y1, 0.0, 1.0)),
    )


def _bbox(cx, cy, size, aspect=1.0):
    """Axis-aligned normalised box of a centred shape."""
    half_w = size / 2
    half_h = size * aspect / 2
    return _clip_box(cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _rotated_bbox(cx, cy, width, height, angle_deg):
    """Axis-aligned bounding box of a rotated rectangle."""
    corners = [
        (-width / 2, -height / 2),
        (width / 2, -height / 2),
        (width / 2, height / 2),
        (-width / 2, height / 2),
    ]
    rotated = _rotate(corners, cx, cy, angle_deg)
    return _clip_box(
        rotated[:, 0].min(), rotated[:, 1].min(), rotated[:, 0].max(), rotated[:, 1].max()
    )


def _random_rotation(rng):
    """A random orientation in degrees."""
    return float(rng.uniform(0.0, 360.0))


def _fill(ax, patch, materials, key, fallback, budget, zorder=2, rotation=0.0, center=None):
    """Fill a patch with a material texture, falling back to a flat colour."""
    if materials and key in materials:
        return fill_patch(
            ax,
            patch,
            materials[key],
            budget=budget,
            zorder=zorder,
            rotation=rotation,
            center=center,
        )
    patch.set_facecolor(fallback)
    budget.add_patch(ax, patch)
    return patch


def draw_tree(ax, rng, budget, cx, cy, size, palette="#2f6b2c", materials=None, rotation=None):
    """Broadleaf tree seen top-down: a foliage-filled crown, no trunk."""
    half = size / 2
    budget.add_patch(
        ax,
        patches.Ellipse(
            (cx + 0.18 * half, cy - 0.22 * half),
            size * 1.15,
            size * 0.8,
            facecolor=_SHADOW,
            edgecolor="none",
        ),
    )
    center = (cx, cy + 0.15 * half)
    canopy = _smooth_blob(rng, center[0], center[1], size * 0.52, roughness=0.24)
    canopy_patch = patches.Polygon(canopy, closed=True, edgecolor="none")
    _fill(
        ax,
        canopy_patch,
        materials,
        "foliage",
        _jitter(rng, palette, 0.16),
        budget,
        zorder=3,
    )
    budget.add_patch(
        ax,
        patches.Circle(
            center,
            size * 0.1,
            facecolor=_jitter(rng, "#1f3d1a", 0.12),
            edgecolor="none",
            zorder=6,
        ),
    )
    for _ in range(int(rng.integers(3, 6))):
        ox, oy = rng.normal(0.0, size * 0.22, size=2)
        budget.add_patch(
            ax,
            patches.Polygon(
                _smooth_blob(
                    rng, cx + ox, cy + 0.15 * half + oy, size * rng.uniform(0.08, 0.15),
                    roughness=0.28,
                ),
                closed=True,
                facecolor=_jitter(rng, palette, 0.16),
                edgecolor="none",
                zorder=4,
            ),
        )
    for _ in range(int(rng.integers(2, 4))):
        ox, oy = rng.normal(0.0, size * 0.16, size=2)
        budget.add_patch(
            ax,
            patches.Polygon(
                _smooth_blob(
                    rng, cx + ox, cy + 0.1 * half + oy, size * rng.uniform(0.05, 0.09),
                    roughness=0.3,
                ),
                closed=True,
                facecolor=_jitter(rng, "#6fae45", 0.12),
                edgecolor="none",
                zorder=5,
            ),
        )
    return _bbox(cx, cy, size)


def draw_conifer(
    ax, rng, budget, cx, cy, size, palette="#1f4d24", materials=None, rotation=None
):
    """Conifer seen top-down: overlapping jagged foliage rings (fir crown)."""
    radius = size / 2
    budget.add_patch(
        ax,
        patches.Ellipse(
            (cx + 0.12 * radius, cy - 0.12 * radius),
            size * 1.0,
            size * 0.92,
            facecolor=_SHADOW,
            edgecolor="none",
        ),
    )
    base = patches.Polygon(
        _smooth_blob(rng, cx, cy, radius * 0.82, roughness=0.16), closed=True, edgecolor="none"
    )
    _fill(ax, base, materials, "foliage", _jitter(rng, palette, 0.10), budget, zorder=3)
    # Soft concentric crowns (lighter rim -> darker heart), not a spiky star.
    for layer, scale in enumerate([0.66, 0.42]):
        blob = patches.Polygon(
            _smooth_blob(rng, cx, cy, radius * scale, roughness=0.12),
            closed=True,
            edgecolor="none",
        )
        _fill(
            ax,
            blob,
            materials,
            "foliage",
            _jitter(rng, palette, 0.10 + 0.06 * (layer + 1)),
            budget,
            zorder=4 + layer,
        )
    # Radial needle streaks give the fir its branch texture without a star outline.
    n_needles = int(rng.integers(10, 15))
    ang0 = rng.uniform(0.0, 2 * np.pi)
    needle_segments = []
    for k in range(n_needles):
        angle = ang0 + k * 2 * np.pi / n_needles + rng.uniform(-0.08, 0.08)
        inner = radius * 0.2
        outer = radius * rng.uniform(0.72, 0.9)
        needle_segments.append([
            (cx + inner * np.cos(angle), cy + inner * np.sin(angle)),
            (cx + outer * np.cos(angle), cy + outer * np.sin(angle)),
        ])
    budget.add_line_collection(
        ax,
        needle_segments,
        color=_jitter(rng, "#163d1c", 0.18),
        linewidth=0.5,
        alpha=0.5,
        zorder=6,
    )
    budget.add_patch(
        ax,
        patches.Circle(
            (cx, cy),
            radius * 0.14,
            facecolor=_jitter(rng, "#16361a", 0.1),
            edgecolor="none",
            zorder=7,
        ),
    )
    return _bbox(cx, cy, size)


def draw_bush(ax, rng, budget, cx, cy, size, palette="#4a7a33", materials=None, rotation=None):
    """Shrub: a compact foliage-filled cluster with a shadow."""
    budget.add_patch(
        ax,
        patches.Ellipse(
            (cx + 0.12 * size, cy - 0.18 * size),
            size * 1.0,
            size * 0.55,
            facecolor=_SHADOW,
            edgecolor="none",
        ),
    )
    main = _smooth_blob(rng, cx, cy, size * 0.4, roughness=0.22)
    main_patch = patches.Polygon(main, closed=True, edgecolor="none")
    _fill(ax, main_patch, materials, "foliage", _jitter(rng, palette, 0.18), budget, zorder=3)
    for _ in range(int(rng.integers(3, 6))):
        angle = rng.uniform(0.0, 2 * np.pi)
        distance = size * rng.uniform(0.05, 0.14)
        ox, oy = distance * np.cos(angle), distance * np.sin(angle)
        budget.add_patch(
            ax,
            patches.Polygon(
                _smooth_blob(rng, cx + ox, cy + oy, size * rng.uniform(0.08, 0.14)),
                closed=True,
                facecolor=_jitter(rng, palette, 0.18),
                edgecolor="none",
                zorder=4,
            ),
        )
    for _ in range(int(rng.integers(2, 4))):
        ox, oy = rng.normal(0.0, size * 0.18, size=2)
        budget.add_patch(
            ax,
            patches.Polygon(
                _smooth_blob(rng, cx + ox, cy + oy, size * rng.uniform(0.04, 0.08)),
                closed=True,
                facecolor=_jitter(rng, "#6fae45", 0.12),
                edgecolor="none",
                zorder=5,
            ),
        )
    return _bbox(cx, cy, size)


def draw_building(ax, rng, budget, cx, cy, size, materials=None, rotation=None):
    """Building footprint with a material-filled, textured roof."""
    if rotation is None:
        rotation = _random_rotation(rng)
    width = size
    height = size * rng.uniform(0.6, 0.95)
    for scale, alpha in ((1.07, 0.18), (1.0, 0.30)):
        offset = 0.05 * scale
        budget.add_patch(
            ax,
            patches.Rectangle(
                (
                    cx - width * scale / 2 + offset * width,
                    cy - height * scale / 2 - offset * height,
                ),
                width * scale,
                height * scale,
                angle=rotation,
                rotation_point="center",
                facecolor=(0.08, 0.08, 0.08, alpha),
                edgecolor="none",
            ),
        )

    roof = _jitter(rng, _ROOF_COLORS[rng.integers(len(_ROOF_COLORS))], 0.05)
    footprint = patches.Rectangle(
        (cx - width / 2, cy - height / 2), width, height, angle=rotation, rotation_point="center"
    )
    _fill(
        ax,
        footprint,
        materials,
        "roof",
        roof,
        budget,
        zorder=3,
        rotation=rotation,
        center=(cx, cy),
    )
    budget.add_patch(
        ax,
        patches.Rectangle(
            (cx - width / 2, cy - height / 2),
            width,
            height,
            angle=rotation,
            rotation_point="center",
            facecolor="none",
            edgecolor="#3a3a3a",
            linewidth=0.5,
            zorder=4,
        ),
    )
    for _ in range(int(rng.integers(1, 4))):
        uw = width * rng.uniform(0.1, 0.22)
        uh = height * rng.uniform(0.1, 0.22)
        ux, uy = rng.uniform(-0.35, 0.35, size=2) * np.array([width, height])
        unit_center = _rotate([(ux, uy)], cx, cy, rotation)[0]
        budget.add_patch(
            ax,
            patches.Rectangle(
                (unit_center[0] - uw / 2, unit_center[1] - uh / 2),
                uw,
                uh,
                angle=rotation,
                rotation_point="center",
                facecolor=_jitter(rng, "#6f6f6f", 0.1),
                edgecolor="#4a4a4a",
                linewidth=0.3,
                zorder=5,
            ),
        )
    return _rotated_bbox(cx, cy, width, height, rotation)


def draw_field_plot(
    ax,
    rng,
    budget,
    cx,
    cy,
    size,
    color=None,
    aspect=None,
    materials=None,
    rotation=None,
    material_key="crop",
):
    """Agricultural plot: a material-filled rectangle with oriented crop rows."""
    if rotation is None:
        rotation = _random_rotation(rng)
    width = size
    height = size * (rng.uniform(0.55, 0.95) if aspect is None else aspect)
    if color is None:
        color = _CROP_COLORS[rng.integers(len(_CROP_COLORS))]
    color = _jitter(rng, color, 0.05)
    plot = patches.Rectangle(
        (cx - width / 2, cy - height / 2), width, height, angle=rotation, rotation_point="center"
    )
    _fill(
        ax,
        plot,
        materials,
        material_key,
        color,
        budget,
        zorder=3,
        rotation=rotation,
        center=(cx, cy),
    )

    spacing = rng.uniform(0.012, 0.02)
    rgb = np.asarray(to_rgb(color), dtype=float) * 0.55
    row_color = _jitter(rng, tuple(np.clip(rgb, 0.0, 1.0)), 0.2)
    horizontal = rng.random() < 0.5
    if horizontal:
        local_y = np.arange(-height / 2 + spacing, height / 2, spacing)
        segments = [
            _rotate([(-width / 2, y), (width / 2, y)], cx, cy, rotation) for y in local_y
        ]
    else:
        local_x = np.arange(-width / 2 + spacing, width / 2, spacing)
        segments = [
            _rotate([(x, -height / 2), (x, height / 2)], cx, cy, rotation) for x in local_x
        ]
    for segment in segments:
        if budget.full:
            break
        line = budget.add_line(
            ax,
            [segment[0][0], segment[1][0]],
            [segment[0][1], segment[1][1]],
            color=row_color,
            linewidth=0.7,
            zorder=4,
        )
        line.set_clip_path(TransformedPatchPath(plot))
    budget.add_patch(
        ax,
        patches.Rectangle(
            (cx - width / 2, cy - height / 2),
            width,
            height,
            angle=rotation,
            rotation_point="center",
            facecolor="none",
            edgecolor="#5f4a2e",
            linewidth=0.7,
            zorder=5,
        ),
    )
    return _rotated_bbox(cx, cy, width, height, rotation)


def draw_field_polygon(
    ax,
    rng,
    budget,
    polygon,
    materials=None,
    material_key="crop",
    color=None,
    row_angle=None,
    zorder=3,
    draw_rows=None,
):
    """An agricultural field as an arbitrary polygon with oriented crop rows.

    Unlike :func:`draw_field_plot`, the footprint is any convex/concave polygon,
    so fields can abut roads, rivers or woodland along natural boundaries.
    """
    polygon = np.asarray(polygon, dtype=float)
    if color is None:
        color = _CROP_COLORS[rng.integers(len(_CROP_COLORS))]
    color = _jitter(rng, color, 0.05)
    patch = patches.Polygon(polygon, closed=True, edgecolor="none")
    _fill(ax, patch, materials, material_key, color, budget, zorder=zorder)

    x0, y0 = float(polygon[:, 0].min()), float(polygon[:, 1].min())
    x1, y1 = float(polygon[:, 0].max()), float(polygon[:, 1].max())
    draw_rows = material_key in ("crop", "soil") if draw_rows is None else draw_rows
    if mask_mode() or not draw_rows:
        return _clip_box(x0, y0, x1, y1)
    rgb = np.asarray(to_rgb(color), dtype=float) * 0.55
    row_color = _jitter(rng, tuple(np.clip(rgb, 0.0, 1.0)), 0.2)
    theta = np.deg2rad(rng.uniform(0.0, 180.0) if row_angle is None else row_angle)
    direction = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-np.sin(theta), np.cos(theta)])
    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    span = float(np.hypot(x1 - x0, y1 - y0)) + 0.1
    spacing = rng.uniform(0.02, 0.035)
    max_rows = 45
    if 2 * span / spacing > max_rows:
        spacing = 2 * span / max_rows
    for offset in np.arange(-span, span, spacing):
        if budget.full:
            break
        base = center + offset * normal
        a = base - direction * span
        b = base + direction * span
        line = budget.add_line(
            ax, [a[0], b[0]], [a[1], b[1]], color=row_color, linewidth=0.7, zorder=zorder + 1
        )
        line.set_clip_path(TransformedPatchPath(patch))
    return _clip_box(x0, y0, x1, y1)


def draw_pond(ax, rng, budget, cx, cy, size, materials=None, rotation=None):
    """Water body: irregular material-filled polygon with shoreline and sheen."""
    radius = size / 2
    water = _smooth_blob(rng, cx, cy, radius)
    center = np.array([cx, cy])
    shoreline = center + (water - center) * 1.12
    budget.add_patch(
        ax,
        patches.Polygon(
            shoreline, closed=True, facecolor=_jitter(rng, "#c8b27a", 0.05), edgecolor="none"
        ),
    )
    water_patch = patches.Polygon(water, closed=True, edgecolor="none")
    _fill(ax, water_patch, materials, "water", _jitter(rng, "#3f6f9c", 0.05), budget, zorder=3)
    for _ in range(int(rng.integers(2, 4))):
        ox = rng.uniform(-0.3, 0.1) * radius
        oy = rng.uniform(0.1, 0.35) * radius
        budget.add_patch(
            ax,
            patches.Ellipse(
                (cx + ox, cy + oy),
                radius * rng.uniform(0.4, 0.7),
                radius * rng.uniform(0.12, 0.22),
                facecolor=_jitter(rng, "#8fb8d8", 0.06),
                edgecolor="none",
                alpha=0.35,
                zorder=4,
            ),
        )
    return _bbox(cx, cy, size)


def draw_rock(ax, rng, budget, cx, cy, size, materials=None, rotation=None):
    """Boulder seen top-down: an irregular rounded silhouette with lit facets."""
    radius = size / 2
    budget.add_patch(
        ax,
        patches.Ellipse(
            (cx + 0.16 * radius, cy - 0.16 * radius),
            size * 1.05,
            size * 0.88,
            facecolor=_SHADOW,
            edgecolor="none",
        ),
    )
    shape = _smooth_blob(rng, cx, cy, radius * 0.95, roughness=0.22)
    rock_patch = patches.Polygon(shape, closed=True, edgecolor="none")
    _fill(ax, rock_patch, materials, "rock", _jitter(rng, "#8a7d68", 0.06), budget, zorder=3)
    budget.add_patch(
        ax,
        patches.Polygon(
            _smooth_blob(
                rng, cx - 0.22 * radius, cy + 0.22 * radius, radius * 0.55, roughness=0.2
            ),
            closed=True,
            facecolor=_jitter(rng, "#a99b83", 0.05),
            edgecolor="none",
            zorder=4,
        ),
    )
    budget.add_patch(
        ax,
        patches.Polygon(
            _smooth_blob(
                rng, cx + 0.26 * radius, cy - 0.26 * radius, radius * 0.45, roughness=0.2
            ),
            closed=True,
            facecolor=_jitter(rng, "#5f5445", 0.06),
            edgecolor="none",
            zorder=4,
        ),
    )
    budget.add_patch(
        ax,
        patches.Polygon(
            shape,
            closed=True,
            facecolor="none",
            edgecolor=_jitter(rng, "#5f5445", 0.08),
            linewidth=0.8,
            zorder=5,
        ),
    )
    return _bbox(cx, cy, size)


def draw_car(ax, rng, budget, cx, cy, size, materials=None, rotation=None):
    """Vehicle seen top-down: body, wheels, cabin and windshield."""
    if rotation is None:
        rotation = _random_rotation(rng)
    width = size
    height = size * 0.45
    body = _jitter(rng, _ROOF_COLORS[rng.integers(len(_ROOF_COLORS))], 0.08)

    def rect(local_x, local_y, w, h, color, zorder, edge="#2f2f2f", lw=0.4):
        center = _rotate([(local_x, local_y)], cx, cy, rotation)[0]
        budget.add_patch(
            ax,
            patches.Rectangle(
                (center[0] - w / 2, center[1] - h / 2),
                w,
                h,
                angle=rotation,
                rotation_point="center",
                facecolor=color,
                edgecolor=edge,
                linewidth=lw,
                zorder=zorder,
            ),
        )

    rect(0.05 * width, -0.1 * height, width, height, (0.08, 0.08, 0.08, 0.3), 2, edge="none")
    for dx in (-0.32, 0.32):
        for dy in (-0.5, 0.5):
            rect(dx * width, dy * height, 0.16 * width, 0.34 * height, "#222222", 2)
    rect(0.0, 0.0, width, height, body, 3)
    rect(0.0, 0.12 * height, width * 0.42, height * 0.5, _jitter(rng, body, 0.1), 4)
    window = _rotate(
        [(-0.42 * width * 0.35, 0.12 * height), (0.42 * width * 0.35, 0.12 * height)],
        cx,
        cy,
        rotation,
    )
    budget.add_line(
        ax,
        [window[0][0], window[1][0]],
        [window[0][1], window[1][1]],
        color="#9fd0e8",
        linewidth=0.8,
        zorder=5,
    )
    return _rotated_bbox(cx, cy, width, height * 1.4, rotation)


def draw_boat(ax, rng, budget, cx, cy, size, materials=None, rotation=None):
    """Small boat seen top-down: a pointed hull, deck and a faint wake."""
    if rotation is None:
        rotation = _random_rotation(rng)
    width = size
    height = size * 0.32
    budget.add_patch(
        ax,
        patches.Ellipse(
            (cx, cy), width * 1.7, height * 2.0,
            facecolor=(0.75, 0.85, 0.92, 0.18), edgecolor="none", zorder=2,
        ),
    )
    hull_color = _jitter(rng, ["#e8e4dc", "#c85a4a", "#3f6f9c", "#d8c27a"][rng.integers(4)], 0.06)
    hull = patches.Polygon(
        _rotate(
            [
                (-0.5 * width, -0.5 * height),
                (0.35 * width, -0.5 * height),
                (0.5 * width, 0.0),
                (0.35 * width, 0.5 * height),
                (-0.5 * width, 0.5 * height),
            ],
            cx,
            cy,
            rotation,
        ),
        closed=True,
        edgecolor="#4a4a4a",
        linewidth=0.5,
        zorder=3,
    )
    hull.set_facecolor(hull_color)
    budget.add_patch(ax, hull)
    center = _rotate([(-0.05 * width, 0.0)], cx, cy, rotation)[0]
    budget.add_patch(
        ax,
        patches.Rectangle(
            (center[0] - width * 0.16, center[1] - height * 0.32),
            width * 0.32,
            height * 0.64,
            angle=rotation,
            rotation_point="center",
            facecolor=_jitter(rng, "#f0ece4", 0.05),
            edgecolor="#5f5f5f",
            linewidth=0.4,
            zorder=4,
        ),
    )
    return _rotated_bbox(cx, cy, width, height, rotation)


def draw_silo(ax, rng, budget, cx, cy, size, materials=None, rotation=None):
    """Farm silo seen top-down: a ribbed cylinder with a domed roof."""
    radius = size / 2
    budget.add_patch(
        ax,
        patches.Ellipse(
            (cx + 0.12 * radius, cy - 0.12 * radius),
            size * 1.02,
            size * 0.9,
            facecolor=_SHADOW,
            edgecolor="none",
        ),
    )
    body = patches.Circle((cx, cy), radius * 0.95)
    _fill(ax, body, materials, "concrete", _jitter(rng, "#c8c2b6", 0.05), budget, zorder=3)
    n_ribs = int(rng.integers(14, 20))
    offset = rng.uniform(0.0, 2 * np.pi)
    for i in range(n_ribs):
        if budget.full:
            break
        angle = offset + i * 2 * np.pi / n_ribs
        budget.add_line(
            ax,
            [cx, cx + radius * 0.9 * np.cos(angle)],
            [cy, cy + radius * 0.9 * np.sin(angle)],
            color=_jitter(rng, "#9a948a", 0.06),
            linewidth=0.4,
            zorder=4,
        )
    budget.add_patch(
        ax,
        patches.Polygon(
            _smooth_blob(rng, cx + 0.3 * radius, cy - 0.3 * radius, radius * 0.5, roughness=0.2),
            closed=True,
            facecolor=(0.15, 0.15, 0.15, 0.12),
            edgecolor="none",
            zorder=4,
        ),
    )
    dome = patches.Circle((cx, cy), radius * 0.42)
    _fill(ax, dome, materials, "roof", _jitter(rng, "#9a9a9a", 0.05), budget, zorder=5)
    budget.add_patch(
        ax,
        patches.Circle(
            (cx, cy), radius * 0.42, facecolor="none", edgecolor="#5f5f5f",
            linewidth=0.6, zorder=6,
        ),
    )
    return _bbox(cx, cy, size)


def draw_pool(ax, rng, budget, cx, cy, size, materials=None, rotation=None):
    """Swimming pool seen top-down: a tiled deck around a water rectangle."""
    if rotation is None:
        rotation = _random_rotation(rng)
    width = size
    height = size * rng.uniform(0.5, 0.7)
    deck = patches.Rectangle(
        (cx - width / 2, cy - height / 2), width, height, angle=rotation, rotation_point="center"
    )
    _fill(
        ax, deck, materials, "concrete", _jitter(rng, "#d8d2c4", 0.05), budget,
        zorder=3, rotation=rotation, center=(cx, cy),
    )
    inset = 0.82
    water = patches.Rectangle(
        (cx - width * inset / 2, cy - height * inset / 2),
        width * inset,
        height * inset,
        angle=rotation,
        rotation_point="center",
    )
    _fill(
        ax, water, materials, "water", _jitter(rng, "#3f8fbf", 0.05), budget,
        zorder=4, rotation=rotation, center=(cx, cy),
    )
    budget.add_patch(
        ax,
        patches.Rectangle(
            (cx - width * inset / 2, cy - height * inset / 2),
            width * inset,
            height * inset,
            angle=rotation,
            rotation_point="center",
            facecolor="none",
            edgecolor="#5f5548",
            linewidth=0.6,
            zorder=5,
        ),
    )
    return _rotated_bbox(cx, cy, width, height, rotation)


def draw_parking_lot(
    ax, rng, budget, cx, cy, size, materials=None, rotation=None, car_callback=None
):
    """Parking lot seen top-down: an asphalt pad with stalls and parked cars.

    When ``car_callback`` is given it is called with the box of every parked car,
    so the caller can add them as separate annotations.
    """
    if rotation is None:
        rotation = _random_rotation(rng)
    width = size
    height = size * rng.uniform(0.5, 0.72)
    pad = patches.Rectangle(
        (cx - width / 2, cy - height / 2), width, height, angle=rotation, rotation_point="center"
    )
    _fill(
        ax, pad, materials, "asphalt", _jitter(rng, "#4a4a4a", 0.05), budget,
        zorder=3, rotation=rotation, center=(cx, cy),
    )
    car_width = size * rng.uniform(0.24, 0.30)
    stall_width = max(car_width * 1.35, width / 4.0)
    n_stalls = max(2, int(width / stall_width))
    for k in range(1, n_stalls):
        local_x = -width / 2 + k * width / n_stalls
        segment = _rotate([(local_x, -height / 2), (local_x, height / 2)], cx, cy, rotation)
        budget.add_line(
            ax,
            [segment[0][0], segment[1][0]],
            [segment[0][1], segment[1][1]],
            color="#c9c4a6",
            linewidth=0.5,
            zorder=4,
        )
    for k in range(n_stalls):
        if rng.random() > 0.5:
            continue
        local_x = -width / 2 + (k + 0.5) * width / n_stalls
        side = rng.choice([-0.25, 0.25])
        px, py = _rotate([(local_x, side * height)], cx, cy, rotation)[0]
        car_bbox = draw_car(
            ax,
            rng,
            budget,
            float(px),
            float(py),
            car_width,
            materials=materials,
            rotation=rotation + 90.0,
        )
        if car_callback is not None:
            car_callback(car_bbox)
    budget.add_patch(
        ax,
        patches.Rectangle(
            (cx - width / 2, cy - height / 2),
            width,
            height,
            angle=rotation,
            rotation_point="center",
            facecolor="none",
            edgecolor="#3a3a3a",
            linewidth=0.6,
            zorder=5,
        ),
    )
    return _rotated_bbox(cx, cy, width, height, rotation)


def draw_plaza(ax, rng, budget, cx, cy, size, materials=None, rotation=None):
    """Paved plaza seen top-down with a fountain or a planted centre and trees."""
    if rotation is None:
        rotation = _random_rotation(rng)
    side = size
    slab = patches.Rectangle(
        (cx - side / 2, cy - side / 2), side, side, angle=rotation, rotation_point="center"
    )
    _fill(
        ax, slab, materials, "concrete", _jitter(rng, "#cfc8ba", 0.04), budget,
        zorder=3, rotation=rotation, center=(cx, cy),
    )
    spacing = side * 0.22
    for offset in np.arange(-side / 2 + spacing, side / 2, spacing):
        for local in (
            [(offset, -side / 2), (offset, side / 2)],
            [(-side / 2, offset), (side / 2, offset)],
        ):
            segment = _rotate(local, cx, cy, rotation)
            budget.add_line(
                ax,
                [segment[0][0], segment[1][0]],
                [segment[0][1], segment[1][1]],
                color="#b6ae9c",
                linewidth=0.4,
                zorder=4,
            )
    if rng.random() < 0.5:
        budget.add_patch(
            ax,
            patches.Circle(
                (cx, cy),
                side * 0.12,
                facecolor="#7fb0d0",
                edgecolor="#8a8272",
                linewidth=0.7,
                zorder=4,
            ),
        )
    else:
        bed = patches.Circle((cx, cy), side * 0.16, edgecolor="none")
        if materials and "grass" in materials:
            fill_patch(ax, bed, materials["grass"], budget=budget, zorder=4)
        draw_bush(ax, rng, budget, cx, cy, side * 0.22, materials=materials)
    for dx, dy in ((-0.34, -0.34), (0.34, -0.34), (0.34, 0.34), (-0.34, 0.34)):
        if rng.random() < 0.7:
            px, py = _rotate([(dx * side, dy * side)], cx, cy, rotation)[0]
            draw_tree(ax, rng, budget, float(px), float(py), side * 0.2, materials=materials)
    return _rotated_bbox(cx, cy, side, side, rotation)


def draw_orchard(ax, rng, budget, cx, cy, size, materials=None, rotation=None):
    """Orchard / vineyard: a block of small trees planted in regular rows."""
    if rotation is None:
        rotation = _random_rotation(rng)
    width = size
    height = size * rng.uniform(0.6, 0.9)
    pad = patches.Rectangle(
        (cx - width / 2, cy - height / 2), width, height, angle=rotation, rotation_point="center"
    )
    _fill(
        ax, pad, materials, "soil", _jitter(rng, "#6b5a3a", 0.05), budget,
        zorder=3, rotation=rotation, center=(cx, cy),
    )
    step = size * rng.uniform(0.15, 0.2)
    tree_size = size * rng.uniform(0.09, 0.13)
    for local_y in np.arange(-height / 2 + step * 0.5, height / 2, step):
        for local_x in np.arange(-width / 2 + step * 0.5, width / 2, step):
            if budget.full:
                break
            px, py = _rotate([(local_x, local_y)], cx, cy, rotation)[0]
            draw_tree(ax, rng, budget, float(px), float(py), tree_size, materials=materials)
    budget.add_patch(
        ax,
        patches.Rectangle(
            (cx - width / 2, cy - height / 2),
            width,
            height,
            angle=rotation,
            rotation_point="center",
            facecolor="none",
            edgecolor="#5f4a2e",
            linewidth=0.6,
            zorder=5,
        ),
    )
    return _rotated_bbox(cx, cy, width, height, rotation)


def draw_greenhouse(ax, rng, budget, cx, cy, size, materials=None, rotation=None):
    """A block of glasshouses seen top-down: parallel gabled glazed ranges."""
    if rotation is None:
        rotation = _random_rotation(rng)
    width = size
    height = size * rng.uniform(0.55, 0.8)
    n_ranges = max(3, int(width / (size * 0.2)))
    range_width = width / n_ranges
    glass = _jitter(rng, "#cfe0da", 0.05)
    # A soft drop shadow under the whole glasshouse block.
    shadow = _rotate([(0.05 * width, -0.08 * height)], cx, cy, rotation)[0]
    budget.add_patch(
        ax,
        patches.Rectangle(
            (shadow[0] - width / 2, shadow[1] - height / 2),
            width,
            height,
            angle=rotation,
            rotation_point="center",
            facecolor=_SHADOW,
            edgecolor="none",
            zorder=2,
        ),
    )
    for k in range(n_ranges):
        if budget.full:
            break
        local_x = -width / 2 + (k + 0.5) * range_width
        center = _rotate([(local_x, 0.0)], cx, cy, rotation)[0]
        panel_width = range_width * 0.82
        # Translucent glazed panel.
        budget.add_patch(
            ax,
            patches.Rectangle(
                (center[0] - panel_width / 2, center[1] - height / 2),
                panel_width,
                height,
                angle=rotation,
                rotation_point="center",
                facecolor=_jitter(rng, glass, 0.06),
                edgecolor="#8f9a96",
                linewidth=0.5,
                zorder=3,
            ),
        )
        # Bright ridge line along the apex of the gable.
        ridge = _rotate(
            [(local_x, -height / 2), (local_x, height / 2)], cx, cy, rotation
        )
        budget.add_line(
            ax,
            [ridge[0][0], ridge[1][0]],
            [ridge[0][1], ridge[1][1]],
            color="#ffffff",
            linewidth=0.6,
            alpha=0.6,
            zorder=4,
        )
        # Perpendicular frame ribs (batched as one collection).
        n_ribs = max(2, int(height / (size * 0.12)))
        rib_segments = []
        for r in range(1, n_ribs):
            local_y = -height / 2 + r * height / n_ribs
            segment = _rotate(
                [(local_x - panel_width / 2, local_y), (local_x + panel_width / 2, local_y)],
                cx,
                cy,
                rotation,
            )
            rib_segments.append(
                [(segment[0][0], segment[0][1]), (segment[1][0], segment[1][1])]
            )
        budget.add_line_collection(
            ax, rib_segments, color="#aeb8b4", linewidth=0.4, zorder=4
        )
    return _rotated_bbox(cx, cy, width, height, rotation)


def draw_truck(ax, rng, budget, cx, cy, size, materials=None, rotation=None):
    """Box truck seen top-down: long body, cab and six wheels."""
    if rotation is None:
        rotation = _random_rotation(rng)
    width = size
    height = size * 0.42
    body = _jitter(rng, _ROOF_COLORS[rng.integers(len(_ROOF_COLORS))], 0.08)

    def rect(local_x, local_y, w, h, color, zorder, edge="#2f2f2f", lw=0.4):
        center = _rotate([(local_x, local_y)], cx, cy, rotation)[0]
        budget.add_patch(
            ax,
            patches.Rectangle(
                (center[0] - w / 2, center[1] - h / 2),
                w,
                h,
                angle=rotation,
                rotation_point="center",
                facecolor=color,
                edgecolor=edge,
                linewidth=lw,
                zorder=zorder,
            ),
        )

    rect(0.04 * width, -0.1 * height, width, height, (0.08, 0.08, 0.08, 0.3), 2, edge="none")
    for dx in (-0.34, 0.02, 0.34):
        for dy in (-0.5, 0.5):
            rect(dx * width, dy * height, 0.13 * width, 0.3 * height, "#222222", 2)
    rect(0.0, 0.0, width, height, body, 3)
    rect(0.31 * width, 0.0, width * 0.28, height, _jitter(rng, body, 0.14), 4)
    window = _rotate(
        [(0.2 * width, 0.0), (0.42 * width, 0.0)], cx, cy, rotation
    )
    budget.add_line(
        ax,
        [window[0][0], window[1][0]],
        [window[0][1], window[1][1]],
        color="#9fd0e8",
        linewidth=0.8,
        zorder=5,
    )
    return _rotated_bbox(cx, cy, width, height * 1.4, rotation)


def draw_hedgerow(ax, rng, budget, cx, cy, size, materials=None, rotation=None):
    """A line of small bushes forming a field hedgerow (top-down)."""
    if rotation is None:
        rotation = _random_rotation(rng)
    n = int(rng.integers(5, 9))
    angle = np.deg2rad(rotation)
    for i in range(n):
        offset = ((i + 0.5) / n - 0.5) * size
        px = cx + offset * np.cos(angle)
        py = cy + offset * np.sin(angle)
        draw_bush(ax, rng, budget, px, py, size * 0.18, materials=materials)
    return _rotated_bbox(cx, cy, size, size * 0.3, rotation)


def _ribbon(points, width):
    """Turn a polyline into a closed ribbon polygon of the given width."""
    points = np.asarray(points, dtype=float)
    tangents = np.gradient(points, axis=0)
    lengths = np.hypot(tangents[:, 0], tangents[:, 1])
    lengths[lengths == 0] = 1.0
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]]) / lengths[:, None]
    upper = points + normals * width / 2
    lower = points - normals * width / 2
    return np.vstack([upper, lower[::-1]])


def _polyline_bbox(points, width):
    """Normalised bounding box of a polyline widened by ``width``."""
    ribbon = _ribbon(points, width)
    return _clip_box(
        ribbon[:, 0].min(), ribbon[:, 1].min(), ribbon[:, 0].max(), ribbon[:, 1].max()
    )


def _centerline(rng, p0, p1, n=14, curve=0.0, meander=0.0, points=None):
    """A centreline for a linear feature: straight, bowed or meandering."""
    if points is not None:
        return np.asarray(points, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    direction = p1 - p0
    length = float(np.hypot(*direction))
    if length == 0:
        return np.asarray([p0], dtype=float)
    unit = direction / length
    normal = np.array([-unit[1], unit[0]])
    t = np.linspace(0.0, 1.0, n)
    center = p0 + np.outer(t, direction)
    center = center + np.outer(curve * np.sin(t * np.pi), normal)
    if meander:
        for k in (1, 2, 3):
            phase = rng.uniform(0.0, 2 * np.pi)
            amplitude = meander / k**1.4
            taper = np.sin(np.pi * t)
            center = center + np.outer(
                amplitude * np.sin(2 * np.pi * k * t + phase) * taper, normal
            )
    return center


def draw_road(
    ax,
    rng,
    budget,
    p0=None,
    p1=None,
    width=0.02,
    materials=None,
    color="#3f3f3f",
    curve=0.0,
    points=None,
    meander=0.0,
    n=14,
    shoulder=True,
    dashes=True,
    material_key="asphalt",
):
    """Road ribbon, straight or curved, with an optional dashed centreline.

    A straight road uses ``p0``/``p1``; pass ``points`` for an explicit
    polyline, ``curve`` for a single bow, or ``meander`` for a wavy country
    road. ``material_key`` lets a dirt track use e.g. the ``gravel`` material.
    Returns the ribbon's normalised bounding box.
    """
    center = _centerline(rng, p0, p1, n=n, curve=curve, meander=meander, points=points)
    if len(center) < 2:
        return None
    if shoulder:
        budget.add_patch(
            ax,
            patches.Polygon(
                _ribbon(center, width * 1.7), closed=True, facecolor="#8a8a8a", edgecolor="none"
            ),
        )
    road_patch = patches.Polygon(_ribbon(center, width), closed=True, edgecolor="none")
    _fill(ax, road_patch, materials, material_key, color, budget, zorder=3)

    if dashes:
        segments = np.diff(center, axis=0)
        seg_lengths = np.hypot(segments[:, 0], segments[:, 1])
        total = float(seg_lengths.sum())
        if total > 0:
            cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
            n_dashes = max(1, int(total / 0.06))
            dash_segments = []
            for i in range(n_dashes):
                if budget.full:
                    break
                distance = (i + 0.5) / n_dashes * total
                idx = int(np.clip(np.searchsorted(cumulative, distance) - 1, 0, len(center) - 2))
                length = seg_lengths[idx]
                frac = (distance - cumulative[idx]) / length if length > 0 else 0.0
                point = center[idx] + frac * segments[idx]
                tangent = segments[idx] / length if length > 0 else np.array([1.0, 0.0])
                a = point - tangent * width * 0.25
                b = point + tangent * width * 0.25
                dash_segments.append([(a[0], a[1]), (b[0], b[1])])
            if dash_segments:
                budget.add_line_collection(
                    ax, dash_segments, colors="#e8e0b0", linewidths=0.5, zorder=4
                )
    return _polyline_bbox(center, width * 1.7)


def draw_river(
    ax,
    rng,
    budget,
    p0=None,
    p1=None,
    width=0.09,
    materials=None,
    meander=0.18,
    points=None,
    n=18,
    color="#3f6f9c",
    bank="#c2b07e",
):
    """Meandering watercourse: a bank ribbon under a material-filled river."""
    center = _centerline(rng, p0, p1, n=n, meander=meander, points=points)
    if len(center) < 2:
        return None
    budget.add_patch(
        ax,
        patches.Polygon(
            _ribbon(center, width * 1.45),
            closed=True,
            facecolor=_jitter(rng, bank, 0.05),
            edgecolor="none",
        ),
    )
    water_patch = patches.Polygon(_ribbon(center, width), closed=True, edgecolor="none")
    _fill(ax, water_patch, materials, "water", _jitter(rng, color, 0.05), budget, zorder=3)
    # Broken, faint highlight streaks (not one hard centre line).
    segments = [
        [tuple(center[i - 1]), tuple(center[i])]
        for i in range(1, len(center), 2)
    ]
    budget.add_line_collection(
        ax, segments, color="#9dc0dc", linewidth=1.0, alpha=0.35, zorder=4,
    )
    return _polyline_bbox(center, width * 1.45)


def draw_object(name, ax, rng, budget, cx, cy, size, **kwargs):
    """Dispatch to a point-object generator from :data:`OBJECTS`."""
    try:
        generator = OBJECTS[name]
    except KeyError:
        known = ", ".join(sorted(OBJECTS))
        raise KeyError(f"unknown object {name!r}; known objects: {known}") from None
    return generator(ax, rng, budget, cx, cy, size, **kwargs)


OBJECTS = {
    "tree": draw_tree,
    "conifer": draw_conifer,
    "bush": draw_bush,
    "building": draw_building,
    "field_plot": draw_field_plot,
    "pond": draw_pond,
    "rock": draw_rock,
    "car": draw_car,
    "truck": draw_truck,
    "silo": draw_silo,
    "pool": draw_pool,
    "parking": draw_parking_lot,
    "plaza": draw_plaza,
    "hedgerow": draw_hedgerow,
    "orchard": draw_orchard,
    "greenhouse": draw_greenhouse,
    "boat": draw_boat,
}