"""Scene generators: a textured background populated with textured objects.

A scene is assembled in two layers:

1. a **material** texture used as ground (see :mod:`mlpsyn.materials`);
2. semantic objects drawn on top (see :mod:`mlpsyn.object`), themselves filled
   with materials.

Three environments are provided: ``urban``, ``forest`` and ``fields``. Layouts
use jittered positions and random orientations so scenes carry no axis-aligned
prior. Objects are placed so their centres stay inside a margin, which keeps
heavily truncated objects out of the frame; bounding boxes are axis-aligned and
clipped to ``[0, 1]``. Each scene returns a :class:`Scene` with the RGB image
and the list of placed objects (normalised bounding boxes), i.e. object-level
annotations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from matplotlib import patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.path import Path

from mlpsyn._mode import MASK_MODE, mask_mode
from mlpsyn.materials import fill_patch, material_textures
from mlpsyn.object import (
    OBJECTS,
    _ribbon,
    draw_building,
    draw_car,
    draw_field_polygon,
    draw_river,
    draw_road,
)
from mlpsyn.textures import _DPI, MAX_SHAPES, ShapeBudget, _blob, _jitter

__all__ = [
    "SCENES",
    "Scene",
    "SceneObject",
    "arid_scene",
    "coastal_scene",
    "fields_scene",
    "forest_fields_scene",
    "mountain_scene",
    "pasture_scene",
    "forest_scene",
    "generate_scene",
    "urban_fringe_scene",
    "coastal_city_scene",
    "urban_scene",
    "wilderness_scene",
]


@dataclass(frozen=True)
class SceneObject:
    """An object placed in a scene, with a normalised ``(x0, y0, x1, y1)`` box."""

    label: str
    bbox: tuple[float, float, float, float]

    def pixel_bbox(self, size: int) -> tuple[int, int, int, int]:
        """Bounding box in pixels with a top-left origin, as used by images."""
        x0, y0, x1, y1 = self.bbox
        return (
            round(x0 * size),
            round((1.0 - y1) * size),
            round(x1 * size),
            round((1.0 - y0) * size),
        )


@dataclass
class Scene:
    """A rendered scene and its object-level annotations."""

    name: str
    seed: int | None
    image: np.ndarray
    objects: list[SceneObject] = field(default_factory=list)

    def labels(self) -> list[str]:
        return [obj.label for obj in self.objects]


def _scatter_positions(rng, n, min_dist, margin=0.06, max_tries=200):
    """Rejection sampling with a minimum separation between points."""
    points = []
    for _ in range(n):
        for _ in range(max_tries):
            candidate = rng.uniform(margin, 1 - margin, size=2)
            if all(np.hypot(*(candidate - p)) >= min_dist for p in points):
                points.append(candidate)
                break
    return points


def _meander_points(rng, p0, p1, n=12, meander=0.04):
    """A gently wavy polyline from ``p0`` to ``p1`` (endpoints preserved)."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    direction = p1 - p0
    length = float(np.hypot(*direction))
    if length == 0:
        return np.asarray([p0], dtype=float)
    unit = direction / length
    normal = np.array([-unit[1], unit[0]])
    t = np.linspace(0.0, 1.0, n)
    points = p0 + np.outer(t, direction)
    for k in (1, 2):
        phase = rng.uniform(0.0, 2 * np.pi)
        amplitude = meander / k
        taper = np.sin(np.pi * t)
        points = points + np.outer(amplitude * np.sin(2 * np.pi * k * t + phase) * taper, normal)
    return points


def _distance_to_polyline(point, points):
    """Minimum distance from ``point`` to a polyline."""
    point = np.asarray(point, dtype=float)
    points = np.asarray(points, dtype=float)
    best = np.inf
    for a, b in zip(points[:-1], points[1:], strict=False):
        segment = b - a
        length_sq = float(segment @ segment)
        if length_sq == 0:
            best = min(best, float(np.hypot(*(point - a))))
            continue
        t = float(np.clip((point - a) @ segment / length_sq, 0.0, 1.0))
        closest = a + t * segment
        best = min(best, float(np.hypot(*(point - closest))))
    return best


def _near_road(point, corridors, clearance):
    """True if ``point`` is within ``clearance`` of any road centreline."""
    return any(
        _distance_to_polyline(point, points) < width / 2 + clearance
        for points, width in corridors
    )


def _road_clearance(point, corridors):
    """Distance from ``point`` to the nearest road edge (large if none)."""
    if not corridors:
        return 1.0
    return min(
        _distance_to_polyline(point, points) - width / 2 for points, width in corridors
    )


def _compose(name, size, seed, background_key, drawer, needed_materials):
    """Render a scene: material ground, then the drawer's objects, then masks."""
    if mask_mode():
        from mlpsyn.segmentation import CLASS_COLORS, MATERIAL_SEG_CLASS

        materials = {
            key: np.full(
                (1, 1, 3),
                CLASS_COLORS.get(MATERIAL_SEG_CLASS.get(key, ""), (0.0, 0.0, 0.0)),
                dtype=np.float32,
            )
            for key in needed_materials
        }
    else:
        # A small, scene-independent seed keeps the material cache warm.
        materials = material_textures(size, (seed or 0) % 8, needed_materials)
    background_image = materials[background_key]
    rng = np.random.default_rng((seed if seed is not None else 0) + 10_000)

    fig = Figure(figsize=(size / _DPI, size / _DPI), dpi=_DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(background_image, extent=(0, 1, 0, 1), zorder=0, interpolation="nearest")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.set_axis_off()
    ax.set_autoscale_on(False)
    # Limits are fixed to [0, 1] and autoscale is off, so skip the per-artist
    # limit bookkeeping that dominates scene-generation time.
    ax._update_patch_limits = lambda *args, **kwargs: None
    ax._update_line_limits = lambda *args, **kwargs: None
    ax._update_collection_limits = lambda *args, **kwargs: None

    budget = ShapeBudget(MAX_SHAPES)
    objects = drawer(ax, rng, budget, materials)

    ax.set_aspect("auto")
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return Scene(name=name, seed=seed, image=rgba[..., :3].copy(), objects=objects)


def _pick_plant(rng, conifer_bias=0.32, bush_prob=0.18):
    """Return ``(label, palette)`` for a random plant, weighted by type."""
    roll = rng.random()
    if roll < 1.0 - conifer_bias - bush_prob:
        return "tree", rng.choice(["#2f6b2c", "#3f7a2f", "#5c8f3a", "#6f9b3f", "#24561f"])
    if roll < 1.0 - bush_prob:
        return "conifer", rng.choice(["#1f4d24", "#2a5a2a", "#163f1c"])
    return "bush", rng.choice(["#4a7a33", "#5f8f3a", "#6f9b45"])


def _place_vegetation(ax, rng, budget, materials, n, size_range, min_dist, margin=0.08):
    """Scatter trees/bushes/conifers and return their scene objects."""
    objects = []
    for cx, cy in _scatter_positions(rng, n, min_dist, margin=margin):
        if budget.full:
            break
        label, palette = _pick_plant(rng)
        size = rng.uniform(*size_range)
        bbox = OBJECTS[label](ax, rng, budget, cx, cy, size, palette=palette, materials=materials)
        objects.append(SceneObject(label, bbox))
    return objects


def _forest_patch(ax, rng, budget, materials, objects, cx, cy, radius, density=7):
    """A clump of plants around ``(cx, cy)``, as found in a woodland stand."""
    for _ in range(int(rng.integers(3, density + 1))):
        if budget.full:
            break
        px = cx + rng.normal(0.0, radius * 0.5)
        py = cy + rng.normal(0.0, radius * 0.5)
        if not (0.02 <= px <= 0.98 and 0.02 <= py <= 0.98):
            continue
        label, palette = _pick_plant(rng, conifer_bias=0.42, bush_prob=0.22)
        size = rng.uniform(0.04, 0.09)
        bbox = OBJECTS[label](ax, rng, budget, px, py, size, palette=palette, materials=materials)
        objects.append(SceneObject(label, bbox))


def _elevation_field(rng, n=96, octaves=3):
    """A smooth random height field: summed random sinusoids plus a few hills."""
    yy, xx = np.mgrid[0 : 1 : n * 1j, 0 : 1 : n * 1j]
    field = np.zeros_like(xx)
    amp = 1.0
    for _ in range(octaves):
        angle = rng.uniform(0.0, np.pi)
        freq = rng.uniform(1.0, 3.0)
        phase = rng.uniform(0.0, 2 * np.pi)
        field += amp * np.sin(2 * np.pi * freq * (np.cos(angle) * xx + np.sin(angle) * yy) + phase)
        amp *= 0.5
    for _ in range(int(rng.integers(2, 5))):
        cx, cy = rng.uniform(0.1, 0.9, size=2)
        r = rng.uniform(0.15, 0.4)
        field += rng.uniform(0.5, 1.5) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
    return field


def _terrain(rng, n=96, relief=5.0, strength=0.30):
    """A shared elevation map for a scene.

    Returns ``(height, slope, hillshade)``: ``height`` normalised to [0, 1],
    ``slope`` the normalised gradient magnitude, and ``hillshade`` an RGBA
    relief overlay lit from the upper-left (consistent with the drop shadows
    used by the objects, so light direction is coherent across a scene).
    """
    z = _elevation_field(rng, n)
    z = (z - z.min()) / (np.ptp(z) + 1e-6)
    gy, gx = np.gradient(z * relief)
    slope = np.hypot(gx, gy)
    slope = slope / (slope.max() + 1e-6)
    lx, ly, lz = -0.6, 0.6, 0.8
    norm = np.sqrt(gx * gx + gy * gy + 1.0)
    shade = (-gx * lx - gy * ly + lz) / norm
    value = np.clip(0.5 + strength * (shade - shade.mean()) / (shade.std() + 1e-6), 0.0, 1.0)
    # Smooth grayscale relief (lit upper-left), faded in mostly on the slopes so
    # flat ground keeps its material and ridges/valleys read clearly.
    rgba = np.zeros((n, n, 4))
    rgba[..., :3] = value[..., None]
    rgba[..., 3] = np.clip(slope * 2.5, 0.0, 1.0) * 0.6
    return z, slope, rgba


def _sample(field, x, y):
    """Nearest-cell sample of a 2D field at normalised ``(x, y)`` in [0, 1]."""
    n = field.shape[0]
    row = int(np.clip(y * (n - 1), 0, n - 1))
    col = int(np.clip(x * (n - 1), 0, n - 1))
    return float(field[row, col])


def _pick_by_elevation(rng, field, predicate, margin=0.05, tries=30):
    """Rejection-sample a position where ``predicate(field(x, y))`` holds."""
    x = y = 0.5
    for _ in range(tries):
        x, y = rng.uniform(margin, 1.0 - margin, size=2)
        if predicate(_sample(field, x, y)):
            break
    return float(x), float(y)


def _low_valley_path(rng, height, n=20, meander=0.15, tries=6):
    """A cross-frame meander that prefers low ground (a stream/valley line)."""
    best = None
    best_score = np.inf
    for _ in range(tries):
        if rng.random() < 0.5:
            a = (-0.1, rng.uniform(0.15, 0.85))
            b = (1.1, rng.uniform(0.15, 0.85))
        else:
            a = (rng.uniform(0.15, 0.85), -0.1)
            b = (rng.uniform(0.15, 0.85), 1.1)
        path = _meander_points(rng, a, b, n=n, meander=meander)
        score = float(np.mean([_sample(height, x, y) for x, y in path]))
        if score < best_score:
            best_score, best = score, path
    return best


def _gentle_path(rng, slope, n=20, meander=0.15, tries=6):
    """A cross-frame track that prefers gentle (low-slope) ground."""
    best = None
    best_score = np.inf
    for _ in range(tries):
        if rng.random() < 0.5:
            a = (-0.1, rng.uniform(0.15, 0.85))
            b = (1.1, rng.uniform(0.15, 0.85))
        else:
            a = (rng.uniform(0.15, 0.85), -0.1)
            b = (rng.uniform(0.15, 0.85), 1.1)
        path = _meander_points(rng, a, b, n=n, meander=meander)
        score = float(np.mean([_sample(slope, x, y) for x, y in path]))
        if score < best_score:
            best_score, best = score, path
    return best


def _relief_overlay(ax, budget, terrain, zorder=1, alpha=1.0):
    """Draw a scene's hillshade overlay, optionally softened by ``alpha``."""
    if mask_mode():
        return
    rgba = terrain[2].copy()
    rgba[..., 3] *= alpha
    budget.add_artist(
        ax.imshow(rgba, extent=(0, 1, 0, 1), zorder=zorder, interpolation="bilinear")
    )


def _organic_edge(ys, split, rng, amplitude=0.05, roughness=0.02):
    """A wavy region boundary with an extra high-frequency, smoothed wobble."""
    phase = rng.uniform(0.0, 2 * np.pi)
    edge = (
        split
        + amplitude * np.sin(2 * np.pi * ys * 1.3 + phase)
        + 0.02 * np.sin(5 * np.pi * ys + phase)
    )
    noise = rng.normal(0.0, roughness, size=len(ys))
    kernel = np.ones(3) / 3.0
    return edge + np.convolve(noise, kernel, mode="same")


def _soft_edge(ax, rng, budget, materials, ys, x_edge, key, side, zorder=2):
    """Fade a region material into its neighbour across an organic boundary.

    Several nested, increasingly wide bands are drawn on the neighbour side of
    the edge with decreasing alpha, so the region colour bleeds out instead of
    ending on a hard line. ``side`` is +1 for a neighbour to the right and -1
    for a neighbour to the left.
    """
    kernel = np.ones(3) / 3.0
    wobble = np.convolve(rng.normal(0.0, 0.35, size=len(ys)), kernel, mode="same")
    for dist, alpha in ((0.02, 0.45), (0.05, 0.25), (0.09, 0.10)):
        x_outer = x_edge + side * dist * (1.0 + wobble)
        band = np.vstack(
            [
                np.column_stack([x_edge, ys]),
                np.column_stack([x_outer[::-1], ys[::-1]]),
            ]
        )
        patch = patches.Polygon(band, closed=True, edgecolor="none")
        if key in materials:
            fill_patch(
                ax,
                patch,
                materials[key],
                budget=budget,
                zorder=zorder,
                alpha=1.0 if mask_mode() else alpha,
            )
        else:
            budget.add_patch(ax, patch)


def _ground_patches(ax, rng, budget, materials, keys, n, size_range=(0.14, 0.34), zorder=1):
    """Soft material patches that break up a flat ground (clearings, variation).

    Each clearing is built from three nested, low-alpha fills so its edge fades
    into the ground instead of showing a hard polygon.
    """
    for _ in range(n):
        if budget.full:
            break
        cx, cy = rng.uniform(0.05, 0.95, size=2)
        radius = rng.uniform(*size_range) / 2
        key = keys[int(rng.integers(len(keys)))]
        for ring in range(3):
            patch = patches.Polygon(
                _blob(
                    rng,
                    cx,
                    cy,
                    radius * (1.0 - 0.28 * ring),
                    n_vertices=int(rng.integers(10, 16)),
                ),
                closed=True,
                edgecolor="none",
            )
            if key in materials:
                power = 1.0 if mask_mode() else 0.18
                fill_patch(ax, patch, materials[key], budget=budget, zorder=zorder, alpha=power)
            else:
                budget.add_patch(ax, patch)


def _clip_halfplane(polygon, point, normal):
    """Keep the part of ``polygon`` where ``(x - point) @ normal <= 0``."""
    out = []
    n = len(polygon)
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        da = float((a - point) @ normal)
        db = float((b - point) @ normal)
        if da <= 0.0:
            out.append(a)
        if (da < 0.0 < db) or (db < 0.0 < da):
            t = da / (da - db)
            out.append(a + t * (b - a))
    return np.asarray(out, dtype=float)


def _voronoi_cells(sites, box=(0.0, 0.0, 1.0, 1.0)):
    """Voronoi cells of ``sites`` clipped to ``box`` (half-plane clipping).

    Each cell is convex and the cells tile the box exactly, so fields built
    from them share natural boundaries instead of floating in grass.
    """
    x0, y0, x1, y1 = box
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    cells = []
    for i, site in enumerate(sites):
        cell = corners.copy()
        for j, other in enumerate(sites):
            if i == j:
                continue
            cell = _clip_halfplane(cell, (site + other) / 2.0, other - site)
            if len(cell) < 3:
                break
        cells.append(cell)
    return cells


def _shared_edges(cells, sites, min_length=0.02):
    """Map ``(i, j) -> (p0, p1)`` for every pair of adjacent Voronoi cells."""
    edges = {}
    for i, poly in enumerate(cells):
        n = len(poly)
        for j in range(i + 1, len(sites)):
            delta = sites[j] - sites[i]
            length = float(np.hypot(*delta))
            if length == 0.0:
                continue
            unit = delta / length
            mid = (sites[i] + sites[j]) / 2.0
            on = [k for k in range(n) if abs(float((poly[k] - mid) @ unit)) < 1e-7]
            best = None
            for a in range(len(on)):
                for b in range(a + 1, len(on)):
                    dist = float(np.hypot(*(poly[on[a]] - poly[on[b]])))
                    if best is None or dist > best[0]:
                        best = (dist, on[a], on[b])
            if best is not None and best[0] >= min_length:
                edges[(i, j)] = (poly[best[1]].copy(), poly[best[2]].copy())
    return edges


def _spanning_tree(n, edges, weight=None):
    """Kruskal spanning forest over cell indices; returns a list of edge keys.

    When ``weight`` is given, edges are processed cheapest-first (a minimum
    spanning tree), so the road network can prefer gentle boundaries.
    """
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ordered = list(edges) if weight is None else sorted(edges, key=weight)
    tree = []
    for (i, j) in ordered:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            tree.append((i, j))
    return tree


def _close_network(tree, edges):
    """Add incident edges to degree-1 cells so no road dead-ends inside the frame.

    A spanning tree necessarily has leaves; a real rural road network does not
    usually stop in the middle of a field. Every leaf is given one extra unused
    edge (when available), raising its degree to at least two.
    """
    tree = set(tree)
    degree: Counter = Counter()
    for i, j in tree:
        degree[i] += 1
        degree[j] += 1
    for key in list(edges):
        if key in tree:
            continue
        i, j = key
        if degree[i] == 1 or degree[j] == 1:
            tree.add(key)
            degree[i] += 1
            degree[j] += 1
    return tree


def _poly_bbox(polygon):
    """Normalised (x0, y0, x1, y1) bounding box of a polygon."""
    polygon = np.asarray(polygon, dtype=float)
    return (
        float(polygon[:, 0].min()),
        float(polygon[:, 1].min()),
        float(polygon[:, 0].max()),
        float(polygon[:, 1].max()),
    )


def _hedge_edge(ax, rng, budget, materials, p0, p1, objects):
    """A line of bushes along a field boundary (a hedgerow)."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    length = float(np.hypot(*(p1 - p0)))
    if length < 0.05:
        return
    n = max(2, int(length / 0.045))
    for t in np.linspace(0.1, 0.9, n):
        if budget.full:
            break
        point = p0 + t * (p1 - p0)
        OBJECTS["bush"](
            ax, rng, budget, float(point[0]), float(point[1]), rng.uniform(0.025, 0.04),
            materials=materials,
        )
    objects.append(SceneObject("hedgerow", _poly_bbox(np.vstack([p0, p1]))))


def _field_mosaic(ax, rng, budget, materials, objects, box, zorder=2, terrain=None):
    """An irregular field mosaic inside ``box`` with roads and hedges.

    Used by the boundary scenes so their agricultural half is not a grid of
    rectangles: cells tile ``box`` and a small road network gives frontage to
    every parcel. When ``terrain`` is given, roads prefer gentle boundaries.
    """
    x0, y0, x1, y1 = box
    width, box_height = x1 - x0, y1 - y0
    if width < 0.16 or box_height < 0.16:
        return
    n_sites = int(np.clip(round(width * box_height * 40), 3, 8))
    min_dist = 0.14 * max(width, box_height)
    sites = []
    for _ in range(n_sites):
        for _ in range(80):
            candidate = rng.uniform([x0, y0], [x1, y1])
            if all(np.hypot(*(candidate - other)) >= min_dist for other in sites):
                sites.append(candidate)
                break
    sites = [np.asarray(site, dtype=float) for site in sites]
    if len(sites) < 2:
        return
    cells = _voronoi_cells(sites, box=box)
    edges = _shared_edges(cells, sites)
    uses = []
    for _ in sites:
        roll = rng.random()
        uses.append("grass" if roll < 0.12 else ("soil" if roll < 0.22 else "crop"))
    for index, poly in enumerate(cells):
        draw_field_polygon(
            ax, rng, budget, poly, materials=materials, material_key=uses[index], zorder=zorder
        )
        objects.append(SceneObject("field", _poly_bbox(poly)))
    if terrain is not None:
        elevation = terrain[0]

        def _grade(key):
            p0, p1 = edges[key]
            return abs(_sample(elevation, p0[0], p0[1]) - _sample(elevation, p1[0], p1[1]))

        tree = _close_network(_spanning_tree(len(sites), edges, weight=_grade), edges)
    else:
        tree = _close_network(_spanning_tree(len(sites), edges), edges)
    for key in tree:
        p0, p1 = edges[key]
        draw_road(
            ax,
            rng,
            budget,
            points=[p0, p1],
            width=rng.uniform(0.012, 0.018),
            materials=materials,
            dashes=False,
            shoulder=False,
        )
    for key in edges:
        if key in tree or rng.random() > 0.35:
            continue
        p0, p1 = edges[key]
        _hedge_edge(ax, rng, budget, materials, p0, p1, objects)


def forest_scene(size: int = 600, seed: int | None = None) -> Scene:
    """Dense forest: leaf-litter ground, woodland stands, clearings and tracks."""

    def drawer(ax, rng, budget, materials):
        objects = []
        terrain = _terrain(rng)
        _relief_overlay(ax, budget, terrain, zorder=1.5, alpha=0.4)
        _ground_patches(
            ax,
            rng,
            budget,
            materials,
            ["grass", "soil", "soil"],
            int(rng.integers(4, 7)),
            size_range=(0.12, 0.26),
        )
        if rng.random() < 0.55:
            path = _low_valley_path(rng, terrain[0], meander=0.13)
            draw_river(
                ax,
                rng,
                budget,
                points=path,
                width=rng.uniform(0.045, 0.075),
                materials=materials,
            )
        for _ in range(int(rng.integers(4, 8))):
            cx, cy = rng.uniform(0.1, 0.9, size=2)
            _forest_patch(ax, rng, budget, materials, objects, cx, cy, rng.uniform(0.12, 0.22))
        n = int(np.clip(size * size / 4200, 25, 130))
        objects.extend(_place_vegetation(ax, rng, budget, materials, n, (0.05, 0.1), 0.06))
        objects.extend(_place_vegetation(ax, rng, budget, materials, n // 4, (0.025, 0.05), 0.035))
        if rng.random() < 0.6:
            path = _gentle_path(rng, terrain[1], n=18, meander=0.12)
            draw_road(
                ax,
                rng,
                budget,
                points=path,
                width=0.018,
                materials=materials,
                dashes=False,
            )
        return objects

    return _compose(
        "forest",
        size,
        seed,
        "soil",
        drawer,
        ["foliage", "bark", "soil", "grass", "water"],
    )


def wilderness_scene(size: int = 600, seed: int | None = None) -> Scene:
    """Dense, road-less forest: an almost continuous canopy with clearings."""

    def drawer(ax, rng, budget, materials):
        objects = []
        terrain = _terrain(rng)
        _relief_overlay(ax, budget, terrain, zorder=1.5, alpha=0.4)
        for _ in range(int(rng.integers(2, 5))):
            cx, cy = rng.uniform(0.1, 0.9, size=2)
            fill_patch(
                ax,
                patches.Polygon(
                    _blob(rng, cx, cy, rng.uniform(0.05, 0.1), n_vertices=int(rng.integers(9, 14))),
                    closed=True,
                    edgecolor="none",
                ),
                materials["grass"],
                budget=budget,
                zorder=1,
                alpha=0.5,
            )
        if rng.random() < 0.5:
            path = _low_valley_path(rng, terrain[0], meander=0.14)
            draw_river(
                ax, rng, budget, points=path,
                width=rng.uniform(0.03, 0.05), materials=materials,
            )
        n = int(np.clip(size * size / 2200, 30, 260))
        objects.extend(_place_vegetation(ax, rng, budget, materials, n, (0.05, 0.11), 0.045))
        objects.extend(
            _place_vegetation(ax, rng, budget, materials, n // 3, (0.025, 0.055), 0.028)
        )
        return objects

    return _compose(
        "wilderness", size, seed, "foliage", drawer,
        ["foliage", "bark", "soil", "grass", "water"],
    )


def pasture_scene(size: int = 600, seed: int | None = None) -> Scene:
    """Open pasture: meadows, hedgerows, scattered trees, a track and a farm."""

    def drawer(ax, rng, budget, materials):
        objects = []
        terrain = _terrain(rng)
        _relief_overlay(ax, budget, terrain, zorder=1.5, alpha=0.35)
        _ground_patches(
            ax, rng, budget, materials, ["grass", "grass", "soil"],
            int(rng.integers(6, 11)), size_range=(0.18, 0.4),
        )
        if rng.random() < 0.8:
            path = _gentle_path(rng, terrain[1], n=18, meander=0.12)
            draw_road(
                ax, rng, budget, points=path, width=rng.uniform(0.014, 0.02),
                materials=materials, material_key="gravel", color="#b7a06a",
                meander=0.0, dashes=False, shoulder=False,
            )
            track_mid = path[len(path) // 2]
        else:
            track_mid = np.array([0.5, 0.5])
        for _ in range(int(rng.integers(3, 6))):
            p0 = rng.uniform(0.05, 0.95, size=2)
            angle = rng.uniform(0.0, np.pi)
            length = rng.uniform(0.2, 0.45)
            p1 = np.clip(p0 + length * np.array([np.cos(angle), np.sin(angle)]), 0.03, 0.97)
            _hedge_edge(ax, rng, budget, materials, p0, p1, objects)
        for _ in range(int(rng.integers(5, 9))):
            cx, cy = rng.uniform(0.08, 0.92, size=2)
            _forest_patch(ax, rng, budget, materials, objects, cx, cy, rng.uniform(0.06, 0.13))
        for _ in range(int(rng.integers(7, 13))):
            if budget.full:
                break
            label, palette = _pick_plant(rng, conifer_bias=0.2, bush_prob=0.15)
            bbox = OBJECTS[label](
                ax, rng, budget, rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95),
                rng.uniform(0.03, 0.06), palette=palette, materials=materials,
            )
            objects.append(SceneObject(label, bbox))
        for _ in range(int(rng.integers(1, 3))):
            if budget.full:
                break
            offset = rng.uniform(-0.08, 0.08, size=2)
            bbox = OBJECTS["building"](
                ax, rng, budget, float(track_mid[0] + offset[0]), float(track_mid[1] + offset[1]),
                rng.uniform(0.06, 0.1), materials=materials,
                rotation=float(rng.uniform(0.0, 180.0)),
            )
            objects.append(SceneObject("building", bbox))
        if rng.random() < 0.6:
            bbox = OBJECTS["silo"](
                ax, rng, budget, float(track_mid[0] + rng.uniform(-0.1, 0.1)),
                float(track_mid[1] + rng.uniform(-0.1, 0.1)), rng.uniform(0.05, 0.08),
                materials=materials,
            )
            objects.append(SceneObject("silo", bbox))
        if rng.random() < 0.45:
            bbox = OBJECTS["greenhouse"](
                ax, rng, budget, rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85),
                rng.uniform(0.14, 0.22), materials=materials,
                rotation=float(rng.uniform(0.0, 180.0)),
            )
            objects.append(SceneObject("greenhouse", bbox))
        if rng.random() < 0.5:
            bbox = OBJECTS["pond"](
                ax, rng, budget, rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8),
                rng.uniform(0.08, 0.14), materials=materials,
            )
            objects.append(SceneObject("pond", bbox))
        return objects

    return _compose(
        "pasture", size, seed, "grass", drawer,
        ["grass", "soil", "foliage", "bark", "water", "concrete", "gravel", "roof"],
    )


def fields_scene(size: int = 600, seed: int | None = None) -> Scene:
    """Agricultural mosaic: irregular fields bounded by roads, hedges and rivers.

    The frame is partitioned into a Voronoi mosaic of field cells that tile the
    ground, so every parcel has a natural, non-rectangular boundary. A country
    road / dirt-track network (a spanning tree over the cells) gives every field
    frontage; hedgerows, woodland blocks and a river fill the other boundaries.
    """

    def drawer(ax, rng, budget, materials):
        objects = []
        terrain = _terrain(rng)
        height, slope, _ = terrain
        _relief_overlay(ax, budget, terrain, zorder=1.5, alpha=0.25)
        n_sites = int(np.clip(round(size / 55), 8, 16))
        sites = _scatter_positions(rng, n_sites, min_dist=0.13, margin=0.06)
        sites = [np.asarray(site, dtype=float) for site in sites]
        cells = _voronoi_cells(sites)
        edges = _shared_edges(cells, sites)

        # This is an agricultural mosaic, so most parcels are cropland. Only the
        # steepest and lowest ground becomes woodland / meadow and fallow
        # bare-soil parcels are rare. This keeps cropland reaching the frame edge
        # instead of leaving large non-crop regions around the border.
        uses = []
        eligible = []
        for index, site in enumerate(sites):
            cell_height = _sample(height, site[0], site[1])
            cell_slope = _sample(slope, site[0], site[1])
            if cell_slope > 0.78:
                uses.append("woodland")
            elif cell_height < 0.12:
                uses.append("grass")
            else:
                uses.append("crop")
                eligible.append((cell_slope, index))
        if eligible:
            uses[min(eligible)[1]] = "orchard"
        for index in range(len(uses)):
            if uses[index] == "crop" and rng.random() < 0.08:
                uses[index] = "soil"

        for index, poly in enumerate(cells):
            if budget.full:
                break
            if uses[index] == "woodland":
                fill_patch(
                    ax,
                    patches.Polygon(poly, closed=True, edgecolor="none"),
                    materials["foliage"],
                    budget=budget,
                    zorder=1,
                )
            elif uses[index] == "orchard":
                draw_field_polygon(
                    ax, rng, budget, poly, materials=materials,
                    material_key="soil", zorder=1, draw_rows=False,
                )
                center = poly.mean(axis=0)
                x0, y0, x1, y1 = _poly_bbox(poly)
                orchard_size = min(x1 - x0, y1 - y0) * 0.8
                bbox = OBJECTS["orchard"](
                    ax, rng, budget, float(center[0]), float(center[1]), orchard_size,
                    materials=materials, rotation=float(rng.uniform(0.0, 180.0)),
                )
                objects.append(SceneObject("orchard", bbox))
            else:
                draw_field_polygon(
                    ax,
                    rng,
                    budget,
                    poly,
                    materials=materials,
                    material_key=uses[index],
                    zorder=1,
                )
            objects.append(SceneObject("field", _poly_bbox(poly)))

        def _edge_grade(key):
            p0, p1 = edges[key]
            return abs(_sample(height, p0[0], p0[1]) - _sample(height, p1[0], p1[1]))

        road_edges = _close_network(
            _spanning_tree(len(sites), edges, weight=_edge_grade), edges
        )
        extra = [key for key in edges if key not in road_edges]
        rng.shuffle(extra)
        road_edges.update(extra[: int(rng.integers(1, 4))])
        road_midpoints = []
        for key in road_edges:
            p0, p1 = edges[key]
            road_midpoints.append((p0 + p1) / 2.0)
            if rng.random() < 0.55:
                draw_road(
                    ax,
                    rng,
                    budget,
                    points=[p0, p1],
                    width=rng.uniform(0.016, 0.024),
                    materials=materials,
                    meander=0.0,
                    dashes=False,
                    shoulder=False,
                )
            else:
                draw_road(
                    ax,
                    rng,
                    budget,
                    points=[p0, p1],
                    width=rng.uniform(0.010, 0.016),
                    materials=materials,
                    material_key="gravel",
                    color="#b7a06a",
                    meander=0.0,
                    dashes=False,
                    shoulder=False,
                )

        for key in edges:
            if key in road_edges or rng.random() > 0.4:
                continue
            p0, p1 = edges[key]
            _hedge_edge(ax, rng, budget, materials, p0, p1, objects)

        if rng.random() < 0.5:
            path = _low_valley_path(rng, height, n=18, meander=0.16)
            draw_river(
                ax,
                rng,
                budget,
                points=path,
                width=rng.uniform(0.05, 0.08),
                materials=materials,
            )

        for index, poly in enumerate(cells):
            if uses[index] != "woodland" or budget.full:
                continue
            polygon_path = Path(poly)
            candidates = rng.uniform(0.0, 1.0, size=(80, 2))
            inside = polygon_path.contains_points(candidates)
            points = candidates[inside]
            n_trees = int(np.clip(len(points) * 0.6, 6, 45))
            for cx, cy in points[:n_trees]:
                if budget.full:
                    break
                label, palette = _pick_plant(rng, conifer_bias=0.3, bush_prob=0.2)
                bbox = OBJECTS[label](
                    ax,
                    rng,
                    budget,
                    float(cx),
                    float(cy),
                    rng.uniform(0.035, 0.075),
                    palette=palette,
                    materials=materials,
                )
                objects.append(SceneObject(label, bbox))

        for index, poly in enumerate(cells):
            if uses[index] != "grass" or budget.full:
                continue
            center = poly.mean(axis=0)
            for _ in range(int(rng.integers(1, 4))):
                if budget.full:
                    break
                cx = float(center[0] + rng.normal(0.0, 0.03))
                cy = float(center[1] + rng.normal(0.0, 0.03))
                bbox = OBJECTS["tree"](
                    ax, rng, budget, cx, cy, rng.uniform(0.03, 0.055), materials=materials
                )
                objects.append(SceneObject("tree", bbox))

        for _ in range(int(rng.integers(1, 3))):
            if budget.full or not road_midpoints:
                break
            base = road_midpoints[int(rng.integers(len(road_midpoints)))]
            offset = rng.uniform(-0.06, 0.06, size=2)
            bbox = OBJECTS["silo"](
                ax,
                rng,
                budget,
                float(base[0] + offset[0]),
                float(base[1] + offset[1]),
                rng.uniform(0.05, 0.08),
                materials=materials,
            )
            objects.append(SceneObject("silo", bbox))

        for index, poly in enumerate(cells):
            if uses[index] != "grass" or budget.full:
                continue
            if rng.random() < 0.5:
                center = poly.mean(axis=0)
                pond_size = rng.uniform(0.08, 0.14)
                if pond_size < np.ptp(poly[:, 0]) and pond_size < np.ptp(poly[:, 1]):
                    bbox = OBJECTS["pond"](
                        ax,
                        rng,
                        budget,
                        float(center[0]),
                        float(center[1]),
                        pond_size,
                        materials=materials,
                    )
                    objects.append(SceneObject("pond", bbox))
            break
        return objects

    return _compose(
        "fields",
        size,
        seed,
        "grass",
        drawer,
        ["grass", "crop", "soil", "water", "foliage", "bark", "concrete", "gravel"],
    )


def urban_scene(size: int = 600, seed: int | None = None) -> Scene:
    """City block: a warped street network with buildings, parks and cars."""

    def drawer(ax, rng, budget, materials):
        objects = []
        corridors = []
        n = int(rng.integers(6, 9))
        theta = float(rng.uniform(0.0, 180.0))
        shear = float(rng.uniform(-0.35, 0.35))

        def transform(points):
            points = np.asarray(points, dtype=float)
            sheared = np.column_stack([points[:, 0] + shear * points[:, 1], points[:, 1]])
            angle = np.deg2rad(theta)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            return sheared @ rotation.T + np.array([0.5, 0.5])

        coords = np.linspace(-0.5, 1.5, n + 1)
        step = float(coords[1] - coords[0])
        x_offsets = coords + rng.uniform(-0.015, 0.015, n + 1)
        y_offsets = coords + rng.uniform(-0.015, 0.015, n + 1)
        for i in range(n + 1):
            if budget.full:
                break
            local = _meander_points(rng, (x_offsets[i], -0.9), (x_offsets[i], 1.9), meander=0.022)
            points = transform(local)
            draw_road(ax, rng, budget, points=points, width=0.06, materials=materials)
            corridors.append((points, 0.06))
            local = _meander_points(rng, (-0.9, y_offsets[i]), (1.9, y_offsets[i]), meander=0.022)
            points = transform(local)
            draw_road(ax, rng, budget, points=points, width=0.06, materials=materials)
            corridors.append((points, 0.06))
        if rng.random() < 0.7:
            arterial = _meander_points(rng, (-0.5, -0.4), (1.5, 1.4), n=16, meander=0.12)
            points = transform(arterial)
            draw_road(ax, rng, budget, points=points, width=0.085, materials=materials)
            corridors.append((points, 0.085))

        cells = [(i, j) for i in range(n) for j in range(n)]
        rng.shuffle(cells)
        n_park = int(rng.integers(2, 5))
        park_cells = set(cells[:n_park])
        amenity_pool = set(cells[n_park:])
        parking_budget = int(rng.integers(1, 3))
        plaza_budget = 1 if rng.random() < 0.6 else 0
        for i in range(n):
            for j in range(n):
                if budget.full:
                    break
                center = transform(
                    [
                        (
                            (x_offsets[i] + x_offsets[i + 1]) / 2,
                            (y_offsets[j] + y_offsets[j + 1]) / 2,
                        )
                    ]
                )[0]
                cx, cy = float(center[0]), float(center[1])
                if not (0.04 <= cx <= 0.96 and 0.04 <= cy <= 0.96):
                    continue
                if (i, j) in park_cells:
                    _ground_patches(
                        ax,
                        rng,
                        budget,
                        materials,
                        ["grass"],
                        1,
                        size_range=(step * 0.7, step * 0.8),
                    )
                    _forest_patch(
                        ax, rng, budget, materials, objects, cx, cy, step * 0.35, density=5
                    )
                    continue
                if parking_budget > 0 and (i, j) in amenity_pool:
                    clearance = _road_clearance((cx, cy), corridors)
                    parking_size = min(step * rng.uniform(0.85, 1.0), 1.4 * clearance - 0.01)
                    if parking_size >= step * 0.28:
                        bbox = OBJECTS["parking"](
                            ax, rng, budget, cx, cy, parking_size,
                            materials=materials, rotation=theta,
                        )
                        objects.append(SceneObject("parking", bbox))
                        parking_budget -= 1
                        continue
                if plaza_budget > 0 and (i, j) in amenity_pool:
                    clearance = _road_clearance((cx, cy), corridors)
                    plaza_size = min(step * rng.uniform(0.45, 0.6), 1.4 * clearance - 0.01)
                    if plaza_size >= step * 0.28:
                        bbox = OBJECTS["plaza"](
                            ax, rng, budget, cx, cy, plaza_size,
                            materials=materials, rotation=theta,
                        )
                        objects.append(SceneObject("plaza", bbox))
                        plaza_budget -= 1
                        continue
                if rng.random() < 0.06:
                    pool_size = step * rng.uniform(0.3, 0.45)
                    if _near_road((cx, cy), corridors, pool_size * 0.6):
                        continue
                    bbox = OBJECTS["pool"](
                        ax, rng, budget, cx, cy, pool_size, materials=materials, rotation=theta
                    )
                    objects.append(SceneObject("pool", bbox))
                    continue
                if rng.random() < 0.16:
                    bbox = OBJECTS["tree"](
                        ax, rng, budget, cx, cy, step * rng.uniform(0.4, 0.6), materials=materials
                    )
                    objects.append(SceneObject("tree", bbox))
                    continue
                building_size = min(
                    step * rng.uniform(0.32, 0.62),
                    1.3 * _road_clearance((cx, cy), corridors) - 0.005,
                )
                if building_size < step * 0.24:
                    continue
                building_rot = theta + float(rng.uniform(-12.0, 12.0))
                bbox = draw_building(
                    ax,
                    rng,
                    budget,
                    cx,
                    cy,
                    building_size,
                    materials=materials,
                    rotation=building_rot,
                )
                objects.append(SceneObject("building", bbox))
                # A second, smaller building fills the block like a real city
                # block instead of leaving a large slab of bare concrete.
                if rng.random() < 0.6 and not budget.full:
                    offset = building_size * rng.uniform(0.55, 0.85)
                    angle = building_rot + float(rng.uniform(-90.0, 90.0))
                    second_x = cx + offset * np.cos(np.deg2rad(angle))
                    second_y = cy + offset * np.sin(np.deg2rad(angle))
                    second_size = building_size * rng.uniform(0.55, 0.8)
                    if (0.06 <= second_x <= 0.94 and 0.06 <= second_y <= 0.94
                            and _road_clearance((second_x, second_y), corridors)
                            > second_size * 0.75):
                        second = draw_building(
                            ax, rng, budget, float(second_x), float(second_y), second_size,
                            materials=materials,
                            rotation=theta + float(rng.uniform(-12.0, 12.0)),
                        )
                        objects.append(SceneObject("building", second))
        if rng.random() < 0.45:
            k = int(rng.integers(0, n + 1))
            col = int(rng.integers(0, n + 1))
            island = transform([(x_offsets[k], y_offsets[col])])[0]
            if 0.07 <= island[0] <= 0.93 and 0.07 <= island[1] <= 0.93:
                patch = patches.Polygon(
                    _blob(rng, float(island[0]), float(island[1]), 0.05, n_vertices=12),
                    closed=True,
                    edgecolor="none",
                )
                if "grass" in materials:
                    fill_patch(ax, patch, materials["grass"], budget=budget, zorder=4)
                _forest_patch(
                    ax,
                    rng,
                    budget,
                    materials,
                    objects,
                    float(island[0]),
                    float(island[1]),
                    0.035,
                    density=4,
                )
        def _place_vehicle(px, py, rotation_deg):
            """Place a car (or a truck) at a road point and record its box."""
            if not (0.03 <= px <= 0.97 and 0.03 <= py <= 0.97):
                return  # keep vehicles fully inside the frame
            if rng.random() < 0.22:
                bbox = OBJECTS["truck"](
                    ax, rng, budget, float(px), float(py), rng.uniform(0.05, 0.065),
                    materials=materials, rotation=rotation_deg + float(rng.uniform(-8.0, 8.0)),
                )
                objects.append(SceneObject("truck", bbox))
            else:
                bbox = draw_car(
                    ax, rng, budget, float(px), float(py), rng.uniform(0.04, 0.056),
                    materials=materials, rotation=rotation_deg + float(rng.uniform(-8.0, 8.0)),
                )
                objects.append(SceneObject("car", bbox))

        # Moving traffic along the streets.
        for _ in range(n * 5):
            if budget.full:
                break
            row = int(rng.integers(0, n + 1))
            point = transform([(rng.uniform(0.05, 0.95), y_offsets[row])])[0]
            if not (0.05 <= point[0] <= 0.95 and 0.05 <= point[1] <= 0.95):
                continue
            _place_vehicle(point[0], point[1], theta)

        # Cars parked along the kerb (just off the centre line).
        for _ in range(n * 2):
            if budget.full:
                break
            row = int(rng.integers(0, n + 1))
            offset = -0.022 if rng.random() < 0.5 else 0.022
            point = transform([(rng.uniform(0.05, 0.95), y_offsets[row] + offset)])[0]
            if not (0.05 <= point[0] <= 0.95 and 0.05 <= point[1] <= 0.95):
                continue
            _place_vehicle(point[0], point[1], theta)

        # Occasional traffic jam: a queue of cars bumper-to-bumper on one street.
        for _ in range(int(rng.integers(1, 4))):
            row = int(rng.integers(0, n + 1))
            start_t = rng.uniform(0.05, 0.55)
            direction = 1.0 if rng.random() < 0.5 else -1.0
            for k in range(int(rng.integers(5, 10))):
                if budget.full:
                    break
                px = start_t + direction * k * rng.uniform(0.02, 0.028)
                if not (0.05 <= px <= 0.95):
                    break
                point = transform([(px, y_offsets[row])])[0]
                _place_vehicle(point[0], point[1], theta)
        return objects

    return _compose(
        "urban",
        size,
        seed,
        "concrete",
        drawer,
        ["concrete", "asphalt", "roof", "foliage", "bark", "water", "grass"],
    )


def coastal_scene(size: int = 600, seed: int | None = None) -> Scene:
    """Coastline: a sandy shore, a wavy waterline with foam, boats and rocks."""

    def drawer(ax, rng, budget, materials):
        objects = []
        xs = np.linspace(-0.1, 1.1, 24)
        base_y = rng.uniform(0.42, 0.58)
        phase = rng.uniform(0.0, 2 * np.pi)
        phase2 = rng.uniform(0.0, 2 * np.pi)
        shore = (
            base_y
            + 0.05 * np.sin(2 * np.pi * xs + phase)
            + 0.03 * np.sin(5 * np.pi * xs + phase)
            + 0.012 * np.sin(9 * np.pi * xs + phase2)
        )
        sea = np.vstack([np.column_stack([xs, shore]), [[1.1, -0.1], [-0.1, -0.1]]])
        fill_patch(
            ax,
            patches.Polygon(sea, closed=True, edgecolor="none"),
            materials["water"],
            budget=budget,
            zorder=2,
        )
        # A wet-sand gradient just above the waterline (two soft bands).
        for k, (band, tone) in enumerate([(0.03, "#a88d59"), (0.06, "#c0a569")]):
            wet = np.vstack(
                [
                    np.column_stack([xs, shore]),
                    np.column_stack([xs[::-1], shore[::-1] + band]),
                ]
            )
            budget.add_patch(
                ax,
                patches.Polygon(
                    wet,
                    closed=True,
                    facecolor=_jitter(rng, tone, 0.04),
                    edgecolor="none",
                    alpha=0.55 - 0.2 * k,
                    zorder=3,
                ),
            )
        # Foam broken into short segments so it is not a hard full-width line.
        for k in range(int(rng.integers(2, 4))):
            offset = 0.01 + k * 0.028
            yline = shore + offset + 0.006 * np.sin(7 * np.pi * xs + phase)
            segments = []
            j = 0
            while j + 1 < len(xs):
                span = j + int(rng.integers(2, 4))
                span = min(span, len(xs) - 1)
                segments.append([(xs[j], yline[j]), (xs[span], yline[span])])
                j = span + int(rng.integers(1, 3))
            budget.add_line_collection(
                ax,
                segments,
                color=(1.0, 1.0, 1.0),
                linewidth=0.9,
                alpha=float(rng.uniform(0.35, 0.6)),
                zorder=4,
            )
        for _ in range(int(rng.integers(1, 4))):
            if budget.full:
                break
            bx = rng.uniform(0.05, 0.95)
            by = rng.uniform(0.06, 0.38)
            bbox = OBJECTS["boat"](
                ax, rng, budget, bx, by, rng.uniform(0.08, 0.14), materials=materials
            )
            objects.append(SceneObject("boat", bbox))
        for _ in range(int(rng.integers(2, 5))):
            if budget.full:
                break
            bbox = OBJECTS["rock"](
                ax,
                rng,
                budget,
                rng.uniform(0.05, 0.95),
                rng.uniform(0.62, 0.95),
                rng.uniform(0.04, 0.08),
                materials=materials,
            )
            objects.append(SceneObject("rock", bbox))
        for _ in range(int(rng.integers(3, 7))):
            if budget.full:
                break
            label, palette = _pick_plant(rng, conifer_bias=0.2, bush_prob=0.3)
            bbox = OBJECTS[label](
                ax,
                rng,
                budget,
                rng.uniform(0.05, 0.95),
                rng.uniform(0.62, 0.95),
                rng.uniform(0.03, 0.07),
                palette=palette,
                materials=materials,
            )
            objects.append(SceneObject(label, bbox))
        return objects

    return _compose(
        "coastal",
        size,
        seed,
        "sand",
        drawer,
        ["sand", "water", "rock", "foliage", "bark", "concrete"],
    )


def mountain_scene(size: int = 600, seed: int | None = None) -> Scene:
    """Rocky mountain terrain: rock ground, snow caps, conifers and lakes."""

    def drawer(ax, rng, budget, materials):
        objects = []
        terrain = _terrain(rng)
        height, slope, _ = terrain
        _relief_overlay(ax, budget, terrain, zorder=1, alpha=0.45)
        _ground_patches(
            ax,
            rng,
            budget,
            materials,
            ["grass", "soil", "grass", "soil"],
            int(rng.integers(9, 16)),
            size_range=(0.22, 0.5),
            zorder=2,
        )
        for _ in range(int(rng.integers(4, 8))):
            if budget.full:
                break
            cx, cy = _pick_by_elevation(rng, height, lambda h: h > 0.55)
            fill_patch(
                ax,
                patches.Polygon(
                    _blob(
                        rng, cx, cy, rng.uniform(0.06, 0.14), n_vertices=int(rng.integers(8, 14))
                    ),
                    closed=True,
                ),
                materials["snow"],
                budget=budget,
                zorder=2,
                alpha=float(rng.uniform(0.3, 0.5)),
            )
        for _ in range(int(rng.integers(7, 14))):
            if budget.full:
                break
            label, palette = _pick_plant(rng, conifer_bias=0.65, bush_prob=0.15)
            cx, cy = _pick_by_elevation(rng, height, lambda h: 0.15 < h < 0.8)
            bbox = OBJECTS[label](
                ax, rng, budget, cx, cy, rng.uniform(0.03, 0.07),
                palette=palette, materials=materials,
            )
            objects.append(SceneObject(label, bbox))
        for _ in range(int(rng.integers(2, 6))):
            if budget.full:
                break
            cx, cy = _pick_by_elevation(rng, slope, lambda s: s > 0.45)
            bbox = OBJECTS["rock"](
                ax, rng, budget, cx, cy, rng.uniform(0.05, 0.1), materials=materials
            )
            objects.append(SceneObject("rock", bbox))
        if rng.random() < 0.5:
            px, py = _pick_by_elevation(rng, height, lambda h: h < 0.35)
            bbox = OBJECTS["pond"](
                ax, rng, budget, px, py, rng.uniform(0.1, 0.16), materials=materials
            )
            objects.append(SceneObject("pond", bbox))
        return objects

    return _compose(
        "mountain",
        size,
        seed,
        "rock",
        drawer,
        ["rock", "snow", "foliage", "bark", "water", "grass", "soil"],
    )


def arid_scene(size: int = 600, seed: int | None = None) -> Scene:
    """Arid region: sand, a dry gravel wadi, rock outcrops and sparse scrub."""

    def drawer(ax, rng, budget, materials):
        objects = []
        _ground_patches(
            ax, rng, budget, materials, ["gravel", "sand", "sand"],
            int(rng.integers(2, 5)), size_range=(0.12, 0.26),
        )
        if rng.random() < 0.85:   # a dry wadi in most scenes
            if rng.random() < 0.5:
                a = (rng.uniform(-0.1, 0.15), rng.uniform(0.2, 0.8))
                b = (rng.uniform(0.85, 1.1), rng.uniform(0.2, 0.8))
            else:
                a = (rng.uniform(0.2, 0.8), rng.uniform(-0.1, 0.15))
                b = (rng.uniform(0.2, 0.8), rng.uniform(0.85, 1.1))
            path = _meander_points(rng, a, b, n=26, meander=0.14)
            width = rng.uniform(0.09, 0.13)
            budget.add_patch(
                ax,
                patches.Polygon(
                    _ribbon(path, width * 1.6),
                    closed=True,
                    facecolor=_jitter(rng, "#bfae82", 0.05),
                    edgecolor="none",
                    zorder=2,
                ),
            )
            fill_patch(
                ax,
                patches.Polygon(_ribbon(path, width), closed=True),
                materials["gravel"],
                budget=budget,
                zorder=2,
            )
            budget.add_line(
                ax,
                path[:, 0].tolist(),
                path[:, 1].tolist(),
                color="#6f6349",
                linewidth=1.1,
                alpha=0.55,
                zorder=3,
            )
        for _ in range(int(rng.integers(8, 15))):
            if budget.full:
                break
            bbox = OBJECTS["rock"](
                ax,
                rng,
                budget,
                rng.uniform(0.05, 0.95),
                rng.uniform(0.05, 0.95),
                rng.uniform(0.04, 0.09),
                materials=materials,
            )
            objects.append(SceneObject("rock", bbox))
        for _ in range(int(rng.integers(6, 12))):
            if budget.full:
                break
            bbox = OBJECTS["bush"](
                ax,
                rng,
                budget,
                rng.uniform(0.05, 0.95),
                rng.uniform(0.05, 0.95),
                rng.uniform(0.03, 0.06),
                palette=str(rng.choice(["#6f8f45", "#7a8a4a", "#8a9a5a"])),
                materials=materials,
            )
            objects.append(SceneObject("bush", bbox))
        return objects

    return _compose(
        "arid", size, seed, "sand", drawer, ["sand", "gravel", "rock", "foliage", "bark"]
    )


def forest_fields_scene(size: int = 600, seed: int | None = None) -> Scene:
    """Boundary scene: woodland on one side, agricultural plots on the other."""

    def drawer(ax, rng, budget, materials):
        objects = []
        terrain = _terrain(rng)
        split = rng.uniform(0.4, 0.6)
        ys = np.linspace(-0.1, 1.1, 24)
        x_edge = _organic_edge(ys, split, rng)
        # Woodland half: fill with canopy material (not bare soil) and pile on
        # forest patches so the boundary reads as forest against open fields.
        left = np.vstack([[-0.1, -0.1], np.column_stack([x_edge, ys]), [-0.1, 1.1]])
        fill_patch(
            ax, patches.Polygon(left, closed=True), materials["foliage"], budget=budget, zorder=1
        )
        _soft_edge(ax, rng, budget, materials, ys, x_edge, "foliage", side=1)
        _soft_edge(ax, rng, budget, materials, ys, x_edge, "grass", side=-1)
        for _ in range(int(rng.integers(6, 11))):
            if budget.full:
                break
            cy = rng.uniform(0.1, 0.9)
            xe = float(np.interp(cy, ys, x_edge))
            _forest_patch(
                ax, rng, budget, materials, objects,
                rng.uniform(0.03, max(0.05, xe - 0.06)), cy, rng.uniform(0.12, 0.22),
            )
        for _ in range(int(rng.integers(6, 11))):
            if budget.full:
                break
            cy = rng.uniform(0.02, 0.98)
            xe = float(np.interp(cy, ys, x_edge))
            label, palette = _pick_plant(rng, conifer_bias=0.4)
            bbox = OBJECTS[label](
                ax, rng, budget, xe + rng.uniform(-0.03, 0.03), cy,
                rng.uniform(0.03, 0.06), palette=palette, materials=materials,
            )
            objects.append(SceneObject(label, bbox))
        # Field mosaic fills the agricultural half all the way to the frame
        # edges (no unexplained grass/soil margin around the parcels).
        right0 = float(x_edge.max()) + 0.03
        _field_mosaic(
            ax, rng, budget, materials, objects, (right0, 0.0, 1.0, 1.0),
            terrain=terrain,
        )
        return objects

    return _compose(
        "forest_fields", size, seed, "grass", drawer,
        ["grass", "crop", "soil", "water", "foliage", "bark", "concrete", "gravel"],
    )


def urban_fringe_scene(size: int = 600, seed: int | None = None) -> Scene:
    """Boundary scene: city blocks on one side, agricultural plots on the other."""

    def drawer(ax, rng, budget, materials):
        objects = []
        terrain = _terrain(rng)
        split = rng.uniform(0.42, 0.6)
        ys = np.linspace(-0.1, 1.1, 24)
        x_edge = _organic_edge(ys, split, rng, amplitude=0.04)
        right = np.vstack([[1.1, -0.1], [1.1, 1.1], np.column_stack([x_edge, ys])[::-1]])
        fill_patch(
            ax, patches.Polygon(right, closed=True), materials["concrete"], budget=budget, zorder=1
        )
        _soft_edge(ax, rng, budget, materials, ys, x_edge, "concrete", side=-1)
        _soft_edge(ax, rng, budget, materials, ys, x_edge, "grass", side=1)
        if rng.random() < 0.6:
            path = np.column_stack([x_edge + 0.015, ys])
            draw_river(
                ax, rng, budget, points=path, width=rng.uniform(0.04, 0.07), materials=materials
            )
        x_min = float(x_edge.min())
        for _ in range(int(rng.integers(2, 4))):
            x = rng.uniform(x_min + 0.12, 0.95)
            draw_road(
                ax, rng, budget, (x, -0.1), (x + rng.uniform(-0.1, 0.1), 1.1),
                width=0.04, materials=materials, meander=0.03,
            )
        for _ in range(int(rng.integers(2, 5))):
            if budget.full:
                break
            bbox = draw_building(
                ax, rng, budget, rng.uniform(x_min + 0.14, 0.93), rng.uniform(0.1, 0.9),
                rng.uniform(0.08, 0.14), materials=materials,
                rotation=float(rng.uniform(0.0, 180.0)),
            )
            objects.append(SceneObject("building", bbox))
        # The agricultural half must reach the frame edges (no unexplained grass
        # margin around the parcels).
        left_end = x_min - 0.03
        _field_mosaic(
            ax, rng, budget, materials, objects, (0.0, 0.0, left_end, 1.0),
            terrain=terrain,
        )
        return objects

    return _compose(
        "urban_fringe", size, seed, "grass", drawer,
        ["grass", "crop", "soil", "water", "concrete", "asphalt", "roof", "foliage", "bark"],
    )



def _rubble_breakwater(ax, rng, budget, materials, objects, p0, p1, width):
    """A rubble-mound breakwater: a concrete core lined with large boulders."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    direction = p1 - p0
    length = float(np.hypot(*direction))
    if length < 0.06:
        return
    unit = direction / length
    normal = np.array([-unit[1], unit[0]])
    half = width / 2.0
    core = np.array([
        p0 + normal * half, p1 + normal * half, p1 - normal * half, p0 - normal * half,
    ])
    fill_patch(
        ax, patches.Polygon(core, closed=True), materials["concrete"], budget=budget, zorder=4
    )
    for t in np.linspace(0.04, 0.96, max(3, int(length / (width * 0.6)))):
        for side in (-1.0, 1.0):
            centre = p0 + t * direction + side * normal * half
            OBJECTS["rock"](
                ax, rng, budget, float(centre[0]), float(centre[1]),
                width * rng.uniform(0.55, 0.85), materials=materials,
            )
    objects.append(SceneObject("breakwater", (
        min(p0[0], p1[0]) - half, min(p0[1], p1[1]) - half,
        max(p0[0], p1[0]) + half, max(p0[1], p1[1]) + half,
    )))


def _pier(ax, rng, budget, materials, objects, p0, p1, width):
    """A pier deck on pilings extending from the shore into the water."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    direction = p1 - p0
    length = float(np.hypot(*direction))
    if length < 0.06:
        return
    unit = direction / length
    normal = np.array([-unit[1], unit[0]])
    half = width / 2.0
    deck = np.array([
        p0 + normal * half, p1 + normal * half, p1 - normal * half, p0 - normal * half,
    ])
    budget.add_patch(ax, patches.Polygon(
        deck, closed=True, facecolor=_jitter(rng, "#c9c2b4", 0.05),
        edgecolor="#6f6757", linewidth=0.6, zorder=5,
    ))
    for t in np.linspace(0.1, 0.9, max(2, int(length / 0.05))):
        for side in (-1.0, 1.0):
            centre = p0 + t * direction + side * normal * (half * 0.8)
            budget.add_patch(ax, patches.Circle(
                (float(centre[0]), float(centre[1])), width * 0.08,
                facecolor="#5b5348", edgecolor="none", zorder=6,
            ))
    objects.append(SceneObject("pier", (
        min(p0[0], p1[0]) - half, min(p0[1], p1[1]) - half,
        max(p0[0], p1[0]) + half, max(p0[1], p1[1]) + half,
    )))


def _marina(ax, rng, budget, materials, objects, cx, cy, size, angle):
    """A marina basin: finger docks with rows of small moored boats."""
    theta = np.deg2rad(angle)
    direction = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-np.sin(theta), np.cos(theta)])
    centre = np.array([cx, cy])
    basin = _blob(rng, cx, cy, size * 0.62, n_vertices=16)
    # Water-filled basin with a quay edge so it reads apart from the open sea.
    fill_patch(ax, patches.Polygon(basin, closed=True), materials["water"], budget=budget, zorder=3)
    budget.add_patch(ax, patches.Polygon(
        basin, closed=True, facecolor="none", edgecolor="#8d8474", linewidth=1.2, zorder=4,
    ))
    for k in np.linspace(-0.7, 0.7, int(rng.integers(3, 6))):
        base = centre + k * size * normal
        a = base - direction * size * 0.45
        b = base + direction * size * 0.45
        budget.add_line(ax, [a[0], b[0]], [a[1], b[1]], color="#cfc7b6", linewidth=1.6, zorder=5)
        for t in (0.3, 0.7):
            for side in (-1.0, 1.0):
                spot = base + (t - 0.5) * 2 * direction * size * 0.45 + side * normal * size * 0.12
                bbox = OBJECTS["boat"](
                    ax, rng, budget, float(spot[0]), float(spot[1]),
                    size * rng.uniform(0.16, 0.24), materials=materials, rotation=angle + 90.0,
                )
                objects.append(SceneObject("boat", bbox))
    objects.append(SceneObject("marina", (cx - size, cy - size, cx + size, cy + size)))


def coastal_city_scene(size: int = 600, seed: int | None = None) -> Scene:
    """Coastal city: waterfront blocks, a marina, piers and a rubble breakwater."""

    def drawer(ax, rng, budget, materials):
        objects = []
        xs = np.linspace(-0.1, 1.1, 20)
        base_y = rng.uniform(0.30, 0.40)
        phase = rng.uniform(0.0, 2 * np.pi)
        shore = (
            base_y + 0.03 * np.sin(2 * np.pi * xs + phase)
            + 0.015 * np.sin(5 * np.pi * xs + phase)
        )
        # Sea fills the lower band; the city sits on the land side above it.
        sea = np.vstack([np.column_stack([xs, shore]), [[1.1, -0.1], [-0.1, -0.1]]])
        fill_patch(ax, patches.Polygon(sea, closed=True, edgecolor="none"),
                   materials["water"], budget=budget, zorder=2)
        wet = np.vstack([np.column_stack([xs, shore]),
                         np.column_stack([xs[::-1], shore[::-1] + 0.025])])
        budget.add_patch(ax, patches.Polygon(
            wet, closed=True, facecolor=_jitter(rng, "#b79a63", 0.05),
            edgecolor="none", zorder=3))
        for k in range(int(rng.integers(2, 4))):
            budget.add_line(
                ax, xs, shore - 0.008 - k * 0.02 + 0.006 * np.sin(7 * np.pi * xs + phase),
                color=(1.0, 1.0, 1.0, 0.5), linewidth=0.9, zorder=4,
            )
        # Breakwater parallel to the coast, out in the water.
        bx0 = rng.uniform(0.05, 0.35)
        _rubble_breakwater(
            ax, rng, budget, materials, objects,
            (bx0, base_y - 0.18), (bx0 + rng.uniform(0.4, 0.6), base_y - 0.20),
            width=rng.uniform(0.03, 0.05),
        )
        # Piers running from the shore out into the water.
        for _ in range(int(rng.integers(1, 3))):
            px = rng.uniform(0.1, 0.85)
            _pier(
                ax, rng, budget, materials, objects,
                (px, float(np.interp(px, xs, shore))),
                (px + rng.uniform(-0.05, 0.05), base_y - 0.16),
                width=rng.uniform(0.018, 0.03),
            )
        # Sheltered marina between the breakwater and the shore.
        _marina(
            ax, rng, budget, materials, objects,
            rng.uniform(0.45, 0.8), base_y - 0.10, rng.uniform(0.12, 0.18),
            angle=float(rng.uniform(0.0, 180.0)),
        )
        # Waterfront: a coast road, then buildings, parking and greenery.
        road = _meander_points(rng, (-0.1, base_y + 0.06), (1.1, base_y + 0.08), meander=0.01)
        draw_road(ax, rng, budget, points=road, width=0.05, materials=materials)
        corridors = [(road, 0.05)]
        # Cross streets give the waterfront blocks frontage instead of an open slab.
        for _ in range(int(rng.integers(2, 4))):
            cross_x = rng.uniform(0.15, 0.85)
            cross = _meander_points(
                rng, (cross_x, base_y + 0.06),
                (cross_x + rng.uniform(-0.05, 0.05), 1.1), meander=0.015,
            )
            draw_road(ax, rng, budget, points=cross, width=0.035, materials=materials)
            corridors.append((cross, 0.035))
        # A small waterfront park (kept on the land side, above the road).
        park_x, park_y = rng.uniform(0.15, 0.85), rng.uniform(base_y + 0.18, 0.85)
        park = _blob(rng, park_x, park_y, rng.uniform(0.10, 0.16), n_vertices=12)
        fill_patch(
            ax, patches.Polygon(park, closed=True), materials["grass"], budget=budget, zorder=1
        )
        # Buildings sit on a jittered grid of blocks, kept clear of the roads so
        # the waterfront reads as city blocks rather than scattered loose slabs.
        rows = np.linspace(base_y + 0.16, 0.94, 4)
        cols = np.linspace(0.08, 0.92, 4)
        cells = [(cx, cy) for cy in rows for cx in cols]
        rng.shuffle(cells)
        for cx, cy in cells:
            if budget.full:
                break
            jx = cx + rng.uniform(-0.02, 0.02)
            jy = cy + rng.uniform(-0.02, 0.02)
            building_size = rng.uniform(0.07, 0.12)
            if _road_clearance((jx, jy), corridors) < building_size * 0.75:
                continue
            bbox = OBJECTS["building"](
                ax, rng, budget, jx, jy, building_size, materials=materials,
                rotation=float(rng.uniform(0.0, 180.0)),
            )
            objects.append(SceneObject("building", bbox))
        for _ in range(int(rng.integers(1, 3))):
            if budget.full:
                break
            bbox = OBJECTS["parking"](
                ax, rng, budget, rng.uniform(0.1, 0.9), rng.uniform(base_y + 0.1, 0.6),
                rng.uniform(0.12, 0.18), materials=materials,
                rotation=float(rng.uniform(0.0, 180.0)),
            )
            objects.append(SceneObject("parking", bbox))
        for _ in range(int(rng.integers(3, 7))):
            if budget.full:
                break
            label, palette = _pick_plant(rng, conifer_bias=0.2, bush_prob=0.2)
            bbox = OBJECTS[label](
                ax, rng, budget, rng.uniform(0.05, 0.95), rng.uniform(base_y + 0.08, 0.95),
                rng.uniform(0.03, 0.06), palette=palette, materials=materials,
            )
            objects.append(SceneObject(label, bbox))
        for _ in range(int(rng.integers(8, 16))):
            if budget.full:
                break
            cx = rng.uniform(0.05, 0.95)
            cy = float(np.interp(cx, road[:, 0], road[:, 1])) + rng.uniform(-0.02, 0.02)
            bbox = OBJECTS["car"](
                ax, rng, budget, cx, cy, rng.uniform(0.04, 0.056),
                materials=materials, rotation=float(rng.uniform(0.0, 180.0)),
            )
            objects.append(SceneObject("car", bbox))
        # A short queue of cars on the coast road.
        jam_start = rng.uniform(0.05, 0.5)
        for k in range(int(rng.integers(4, 8))):
            if budget.full:
                break
            cx = jam_start + k * rng.uniform(0.022, 0.03)
            if cx > 0.95:
                break
            cy = float(np.interp(cx, road[:, 0], road[:, 1]))
            bbox = OBJECTS["car"](
                ax, rng, budget, cx, cy, rng.uniform(0.04, 0.052),
                materials=materials, rotation=float(rng.uniform(0.0, 180.0)),
            )
            objects.append(SceneObject("car", bbox))
        for _ in range(int(rng.integers(2, 5))):
            if budget.full:
                break
            bbox = OBJECTS["rock"](
                ax, rng, budget, rng.uniform(0.05, 0.95),
                float(rng.uniform(base_y + 0.01, base_y + 0.06)),
                rng.uniform(0.03, 0.06), materials=materials,
            )
            objects.append(SceneObject("rock", bbox))
        return objects

    return _compose(
        "coastal_city", size, seed, "concrete", drawer,
        ["sand", "water", "rock", "foliage", "bark", "concrete", "asphalt", "roof", "grass"],
    )


SCENES = {
    "urban": urban_scene,
    "forest": forest_scene,
    "fields": fields_scene,
    "coastal": coastal_scene,
    "mountain": mountain_scene,
    "arid": arid_scene,
    "pasture": pasture_scene,
    "wilderness": wilderness_scene,
    "forest_fields": forest_fields_scene,
    "urban_fringe": urban_fringe_scene,
    "coastal_city": coastal_city_scene,
}


def generate_scene(
    name: str, size: int = 600, seed: int | None = None, mask: bool = False
) -> Scene:
    """Render a named scene from :data:`SCENES`.

    With ``mask=True`` the scene is drawn with flat semantic-class colours
    instead of materials, for :mod:`mlpsyn.segmentation`.
    """
    try:
        generator = SCENES[name]
    except KeyError:
        known = ", ".join(sorted(SCENES))
        raise KeyError(f"unknown scene {name!r}; known scenes: {known}") from None
    previous = MASK_MODE["on"]
    MASK_MODE["on"] = mask
    try:
        return generator(size=size, seed=seed)
    finally:
        MASK_MODE["on"] = previous
