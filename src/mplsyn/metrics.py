"""Dataset-level diversity metrics.

These complement the human visual-inspection loop in ``VISUAL_QA.md`` by
summarising a collection of :class:`~mplsyn.scene.Scene` objects: what labels
appear, how balanced they are, how many objects each scene holds and how evenly
objects are spread over the frame.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from mplsyn.scene import Scene

__all__ = [
    "label_histogram",
    "objects_per_scene",
    "spatial_coverage",
    "diversity_report",
]


def label_histogram(scenes) -> dict[str, int]:
    """Total count of every object label across ``scenes``."""
    counter: Counter[str] = Counter()
    for scene in scenes:
        counter.update(scene.labels())
    return dict(sorted(counter.items()))


def objects_per_scene(scenes) -> list[int]:
    """Number of annotated objects in each scene."""
    return [len(scene.objects) for scene in scenes]


def spatial_coverage(scene: Scene, grid: int = 4) -> float:
    """Fraction of a ``grid x grid`` partition that contains an object centre."""
    if not scene.objects:
        return 0.0
    occupied = set()
    for obj in scene.objects:
        x0, y0, x1, y1 = obj.bbox
        cx = np.clip((x0 + x1) / 2.0, 0.0, 1.0)
        cy = np.clip((y0 + y1) / 2.0, 0.0, 1.0)
        occupied.add((min(int(cx * grid), grid - 1), min(int(cy * grid), grid - 1)))
    return len(occupied) / (grid * grid)


def diversity_report(scenes, grid: int = 4) -> dict:
    """A compact summary of label, count and spatial diversity."""
    histogram = label_histogram(scenes)
    counts = objects_per_scene(scenes)
    coverages = [spatial_coverage(scene, grid=grid) for scene in scenes]
    total = sum(histogram.values())
    probs = np.array(list(histogram.values()), dtype=float) / max(total, 1)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum()) if total else 0.0
    return {
        "n_scenes": len(scenes),
        "labels": histogram,
        "objects_total": total,
        "objects_per_scene": counts,
        "objects_min": int(min(counts)) if counts else 0,
        "objects_max": int(max(counts)) if counts else 0,
        "label_entropy": entropy,
        "mean_spatial_coverage": float(np.mean(coverages)) if coverages else 0.0,
    }