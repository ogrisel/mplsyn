"""Annotation export helpers and dataset splitting.

A :class:`~mplsyn.scene.Scene` already carries object-level boxes in normalised
coordinates with a bottom-left origin. These helpers convert them to the two
formats used by most detection pipelines:

- **COCO** ``{"images", "annotations", "categories"}`` with pixel
  ``[x, y, width, height]`` boxes and a top-left origin;
- **YOLO** ``"cls cx cy w h"`` lines with normalised values and a top-left
  origin.

Splitting is deterministic for a given seed so a dataset can be regenerated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mplsyn.scene import Scene

__all__ = [
    "category_index",
    "coco_dataset",
    "scene_to_coco",
    "train_val_test_split",
    "yolo_lines",
]


def category_index(scenes) -> dict[str, int]:
    """Build a stable ``{label: id}`` map (ids start at 1) from scene labels."""
    labels = sorted({label for scene in scenes for label in scene.labels()})
    return {label: index + 1 for index, label in enumerate(labels)}


def _pixel_xywh(obj, size: int) -> tuple[float, float, float, float]:
    """Object box in pixel (x, y, w, h) with a top-left origin."""
    x0, y0, x1, y1 = obj.pixel_bbox(size)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


def yolo_lines(scene: Scene) -> list[str]:
    """Return ``"cls cx cy w h"`` YOLO lines for a scene's objects."""
    categories = category_index([scene])
    lines = []
    for obj in scene.objects:
        x0, y0, x1, y1 = obj.bbox
        cx = (x0 + x1) / 2.0
        cy = 1.0 - (y0 + y1) / 2.0
        width = x1 - x0
        height = y1 - y0
        cls = categories[obj.label] - 1
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")
    return lines


def scene_to_coco(scene: Scene, size: int, categories: dict[str, int], image_id: int = 0):
    """COCO image + annotation dicts for one scene."""
    image = {
        "id": image_id,
        "file_name": f"{scene.name}_{scene.seed}_{image_id}.png",
        "width": size,
        "height": size,
    }
    annotations = []
    for obj in scene.objects:
        x, y, width, height = _pixel_xywh(obj, size)
        annotations.append(
            {
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": categories[obj.label],
                "bbox": [x, y, width, height],
                "area": width * height,
                "iscrowd": 0,
            }
        )
    return image, annotations


def coco_dataset(scenes, size: int):
    """A complete COCO-style dict for a list of scenes."""
    categories = category_index(scenes)
    images, annotations = [], []
    for image_id, scene in enumerate(scenes):
        image, anns = scene_to_coco(scene, size, categories, image_id=image_id)
        images.append(image)
        annotations.extend(anns)
    category_list = [
        {"id": index, "name": label, "supercategory": "object"}
        for label, index in sorted(categories.items(), key=lambda item: item[1])
    ]
    return {"images": images, "annotations": annotations, "categories": category_list}


@dataclass(frozen=True)
class Split:
    """Deterministic train/val/test index split."""

    train: list[int]
    val: list[int]
    test: list[int]


def train_val_test_split(
    n: int, ratios=(0.8, 0.1, 0.1), seed: int = 0
) -> Split:
    """Shuffle ``range(n)`` with ``seed`` and split it by ``ratios``."""
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must sum to 1")
    indices = np.random.default_rng(seed).permutation(n).tolist()
    n_train = round(ratios[0] * n)
    n_val = round(ratios[1] * n)
    return Split(
        train=indices[:n_train],
        val=indices[n_train : n_train + n_val],
        test=indices[n_train + n_val :],
    )