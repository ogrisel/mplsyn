"""Semantic segmentation labels.

A scene is re-rendered with flat colours keyed by the *material* each surface is
filled with, then mapped back to a small set of semantic classes. This yields a
pixel-accurate land-cover / object label map from the same deterministic spec as
the RGB image.

Class ``0`` is reserved for **unlabeled** pixels (shadows, outlines, anti-aliased
edges and small details), so training losses can ignore it.
"""

from __future__ import annotations

import numpy as np

from mlpsyn.scene import generate_scene

__all__ = [
    "CLASS_COLORS",
    "MATERIAL_SEG_CLASS",
    "SEGMENTATION_CLASSES",
    "semantic_labels",
    "semantic_mask",
]

#: Semantic class names; index 0 is the ignore/unlabeled class.
SEGMENTATION_CLASSES = [
    "unlabeled",
    "grassland",
    "cropland",
    "bare_soil",
    "vegetation",
    "water",
    "built",
    "road",
    "rock",
    "sand",
    "snow",
    "gravel",
    "wetland",
]

#: Flat RGB colour per semantic class, used to render the mask.
CLASS_COLORS = {
    "grassland": (0.36, 0.72, 0.28),
    "cropland": (0.82, 0.82, 0.28),
    "bare_soil": (0.55, 0.40, 0.24),
    "vegetation": (0.10, 0.45, 0.14),
    "water": (0.15, 0.45, 0.80),
    "built": (0.78, 0.22, 0.22),
    "road": (0.22, 0.22, 0.22),
    "rock": (0.55, 0.55, 0.58),
    "sand": (0.92, 0.82, 0.50),
    "snow": (0.95, 0.96, 1.00),
    "gravel": (0.62, 0.56, 0.44),
    "wetland": (0.28, 0.36, 0.20),
}

#: Material name -> semantic class.
MATERIAL_SEG_CLASS = {
    "grass": "grassland",
    "crop": "cropland",
    "soil": "bare_soil",
    "foliage": "vegetation",
    "bark": "vegetation",
    "water": "water",
    "concrete": "built",
    "roof": "built",
    "asphalt": "road",
    "rock": "rock",
    "sand": "sand",
    "snow": "snow",
    "gravel": "gravel",
    "mud": "wetland",
}


def _color_array() -> np.ndarray:
    """RGB lookup table indexed by segmentation class id."""
    return np.array(
        [CLASS_COLORS.get(name, (0.0, 0.0, 0.0)) for name in SEGMENTATION_CLASSES[1:]],
        dtype=np.float32,
    )


def semantic_label_map(rgb: np.ndarray, threshold: float = 0.30) -> np.ndarray:
    """Map a flat-colour mask render to integer class ids.

    Pixels whose nearest palette colour is farther than ``threshold`` become
    ``0`` (unlabeled).
    """
    colors = _color_array()
    image = rgb.astype(np.float32) / 255.0
    diff = image[..., None, :] - colors[None, None, :, :]
    distance = np.sqrt((diff * diff).sum(axis=-1))
    nearest = distance.argmin(axis=-1) + 1
    labels = np.where(distance.min(axis=-1) <= threshold, nearest, 0)
    return labels.astype(np.uint8)


def semantic_mask(scene_name: str, size: int = 256, seed: int | None = None):
    """Render ``scene_name`` in mask mode and return ``(scene, labels)``."""
    scene = generate_scene(scene_name, size=size, seed=seed, mask=True)
    labels = semantic_label_map(scene.image)
    return scene, labels


def semantic_labels(scene, threshold: float = 0.30) -> np.ndarray:
    """Return class ids for a mask-mode :class:`~mlpsyn.scene.Scene`."""
    return semantic_label_map(scene.image, threshold=threshold)