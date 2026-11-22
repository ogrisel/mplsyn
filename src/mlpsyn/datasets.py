"""Reusable synthetic-dataset builders used by the modeling notebooks.

The datasets here are **numpy-only**: they render scenes, apply sensor effects,
build labels/masks and return plain arrays. The modeling notebooks wrap the
arrays in torch tensors, so this module (and `import mlpsyn`) stays free of a
PyTorch dependency. Anchors use scikit-learn, imported lazily.
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path

from mlpsyn.materials import MATERIALS
from mlpsyn.scene import SCENES, generate_scene
from mlpsyn.segmentation import semantic_mask
from mlpsyn.sensor import apply_sensor, clouds

__all__ = [
    "CLASSIFICATION_CLASSES",
    "CLASSIFICATION_CLASS_INDEX",
    "CLASSIFICATION_CONFUSABLE",
    "CLASSIFICATION_SCENE_TO_CLASS",
    "DETECTION_CLASSES",
    "DETECTION_CLASS_INDEX",
    "DETECTION_LABEL_MAP",
    "DETECTION_MAX_OBJECTS",
    "DETECTION_SCENE_TYPES",
    "blend_two_scenes",
    "degrade_image",
    "encode_targets",
    "forest_targets",
    "kmeans_anchors",
    "make_classification_data",
    "make_cloud_data",
    "make_detection_data",
    "make_forest_data",
    "make_segmentation_data",
    "shape_iou",
    "split_classification",
    "tree_box",
]

# ---------------------------------------------------------------------------
# Scene classification
# ---------------------------------------------------------------------------

#: Scene type -> image-level class. Several scenes are deliberately ambiguous:
#: each class mixes a "textbook" scene with one or more transitional scenes so
#: the label depends on the dominant ground, not on an easy giveaway.
CLASSIFICATION_SCENE_TO_CLASS = {
    "fields": "cropland",
    "forest_fields": "cropland",     # farmland/woodland boundary
    "urban": "urban",
    "urban_fringe": "urban",         # city next to fields
    "coastal": "coastal",
    "coastal_city": "coastal",       # a grey harbour city reads almost urban
    "mountain": "mountain",
    "arid": "arid",
    "savanna": "arid",               # dry sand, but nearly forest-dense trees
    "scrubland": "arid",             # rocky and dry, easily taken for mountain
    "forest": "forest",
    "wilderness": "forest",
    "pasture": "pasture",
    "woodland_pasture": "pasture",   # meadow gone half-woodland
}
CLASSIFICATION_CLASSES = sorted(set(CLASSIFICATION_SCENE_TO_CLASS.values()))
CLASSIFICATION_CLASS_INDEX = {
    name: idx for idx, name in enumerate(CLASSIFICATION_CLASSES)
}

#: Classes that are genuinely easy to confuse, used to pick the rival half of a
#: mixture (a random rival such as urban/arid would be too obvious).
CLASSIFICATION_CONFUSABLE = {
    "cropland": ["pasture", "forest"],
    "pasture": ["cropland", "forest"],
    "forest": ["pasture", "cropland", "arid"],
    "arid": ["mountain", "forest"],
    "mountain": ["arid", "forest"],
    "coastal": ["urban", "cropland"],
    "urban": ["coastal"],
}


def blend_two_scenes(scene_a, scene_b, size, rng):
    """Blend two scenes with a soft, wavy boundary; return the RGB image.

    The ``scene_a`` (labelled) region covers a random 42-62 % of the width, so
    the label is *not* simply "the largest area": a frame can be labelled *arid*
    while its trees and soil (forest) cover the larger share. That is what makes
    the task genuinely ambiguous rather than a colour/area shortcut.
    """
    image_a = generate_scene(scene_a, size=size, seed=int(rng.integers(10_000))).image
    image_b = generate_scene(scene_b, size=size, seed=int(rng.integers(10_000))).image
    wobble = np.cumsum(rng.normal(0.0, size * 0.02, size))
    edge = np.clip(rng.uniform(0.42, 0.58) * size + wobble, 3, size - 4)
    columns = np.arange(size)[None, :]
    ramp_width = max(2.0, size * 0.04)
    weight = np.clip((edge[:, None] - columns) / ramp_width + 0.5, 0.0, 1.0)
    return image_a * weight[..., None] + image_b * (1.0 - weight)[..., None]


def make_classification_data(
    n_per_class, size=64, seed=0, mixture_fraction=0.5, mixture_confidence=0.65
):
    """Class-balanced classification dataset; a fraction are two-scene mixtures.

    Returns ``(images, labels)`` with ``images`` a float32 ``(N, 3, H, W)`` array
    in [0, 1] and ``labels`` an int64 ``(N,)`` array.
    """
    rng = np.random.default_rng(seed)
    options_by_class = {name: [] for name in CLASSIFICATION_CLASSES}
    for scene_name, class_name in CLASSIFICATION_SCENE_TO_CLASS.items():
        options_by_class[class_name].append(scene_name)
    images, labels = [], []
    counts = dict.fromkeys(CLASSIFICATION_CLASSES, 0)
    while min(counts.values()) < n_per_class:
        # Fill the class that currently has fewest samples, to stay balanced.
        class_name = min(CLASSIFICATION_CLASSES, key=lambda name: counts[name])
        candidates = options_by_class[class_name]
        scene_a = candidates[int(rng.integers(len(candidates)))]
        if rng.random() < mixture_fraction:
            rival = str(rng.choice(CLASSIFICATION_CONFUSABLE[class_name]))
            rivals = options_by_class[rival]
            scene_b = rivals[int(rng.integers(len(rivals)))]
            image = blend_two_scenes(scene_a, scene_b, size, rng)
            # A near-balanced mixture has no single correct label: assign it to
            # the target class with ``mixture_confidence`` and to the rival
            # otherwise, independently of which texture covers more area.
            label_class = class_name if rng.random() < mixture_confidence else rival
        else:
            image = generate_scene(scene_a, size=size, seed=int(rng.integers(10_000))).image
            label_class = class_name
        degraded = apply_sensor(
            np.clip(image, 0, 255).astype(np.uint8), rng,
            kind="poisson" if rng.random() < 0.5 else None,
            blur_radius=rng.uniform(0.2, 0.9), haze_strength=rng.uniform(0.0, 0.05),
        )
        images.append(degraded)
        labels.append(CLASSIFICATION_CLASS_INDEX[label_class])
        counts[label_class] += 1
    image_array = np.stack(images).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    return image_array, np.asarray(labels, dtype=np.int64)


def split_classification(images, labels, n_val_per_class, n_test_per_class):
    """Split a class-balanced dataset into train/val/test, class by class."""
    cut = n_val_per_class + n_test_per_class
    parts = {name: [] for name in ("train_images", "val_images", "test_images",
                                   "train_labels", "val_labels", "test_labels")}
    for class_id in range(len(CLASSIFICATION_CLASSES)):
        indices = np.nonzero(labels == class_id)[0]
        parts["train_images"].append(images[indices[:-cut]])
        parts["train_labels"].append(labels[indices[:-cut]])
        parts["val_images"].append(images[indices[-cut:-n_test_per_class]])
        parts["val_labels"].append(labels[indices[-cut:-n_test_per_class]])
        parts["test_images"].append(images[indices[-n_test_per_class:]])
        parts["test_labels"].append(labels[indices[-n_test_per_class:]])
    return tuple(np.concatenate(parts[key]) for key in (
        "train_images", "val_images", "test_images",
        "train_labels", "val_labels", "test_labels",
    ))


# ---------------------------------------------------------------------------
# Semantic segmentation and cloud occlusion
# ---------------------------------------------------------------------------


def degrade_image(image, rng):
    """Small realistic sensor noise; ~40 % of images get partial cloud cover."""
    if rng.random() < 0.4:
        return apply_sensor(
            image, rng, kind="poisson", blur_radius=0.5,
            haze_strength=0.06, cloud_coverage=rng.uniform(0.25, 0.5),
        )
    if rng.random() < 0.5:
        return apply_sensor(
            image, rng, kind="poisson" if rng.random() < 0.5 else None,
            blur_radius=rng.uniform(0.0, 0.4), haze_strength=rng.uniform(0.0, 0.06),
        )
    return image.copy()


def make_segmentation_data(n_samples, size=64, seed=0):
    """Paired ``(image, semantic-mask)`` samples with light degradation.

    Returns ``(images, labels)``: float32 ``(N, 3, H, W)`` images in [0, 1] and
    int64 ``(N, H, W)`` label maps (class 0 = unlabeled).
    """
    scene_names = sorted(SCENES)
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for _ in range(n_samples):
        scene_name = scene_names[int(rng.integers(len(scene_names)))]
        scene_seed = int(rng.integers(10_000))
        clean = generate_scene(scene_name, size=size, seed=scene_seed).image
        _, mask = semantic_mask(scene_name, size=size, seed=scene_seed)
        images.append(degrade_image(clean, rng))
        labels.append(mask)
    image_array = np.stack(images).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    return image_array, np.stack(labels).astype(np.int64)


def make_cloud_data(n_samples, size=64, seed=0):
    """Cloudy scenes paired with binary cloud-occlusion masks.

    Returns ``(images, masks)``: float32 ``(N, 3, H, W)`` images and float32
    ``(N, 1, H, W)`` masks in {0, 1}.
    """
    scene_names = sorted(SCENES)
    rng = np.random.default_rng(seed)
    images, masks = [], []
    for _ in range(n_samples):
        scene_name = scene_names[int(rng.integers(len(scene_names)))]
        scene_seed = int(rng.integers(10_000))
        clean_image = generate_scene(scene_name, size=size, seed=scene_seed).image
        cloud_coverage = float(rng.uniform(0.25, 0.6))
        cloudy_image, occlusion = clouds(
            clean_image, rng, coverage=cloud_coverage, return_mask=True
        )
        images.append(cloudy_image)
        masks.append((occlusion > 0.4).astype(np.float32))
    image_array = np.stack(images).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    return image_array, np.stack(masks)[:, None]


# ---------------------------------------------------------------------------
# Object detection and forest instances
# ---------------------------------------------------------------------------

DETECTION_LABEL_MAP = {
    "building": "building", "silo": "building",
    "rock": "rock",
    "tree": "tree", "conifer": "tree", "bush": "tree", "orchard": "tree",
    "car": "car", "truck": "car", "bus": "car",
    "boat": "boat",
    "parking": "parking",
    "breakwater": "rock",   # rubble moles read as rock from above
    "pier": "building",     # long waterfront structures
    "marina": "parking",    # sheltered basin full of moored boats
}
DETECTION_CLASSES = ["building", "rock", "tree", "car", "boat", "parking"]
DETECTION_CLASS_INDEX = {name: idx for idx, name in enumerate(DETECTION_CLASSES)}
#: Mostly urban cities, with some urban fringe and coastal (beach) scenes.
DETECTION_SCENE_TYPES = [
    "urban", "urban", "urban", "urban", "urban_fringe", "urban_fringe",
    "coastal", "coastal", "coastal_city", "coastal_city",
]
DETECTION_MAX_OBJECTS = 24  # keep the many small cars


def make_detection_data(n_samples, size=96, seed=0, max_objects=DETECTION_MAX_OBJECTS):
    """Urban/coastal detection images with their clean scene objects.

    Returns ``(images, objects_per_image)``: float32 ``(N, 3, H, W)`` images and
    a list of ``SceneObject`` lists (sorted large-first, capped at
    ``max_objects``).
    """
    rng = np.random.default_rng(seed)
    images, objects_per_image = [], []
    for _ in range(n_samples):
        scene_name = DETECTION_SCENE_TYPES[int(rng.integers(len(DETECTION_SCENE_TYPES)))]
        scene_seed = int(rng.integers(10_000))
        scene = generate_scene(scene_name, size=size, seed=scene_seed)
        image = degrade_image(scene.image, rng)  # keep boxes from the clean scene
        objects = [obj for obj in scene.objects if obj.label in DETECTION_LABEL_MAP]
        objects.sort(key=lambda obj: -((obj.bbox[2] - obj.bbox[0]) * (obj.bbox[3] - obj.bbox[1])))
        images.append(image)
        objects_per_image.append(objects[:max_objects])
    image_array = np.stack(images).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    return image_array, objects_per_image


def shape_iou(box_wh, anchor_wh):
    """Intersection-over-union of two centred boxes given as (width, height)."""
    width, height = box_wh
    anchor_w, anchor_h = anchor_wh
    intersection = min(width, anchor_w) * min(height, anchor_h)
    union = width * height + anchor_w * anchor_h - intersection
    return intersection / union if union > 0 else 0.0


def kmeans_anchors(boxes_wh, n_anchors, seed=0, log_space=True):
    """Cluster box shapes into ``n_anchors`` (width, height) anchors.

    With ``log_space=True`` clustering runs on ``(log w, log h)`` so large boxes
    do not dominate; pass ``log_space=False`` for a narrow size range. Uses
    scikit-learn's KMeans (imported lazily) with ``n_init=10``.
    """
    from sklearn.cluster import KMeans

    boxes_wh = np.asarray(boxes_wh, dtype=float)
    features = np.log(np.clip(boxes_wh, 1e-4, None)) if log_space else boxes_wh
    kmeans = KMeans(n_clusters=n_anchors, n_init=10, random_state=seed)
    kmeans.fit(features)
    anchors = np.exp(kmeans.cluster_centers_) if log_space else kmeans.cluster_centers_
    return np.sort(anchors, axis=0)


def encode_targets(objects, anchors, grid, classes, class_index, label_map):
    """Encode scene objects into YOLO-style (objectness, class, box) targets."""
    n_anchors = len(anchors)
    objectness = np.zeros((grid, grid, n_anchors), np.float32)
    class_target = np.zeros((grid, grid, n_anchors, len(classes)), np.float32)
    box_target = np.zeros((grid, grid, n_anchors, 4), np.float32)
    for obj in objects:
        class_idx = class_index[label_map[obj.label]]
        x0, y0, x1, y1 = obj.bbox
        width, height = x1 - x0, y1 - y0
        if width <= 0 or height <= 0:
            continue  # skip degenerate (clipped-out) boxes
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        anchor_idx = int(np.argmax([shape_iou((width, height), a) for a in anchors]))
        col = int(np.clip(center_x * grid, 0, grid - 1))
        row = int(np.clip((1 - center_y) * grid, 0, grid - 1))
        objectness[row, col, anchor_idx] = 1
        class_target[row, col, anchor_idx, class_idx] = 1
        box_target[row, col, anchor_idx] = [
            center_x * grid - col,
            (1 - center_y) * grid - row,
            np.log(width / anchors[anchor_idx][0]),
            np.log(height / anchors[anchor_idx][1]),
        ]
    return objectness, class_target, box_target


def make_forest_data(n_samples, size=64, seed=0):
    """Synthetic forests with exact instance masks, from sparse to dense canopy.

    Four layouts vary tree count, spacing and order: sparse / medium / dense
    random stands and a regular avenue of street trees. Scattered boulders lie
    under the canopy but are not instances. Returns ``(images, instances)`` as a
    uint8 ``(N, H, W, 3)`` array and an int32 ``(N, H, W)`` array.
    """
    rng = np.random.default_rng(seed)
    images, instances = [], []
    grid_y, grid_x = np.mgrid[0:1:size * 1j, 0:1:size * 1j]
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    regimes = {
        "sparse": (7, 12, 0.17, (0.05, 0.08)),
        "medium": (14, 22, 0.11, (0.045, 0.075)),
        "dense": (20, 32, 0.075, (0.035, 0.065)),
        "avenue": (8, 14, 0.09, (0.04, 0.06)),
    }
    for _ in range(n_samples):
        regime = str(rng.choice(
            ["sparse", "medium", "dense", "avenue"], p=[0.3, 0.3, 0.2, 0.2]))
        low, high, min_dist, radius_range = regimes[regime]
        background_key = str(rng.choice(["soil", "grass", "foliage"]))
        image = MATERIALS[background_key](size=size, seed=int(rng.integers(1_000_000)))
        image = image.astype(np.float32) / 255.0
        instance = np.zeros((size, size), np.int32)
        n_boulders = int(rng.integers(1, 4)) if regime == "sparse" else (
            int(rng.integers(0, 3)) if regime == "medium" else 0)
        for _ in range(n_boulders):
            bx, by = rng.uniform(0.08, 0.92, size=2)
            brad = rng.uniform(0.018, 0.035)
            angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
            radii = brad * (1 + 0.35 * rng.normal(size=len(angles)))
            poly = np.column_stack([
                bx + radii * np.cos(angles), by + radii * np.sin(angles)])
            inside = Path(poly).contains_points(points).reshape(size, size)
            shade = float(rng.uniform(0.42, 0.6))
            rock = np.array([shade, shade, shade * 0.95])
            image[inside] = rock
            lift = ((grid_x - (bx - brad * 0.3)) ** 2
                    + (grid_y - (by - brad * 0.3)) ** 2) < (brad * 0.4) ** 2
            image[lift & inside] = np.clip(rock + 0.14, 0, 1)
        centres = []
        if regime == "avenue":
            image[int(0.44 * size):int(0.56 * size)] = np.array([0.30, 0.30, 0.31])
            step_x = max(3, size // 24)
            for x in range(0, size, step_x * 3):
                image[int(0.494 * size):int(0.506 * size), x:x + step_x] = 0.78
            for row_y in (0.27, 0.73):
                for k in range(int(rng.integers(4, 7))):
                    centres.append(np.array([
                        0.12 + k * 0.15 + rng.uniform(-0.02, 0.02),
                        row_y + rng.uniform(-0.025, 0.025)]))
        else:
            for _ in range(int(rng.integers(low, high + 1))):
                for _try in range(40):
                    candidate = rng.uniform(0.06, 0.94, size=2)
                    if all(np.hypot(*(candidate - centre)) >= min_dist for centre in centres):
                        centres.append(candidate)
                        break
        for instance_id, centre in enumerate(centres, start=1):
            radius = rng.uniform(*radius_range)
            angles = np.linspace(0, 2 * np.pi, 14, endpoint=False)
            radii = radius * (1 + 0.22 * rng.normal(size=len(angles)))
            polygon = np.column_stack([
                centre[0] + radii * np.cos(angles), centre[1] + radii * np.sin(angles)])
            inside = Path(polygon).contains_points(points).reshape(size, size)
            green = np.clip(np.array([0.15, 0.45, 0.16]) + rng.uniform(-0.05, 0.05, 3), 0, 1)
            if rng.random() < 0.15:
                green = np.clip(green * np.array([1.35, 0.8, 0.55]), 0, 1)
            image[inside] = green
            for _ in range(3):
                highlight = centre + rng.uniform(-radius, radius, 2) * 0.5
                disc = (((grid_x - highlight[0]) ** 2 + (grid_y - highlight[1]) ** 2)
                        < (radius * 0.3) ** 2)
                image[disc & inside] = np.clip(green + 0.12, 0, 1)
            instance[inside] = instance_id  # later crown wins on overlap
        # Relabel so ids are contiguous over *visible* crowns.
        visible = [i for i in range(1, len(centres) + 1) if np.any(instance == i)]
        if len(visible) != len(centres):
            remapped = np.zeros_like(instance)
            for new_id, old_id in enumerate(visible, start=1):
                remapped[instance == old_id] = new_id
            instance = remapped
        frame = (image * 255).astype(np.uint8)
        frame = apply_sensor(
            frame, rng, kind="poisson" if rng.random() < 0.5 else "gaussian",
            blur_radius=rng.uniform(0.2, 0.8),
        )
        images.append(frame)
        instances.append(instance)
    return np.stack(images), np.stack(instances)


def tree_box(instance_map, instance_id):
    """Normalised (x0, y0, x1, y1) box of one tree instance."""
    rows, cols = np.nonzero(instance_map == instance_id)
    size = instance_map.shape[0]
    return (cols.min() / size, 1 - (rows.max() + 1) / size,
            (cols.max() + 1) / size, 1 - rows.min() / size)


def forest_targets(instance_map, anchors, grid):
    """Encode tree instances into (objectness, box, instance-id) targets."""
    objectness = np.zeros((grid, grid, len(anchors)), np.float32)
    box_target = np.zeros((grid, grid, len(anchors), 4), np.float32)
    instance_id = -np.ones((grid, grid, len(anchors)), np.int32)
    for current_id in range(1, int(instance_map.max()) + 1):
        x0, y0, x1, y1 = tree_box(instance_map, current_id)
        width, height = x1 - x0, y1 - y0
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        anchor_idx = int(np.argmax([shape_iou((width, height), anchor) for anchor in anchors]))
        col = int(np.clip(center_x * grid, 0, grid - 1))
        row = int(np.clip((1 - center_y) * grid, 0, grid - 1))
        objectness[row, col, anchor_idx] = 1
        instance_id[row, col, anchor_idx] = current_id
        box_target[row, col, anchor_idx] = [
            center_x * grid - col, (1 - center_y) * grid - row,
            np.log(width / anchors[anchor_idx][0]), np.log(height / anchors[anchor_idx][1])]
    return objectness, box_target, instance_id
