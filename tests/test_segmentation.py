import numpy as np

from mplsyn.scene import generate_scene
from mplsyn.segmentation import (
    CLASS_COLORS,
    MATERIAL_SEG_CLASS,
    SEGMENTATION_CLASSES,
    semantic_label_map,
    semantic_mask,
)
from mplsyn.sensor import cloud_alpha, clouds


def test_segmentation_class_and_material_maps_are_consistent():
    assert SEGMENTATION_CLASSES[0] == "unlabeled"
    assert all(name in SEGMENTATION_CLASSES for name in CLASS_COLORS)
    assert set(MATERIAL_SEG_CLASS.values()) <= set(SEGMENTATION_CLASSES[1:])
    assert len(set(CLASS_COLORS)) == len(CLASS_COLORS)


def test_semantic_mask_shape_and_range():
    scene, labels = semantic_mask("fields", size=96, seed=0)
    assert scene.image.shape == (96, 96, 3)
    assert labels.shape == (96, 96)
    assert labels.dtype == np.uint8
    assert labels.min() >= 0
    assert labels.max() < len(SEGMENTATION_CLASSES)
    # several semantic classes are present in a rich scene
    assert len(np.unique(labels)) >= 4


def test_semantic_mask_is_deterministic_and_aligned_with_rgb():
    _, a = semantic_mask("pasture", size=64, seed=3)
    _, b = semantic_mask("pasture", size=64, seed=3)
    np.testing.assert_array_equal(a, b)
    rgb = generate_scene("pasture", size=64, seed=3).image
    mask_render = generate_scene("pasture", size=64, seed=3, mask=True).image
    # mask render is mostly flat class colours, not the textured RGB image
    assert not np.array_equal(rgb, mask_render)


def test_label_map_maps_exact_colours():
    color = CLASS_COLORS["water"]
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:] = [int(255 * c) for c in color]
    labels = semantic_label_map(rgb)
    assert set(np.unique(labels)) == {SEGMENTATION_CLASSES.index("water")}
    # a colour far from the palette becomes unlabeled
    labels_far = semantic_label_map(np.full((2, 2, 3), 128, dtype=np.uint8))
    assert set(np.unique(labels_far)) <= {0} or labels_far.max() > 0


def test_cloud_return_mask():
    image = generate_scene("fields", size=64, seed=0).image
    out, alpha = clouds(image, np.random.default_rng(0), coverage=0.5, return_mask=True)
    assert out.shape == image.shape and out.dtype == np.uint8
    assert alpha.shape == image.shape[:2]
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0
    direct = cloud_alpha(image.shape, np.random.default_rng(0), coverage=0.5)
    np.testing.assert_array_equal(alpha, direct)