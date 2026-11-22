import numpy as np
import pytest

from mlpsyn.datasets import (
    CLASSIFICATION_CLASSES,
    blend_two_scenes,
    encode_targets,
    forest_targets,
    kmeans_anchors,
    make_classification_data,
    make_cloud_data,
    make_detection_data,
    make_forest_data,
    make_segmentation_data,
    shape_iou,
    split_classification,
    tree_box,
)


def test_blend_two_scenes_deterministic_and_shaped():
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    a = blend_two_scenes("urban", "forest", 48, rng_a)
    b = blend_two_scenes("urban", "forest", 48, rng_b)
    assert a.shape == (48, 48, 3)
    np.testing.assert_array_equal(a, b)


def test_classification_data_is_balanced_and_split():
    images, labels = make_classification_data(3, size=32, seed=0)
    assert images.shape[0] == len(labels) and images.shape[1:] == (3, 32, 32)
    assert images.dtype == np.float32 and 0.0 <= images.min() <= images.max() <= 1.0
    assert labels.dtype == np.int64
    counts = np.bincount(labels, minlength=len(CLASSIFICATION_CLASSES))
    assert counts.min() >= 3  # balanced up to the odd overshoot of one sample
    train, val, test, *_ = split_classification(images, labels, 1, 1)
    assert len(train) == len(labels) - 2 * len(CLASSIFICATION_CLASSES)
    assert len(val) == len(test) == len(CLASSIFICATION_CLASSES)


def test_segmentation_and_cloud_data_shapes():
    images, labels = make_segmentation_data(4, size=32, seed=0)
    assert images.shape == (4, 3, 32, 32) and labels.shape == (4, 32, 32)
    assert images.dtype == np.float32 and labels.dtype == np.int64
    images, masks = make_cloud_data(4, size=32, seed=0)
    assert images.shape == (4, 3, 32, 32) and masks.shape == (4, 1, 32, 32)
    assert set(np.unique(masks).tolist()) <= {0.0, 1.0}


def test_detection_data_returns_objects():
    images, objects = make_detection_data(4, size=64, seed=0)
    assert images.shape == (4, 3, 64, 64)
    assert len(objects) == 4
    for scene_objects in objects:
        for obj in scene_objects:
            x0, y0, x1, y1 = obj.bbox
            assert 0.0 <= x0 <= x1 <= 1.0 and 0.0 <= y0 <= y1 <= 1.0


def test_shape_iou_and_anchors():
    assert shape_iou((2.0, 2.0), (2.0, 2.0)) == pytest.approx(1.0)
    pytest.importorskip("sklearn")
    boxes = [(0.1, 0.1), (0.1, 0.12), (0.3, 0.3), (0.4, 0.35)]
    anchors = kmeans_anchors(boxes, 2, log_space=False)
    assert anchors.shape == (2, 2)


def test_encode_targets_skips_degenerate_boxes():
    from mlpsyn.scene import SceneObject

    objects = [SceneObject("building", (0.1, 0.1, 0.3, 0.3)),
               SceneObject("rock", (0.5, 0.5, 0.5, 0.5))]  # zero-size
    label_map = {"building": "building", "rock": "rock"}
    classes = ["building", "rock"]
    class_index = {"building": 0, "rock": 1}
    anchors = np.array([[0.2, 0.2]])
    obj_t, cls_t, box_t = encode_targets(objects, anchors, 8, classes, class_index, label_map)
    assert obj_t.sum() == 1  # the degenerate box is dropped
    assert cls_t.shape == (8, 8, 1, 2) and box_t.shape == (8, 8, 1, 4)


def test_forest_data_and_targets():
    images, instances = make_forest_data(3, size=48, seed=0)
    assert images.shape == (3, 48, 48, 3) and images.dtype == np.uint8
    assert instances.shape == (3, 48, 48) and instances.dtype == np.int32
    assert instances[0].max() >= 1
    box = tree_box(instances[0], 1)
    assert len(box) == 4
    x0, y0, x1, y1 = box
    assert 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0
    anchors = np.array([[0.1, 0.1], [0.15, 0.15]])
    obj_t, box_t, ids = forest_targets(instances[0], anchors, 12)
    assert obj_t.shape == (12, 12, 2) and box_t.shape == (12, 12, 2, 4)
    assert ids.max() == instances[0].max()
