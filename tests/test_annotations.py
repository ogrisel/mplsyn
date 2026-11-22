import numpy as np
import pytest

from mplsyn.annotations import (
    category_index,
    coco_dataset,
    scene_to_coco,
    train_val_test_split,
    yolo_lines,
)
from mplsyn.scene import generate_scene


@pytest.fixture
def scenes():
    return [generate_scene("fields", size=96, seed=seed) for seed in range(3)]


def test_category_index_is_stable_and_one_based(scenes):
    categories = category_index(scenes)
    assert min(categories.values()) == 1
    assert list(categories.values()) == sorted(categories.values())
    assert category_index(scenes) == categories


def test_coco_boxes_are_pixel_top_left(scenes):
    scene = scenes[0]
    categories = category_index(scenes)
    image, annotations = scene_to_coco(scene, 96, categories)
    assert image["width"] == image["height"] == 96
    assert len(annotations) == len(scene.objects)
    for ann in annotations:
        x, y, width, height = ann["bbox"]
        assert 0 <= x <= 96 and 0 <= y <= 96
        assert width > 0 and height > 0
        assert x + width <= 96.5 and y + height <= 96.5
        assert ann["category_id"] in categories.values()


def test_full_coco_dataset(scenes):
    dataset = coco_dataset(scenes, 96)
    assert len(dataset["images"]) == len(scenes)
    assert len(dataset["categories"]) == len(category_index(scenes))
    assert dataset["annotations"]


def test_yolo_lines_are_normalised_top_left(scenes):
    scene = scenes[0]
    lines = yolo_lines(scene)
    assert len(lines) == len(scene.objects)
    for line in lines:
        cls, cx, cy, width, height = line.split()
        assert int(cls) >= 0
        assert 0.0 <= float(cx) <= 1.0
        assert 0.0 <= float(cy) <= 1.0
        assert 0.0 < float(width) <= 1.0
        assert 0.0 < float(height) <= 1.0


def test_train_val_test_split_is_deterministic_and_disjoint():
    split = train_val_test_split(20, seed=1)
    assert split == train_val_test_split(20, seed=1)
    combined = split.train + split.val + split.test
    assert sorted(combined) == list(range(20))
    assert len(split.train) == 16
    assert len(split.val) == 2


def test_train_val_test_split_rejects_bad_ratios():
    with pytest.raises(ValueError):
        train_val_test_split(10, ratios=(0.5, 0.2, 0.2))


def test_pixel_bbox_consistency():
    scene = generate_scene("urban", size=128, seed=0)
    categories = category_index([scene])
    _, annotations = scene_to_coco(scene, 128, categories)
    for obj, ann in zip(scene.objects, annotations, strict=True):
        assert np.allclose(ann["bbox"], list(_xywh(obj, 128)))


def _xywh(obj, size):
    x0, y0, x1, y1 = obj.pixel_bbox(size)
    return x0, y0, x1 - x0, y1 - y0