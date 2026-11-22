import shutil

import numpy as np
import pytest

from mplsyn.datasets import (
    CLASSIFICATION_CLASSES,
    blend_two_scenes,
    encode_targets,
    forest_targets,
    kmeans_anchors,
    load_classification_data,
    load_cloud_data,
    load_detection_data,
    load_forest_data,
    load_segmentation_data,
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
    from mplsyn.scene import SceneObject

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


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLSYN_DATA_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("MPLSYN_DATA_URL", raising=False)
    monkeypatch.setattr("mplsyn.datasets.DEFAULT_DATA_URL", None)  # stay offline
    return tmp_path / "cache"


def test_load_matches_make_and_is_cached(data_dir):
    first = load_segmentation_data(3, size=24, seed=1)
    assert len(list(data_dir.glob("segmentation-*.npz"))) == 1
    assert not list(data_dir.glob("*.part"))
    second = load_segmentation_data(3, size=24, seed=1)  # served from the cache
    for fresh, cached in zip(first, second, strict=True):
        np.testing.assert_array_equal(fresh, cached)
    for loaded, direct in zip(first, make_segmentation_data(3, size=24, seed=1), strict=True):
        np.testing.assert_array_equal(loaded, direct)
    load_segmentation_data(3, size=24, seed=2)  # other arguments -> other artifact
    assert len(list(data_dir.glob("segmentation-*.npz"))) == 2


def test_load_other_kinds_roundtrip(data_dir):
    images, labels = load_classification_data(2, size=24, seed=0)
    assert images.dtype == np.float32 and labels.dtype == np.int64
    images, masks = load_cloud_data(2, size=24, seed=0)
    assert masks.shape == (2, 1, 24, 24)
    images, instances = load_forest_data(2, size=24, seed=0)
    assert images.dtype == np.uint8 and instances.dtype == np.int32


def test_load_detection_preserves_objects(data_dir):
    direct_images, direct_objects = make_detection_data(3, size=48, seed=0)
    for _ in range(2):  # generated, then read back from the cache
        images, objects = load_detection_data(3, size=48, seed=0)
        np.testing.assert_array_equal(images, direct_images)
        assert [[(o.label, o.bbox) for o in objs] for objs in objects] == [
            [(o.label, o.bbox) for o in objs] for objs in direct_objects]


def test_load_downloads_pregenerated_artifact(data_dir, tmp_path, monkeypatch):
    # Publish an artifact on a "release" (a file:// URL), then use a fresh cache.
    load_cloud_data(2, size=24, seed=3)
    release = tmp_path / "release"
    release.mkdir()
    for artifact in data_dir.glob("*.npz"):
        shutil.copy(artifact, release / artifact.name)
    shutil.rmtree(data_dir)
    monkeypatch.setenv("MPLSYN_DATA_URL", release.as_uri())
    monkeypatch.setattr(
        "mplsyn.datasets.make_cloud_data",
        lambda *args, **kwargs: pytest.fail("should have been downloaded"))
    images, _ = load_cloud_data(2, size=24, seed=3)
    assert images.shape == (2, 3, 24, 24)
    assert len(list(data_dir.glob("cloud-*.npz"))) == 1


def test_load_falls_back_to_generation_when_download_fails(data_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MPLSYN_DATA_URL", (tmp_path / "missing").as_uri())
    with pytest.warns(UserWarning, match="Could not download"):
        images, _ = load_cloud_data(2, size=24, seed=4)
    assert images.shape == (2, 3, 24, 24)
    # A corrupt cached file is regenerated rather than raising.
    (artifact,) = data_dir.glob("cloud-*.npz")
    artifact.write_bytes(b"not an npz")
    with pytest.warns(UserWarning, match="Could not download"):
        images, _ = load_cloud_data(2, size=24, seed=4)
    assert images.shape == (2, 3, 24, 24)


def test_images_are_stored_as_uint8_and_converted_on_load(data_dir):
    images, labels = load_segmentation_data(3, size=24, seed=5)
    assert images.dtype == np.float32 and 0.0 <= images.min() <= images.max() <= 1.0
    raw, raw_labels = load_segmentation_data(3, size=24, seed=5, as_float32=False)
    assert raw.dtype == np.uint8 and raw.shape == (3, 3, 24, 24)
    np.testing.assert_array_equal(raw_labels, labels)
    np.testing.assert_array_equal(images, raw.astype(np.float32) / 255.0)
    with np.load(next(data_dir.glob("segmentation-*.npz"))) as archive:
        assert archive["images"].dtype == np.uint8  # what is actually on disk
    # Freshly generated and cached calls agree for either setting.
    fresh_raw, _ = load_cloud_data(2, size=24, seed=6, as_float32=False)
    cached_raw, _ = load_cloud_data(2, size=24, seed=6, as_float32=False)
    assert fresh_raw.dtype == cached_raw.dtype == np.uint8
    np.testing.assert_array_equal(fresh_raw, cached_raw)
    det_raw, _ = load_detection_data(2, size=48, seed=0, as_float32=False)
    assert det_raw.dtype == np.uint8
    det_float, _ = load_detection_data(2, size=48, seed=0)
    assert det_float.dtype == np.float32


def test_stage_datasets_writes_loadable_artifacts(tmp_path, monkeypatch, data_dir):
    from mplsyn import datasets, stage_datasets

    monkeypatch.setattr(stage_datasets, "NOTEBOOK_DATASETS", [
        (datasets.load_cloud_data, {"n_samples": 2, "size": 24}),
        (datasets.load_forest_data, {"n_samples": 2, "size": 24}),
    ])
    staging = tmp_path / "staging"
    created = stage_datasets.stage_datasets(staging)
    assert sorted(path.parent for path in created) == [staging.resolve()] * 2
    assert stage_datasets.stage_datasets(staging) == []  # second run: nothing to do
    # The staged folder works as a cache for the loaders.
    monkeypatch.setenv("MPLSYN_DATA_HOME", str(staging))
    monkeypatch.setattr(
        "mplsyn.datasets.make_cloud_data",
        lambda *args, **kwargs: pytest.fail("should have used the staged file"))
    images, _ = load_cloud_data(2, size=24)
    assert images.shape == (2, 3, 24, 24)


def test_missing_release_asset_warns(data_dir, monkeypatch):
    import io
    import urllib.error

    def not_found(url, timeout=None):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(b""))

    monkeypatch.setenv("MPLSYN_DATA_URL", "https://example.invalid/release")
    monkeypatch.setattr("mplsyn.datasets.urllib.request.urlopen", not_found)
    with pytest.warns(UserWarning, match="404"):
        images, _ = load_cloud_data(2, size=24, seed=8)
    assert images.shape == (2, 3, 24, 24)
