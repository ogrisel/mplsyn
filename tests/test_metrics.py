from mlpsyn.metrics import (
    diversity_report,
    label_histogram,
    objects_per_scene,
    spatial_coverage,
)
from mlpsyn.scene import generate_scene


def _scenes():
    return [generate_scene(name, size=96, seed=0) for name in ("urban", "fields", "pasture")]


def test_label_histogram_counts_every_object():
    scenes = _scenes()
    histogram = label_histogram(scenes)
    assert sum(histogram.values()) == sum(len(scene.objects) for scene in scenes)
    assert list(histogram) == sorted(histogram)


def test_objects_per_scene_matches_scenes():
    scenes = _scenes()
    counts = objects_per_scene(scenes)
    assert counts == [len(scene.objects) for scene in scenes]
    assert all(count > 0 for count in counts)


def test_spatial_coverage_bounds():
    scene = generate_scene("fields", size=128, seed=1)
    coverage = spatial_coverage(scene, grid=4)
    assert 0.0 < coverage <= 1.0
    empty = generate_scene("fields", size=64, seed=1)
    empty.objects = []
    assert spatial_coverage(empty, grid=4) == 0.0


def test_diversity_report_shape():
    report = diversity_report(_scenes(), grid=4)
    assert report["n_scenes"] == 3
    assert report["objects_total"] > 0
    assert report["objects_min"] <= report["objects_max"]
    assert report["label_entropy"] > 0.0
    assert 0.0 < report["mean_spatial_coverage"] <= 1.0