from collections import Counter

import numpy as np
import pytest

from mlpsyn.scene import (
    SCENES,
    _close_network,
    _distance_to_polyline,
    _gentle_path,
    _low_valley_path,
    _meander_points,
    _road_clearance,
    _sample,
    _shared_edges,
    _spanning_tree,
    _terrain,
    _voronoi_cells,
    generate_scene,
)


def test_scene_registry():
    assert set(SCENES) == {
        "urban",
        "forest",
        "fields",
        "coastal",
        "mountain",
        "arid",
        "pasture",
        "wilderness",
        "forest_fields",
        "urban_fringe",
        "coastal_city",
        "savanna",
        "woodland_pasture",
        "scrubland",
    }


@pytest.mark.parametrize("name", sorted(SCENES))
def test_each_scene_renders_with_objects(name):
    scene = generate_scene(name, size=160, seed=0)
    assert scene.image.shape == (160, 160, 3)
    assert scene.image.dtype == np.uint8
    assert scene.objects
    for obj in scene.objects:
        x0, y0, x1, y1 = obj.bbox
        assert 0.0 <= x0 <= x1 <= 1.0
        assert 0.0 <= y0 <= y1 <= 1.0
        assert obj.label


@pytest.mark.parametrize("name", sorted(SCENES))
def test_each_scene_is_deterministic(name):
    a = generate_scene(name, size=128, seed=11)
    b = generate_scene(name, size=128, seed=11)
    np.testing.assert_array_equal(a.image, b.image)
    assert a.objects == b.objects


def test_scene_labels_and_pixel_bbox():
    scene = generate_scene("urban", size=100, seed=0)
    assert set(scene.labels()) <= {
        "building",
        "tree",
        "car",
        "truck",
        "pool",
        "parking",
        "plaza",
        "bush",
        "conifer",
        "crossing",
        "pavement",
        "planter",
    }
    obj = scene.objects[0]
    x0, y0, x1, y1 = obj.pixel_bbox(100)
    assert 0 <= x0 <= x1 <= 100
    assert 0 <= y0 <= y1 <= 100


def test_pixel_bbox_uses_top_left_origin():
    from mlpsyn.scene import SceneObject

    obj = SceneObject("x", (0.1, 0.2, 0.3, 0.4))
    assert obj.pixel_bbox(100) == (10, 60, 30, 80)


def test_unknown_scene_raises():
    with pytest.raises(KeyError, match="unknown scene"):
        generate_scene("underwater")


def test_meander_points_keep_endpoints():
    points = _meander_points(
        np.random.default_rng(0), (0.1, 0.2), (0.9, 0.8), n=10, meander=0.1
    )
    assert points.shape == (10, 2)
    np.testing.assert_allclose(points[0], [0.1, 0.2])
    np.testing.assert_allclose(points[-1], [0.9, 0.8])


def test_fields_mosaic_tiles_the_frame():
    scene = generate_scene("fields", size=220, seed=0)
    fields = [obj.bbox for obj in scene.objects if obj.label == "field"]
    assert len(fields) >= 6
    x0 = min(field[0] for field in fields)
    y0 = min(field[1] for field in fields)
    x1 = max(field[2] for field in fields)
    y1 = max(field[3] for field in fields)
    assert x0 <= 0.02 and y0 <= 0.02
    assert x1 >= 0.98 and y1 >= 0.98


def test_voronoi_cells_tile_the_unit_square():
    rng = np.random.default_rng(0)
    sites = [rng.uniform(0.1, 0.9, size=2) for _ in range(9)]
    cells = _voronoi_cells(sites)
    assert len(cells) == 9
    total_area = 0.0
    for poly in cells:
        x = poly[:, 0]
        y = poly[:, 1]
        total_area += 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    assert abs(total_area - 1.0) < 1e-6


def test_spanning_tree_connects_every_cell():
    rng = np.random.default_rng(1)
    sites = [rng.uniform(0.1, 0.9, size=2) for _ in range(10)]
    cells = _voronoi_cells(sites)
    edges = _shared_edges(cells, sites)
    tree = _spanning_tree(len(sites), edges)
    assert len(tree) == len(sites) - 1
    # every cell has at least one road edge (frontage)
    touched = {index for edge in tree for index in edge}
    assert touched == set(range(len(sites)))


def test_close_network_removes_dead_ends():
    rng = np.random.default_rng(3)
    sites = [rng.uniform(0.1, 0.9, size=2) for _ in range(9)]
    cells = _voronoi_cells(sites)
    edges = _shared_edges(cells, sites)
    tree = _spanning_tree(len(sites), edges)
    closed = _close_network(tree, edges)
    degree = Counter()
    for i, j in closed:
        degree[i] += 1
        degree[j] += 1
    available = Counter()
    for i, j in edges:
        available[i] += 1
        available[j] += 1
    # no remaining dead-end still has an unused incident edge
    for node, node_degree in degree.items():
        if node_degree == 1:
            assert available[node] == 1


def test_weighted_spanning_tree_and_terrain_helpers():
    rng = np.random.default_rng(2)
    sites = [rng.uniform(0.1, 0.9, size=2) for _ in range(8)]
    cells = _voronoi_cells(sites)
    edges = _shared_edges(cells, sites)
    terrain = _terrain(rng)
    height, slope, hillshade = terrain
    assert height.shape == slope.shape == hillshade.shape[:2]
    assert height.min() >= 0.0 and height.max() <= 1.0

    def grade(key):
        p0, p1 = edges[key]
        return abs(_sample(height, p0[0], p0[1]) - _sample(height, p1[0], p1[1]))

    weighted = _spanning_tree(len(sites), edges, weight=grade)
    assert len(weighted) == len(sites) - 1
    valley = _low_valley_path(rng, height)
    gentle = _gentle_path(rng, slope)
    assert valley.shape[1] == 2 and gentle.shape[1] == 2


def test_distance_to_polyline_and_road_clearance():
    line = np.array([[0.0, 0.5], [1.0, 0.5]])
    assert _distance_to_polyline((0.5, 0.8), line) == pytest.approx(0.3)
    corridors = [(line, 0.1)]
    assert _road_clearance((0.5, 0.5), corridors) == pytest.approx(-0.05)
    assert _road_clearance((0.5, 0.8), corridors) == pytest.approx(0.25)
    assert _road_clearance((0.5, 0.8), []) == 1.0


def test_urban_has_buildings_and_amenities():
    for seed in range(5):
        labels = generate_scene("urban", size=220, seed=seed).labels()
        assert "building" in labels
        assert "parking" in labels or "plaza" in labels


def test_coastal_city_has_harbour_elements():
    labels = set()
    for seed in range(6):
        labels.update(generate_scene("coastal_city", size=220, seed=seed).labels())
    # Breakwaters, piers and marinas must all be reachable across seeds.
    assert {"breakwater", "pier", "marina"} <= labels
    assert "boat" in labels


def test_ambiguous_scenes_mix_the_right_cues():
    # Savanna: dry ground but heavy tree/scrub cover (arid vs forest).
    savanna = set(generate_scene("savanna", size=240, seed=1).labels())
    assert savanna & {"tree", "conifer", "bush"}
    # Wooded pasture: green meadow with extensive tree clumps (pasture vs forest).
    wooded = set(generate_scene("woodland_pasture", size=240, seed=1).labels())
    assert wooded & {"tree", "conifer", "bush"}
    # Scrubland: rocky and dry, but no trees (arid vs mountain).
    scrub = set(generate_scene("scrubland", size=240, seed=1).labels())
    assert "rock" in scrub
    assert not (scrub & {"tree", "conifer"})

