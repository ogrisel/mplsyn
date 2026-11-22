import numpy as np
import pytest
from matplotlib.figure import Figure

from mplsyn.object import OBJECTS, draw_object, draw_river, draw_road
from mplsyn.textures import ShapeBudget


def _axes():
    fig = Figure(figsize=(1, 1), dpi=64)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax


def test_registry_has_objects():
    assert set(OBJECTS) >= {
        "tree",
        "conifer",
        "bush",
        "building",
        "field_plot",
        "pond",
        "rock",
        "car",
    }


@pytest.mark.parametrize("name", sorted(OBJECTS))
def test_each_object_draws_with_valid_bbox(name):
    ax = _axes()
    budget = ShapeBudget(max_shapes=1000)
    bbox = OBJECTS[name](ax, np.random.default_rng(0), budget, 0.5, 0.5, 0.3)
    x0, y0, x1, y1 = bbox
    assert 0.0 <= x0 <= x1 <= 1.0
    assert 0.0 <= y0 <= y1 <= 1.0
    assert budget.count > 1


def test_object_deterministic():
    a = _axes()
    b = _axes()
    bbox_a = OBJECTS["tree"](a, np.random.default_rng(3), ShapeBudget(), 0.5, 0.5, 0.3)
    bbox_b = OBJECTS["tree"](b, np.random.default_rng(3), ShapeBudget(), 0.5, 0.5, 0.3)
    assert bbox_a == bbox_b


def test_draw_object_dispatch_and_error():
    ax = _axes()
    bbox = draw_object("building", ax, np.random.default_rng(0), ShapeBudget(), 0.5, 0.5, 0.3)
    assert len(bbox) == 4
    with pytest.raises(KeyError, match="unknown object"):
        draw_object("spaceship", ax, np.random.default_rng(0), ShapeBudget(), 0.5, 0.5, 0.3)


def test_draw_road_adds_shapes():
    ax = _axes()
    budget = ShapeBudget(max_shapes=1000)
    draw_road(ax, np.random.default_rng(0), budget, (0.1, 0.1), (0.9, 0.9), width=0.05)
    assert budget.count > 0


def test_draw_road_meander_and_polyline_return_bbox():
    ax = _axes()
    budget = ShapeBudget(max_shapes=2000)
    bbox = draw_road(
        ax, np.random.default_rng(1), budget, (0.1, 0.2), (0.9, 0.6), width=0.04, meander=0.1
    )
    assert bbox is not None
    x0, y0, x1, y1 = bbox
    assert 0.0 <= x0 <= x1 <= 1.0
    assert 0.0 <= y0 <= y1 <= 1.0

    points = np.array([(0.1, 0.1), (0.4, 0.5), (0.7, 0.3), (0.9, 0.8)])
    bbox = draw_road(
        ax, np.random.default_rng(2), budget, points=points, width=0.03, dashes=False
    )
    assert bbox is not None
    assert budget.count > 0


def test_draw_river_adds_shapes_with_valid_bbox():
    ax = _axes()
    budget = ShapeBudget(max_shapes=2000)
    bbox = draw_river(
        ax, np.random.default_rng(3), budget, (0.05, 0.3), (0.95, 0.6), width=0.07
    )
    assert bbox is not None
    x0, y0, x1, y1 = bbox
    assert 0.0 <= x0 <= x1 <= 1.0
    assert 0.0 <= y0 <= y1 <= 1.0
    assert budget.count > 1


def test_car_artists_stay_within_bbox():
    ax = _axes()
    bbox = OBJECTS["car"](
        ax, np.random.default_rng(0), ShapeBudget(), 0.5, 0.5, 0.3, rotation=37.0
    )
    x0, y0, x1, y1 = bbox
    for line in ax.lines:
        xs = np.asarray(line.get_xdata(), dtype=float)
        ys = np.asarray(line.get_ydata(), dtype=float)
        assert xs.min() >= x0 - 1e-6
        assert xs.max() <= x1 + 1e-6
        assert ys.min() >= y0 - 1e-6
        assert ys.max() <= y1 + 1e-6
