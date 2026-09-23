import numpy as np
import pytest
from matplotlib.figure import Figure

from mlpsyn.materials import (
    MATERIALS,
    fill_patch,
    generate_material,
    material_textures,
)
from mlpsyn.textures import ShapeBudget


def test_material_registry():
    assert len(MATERIALS) >= 12
    assert {"grass", "foliage", "asphalt", "water", "roof"} <= set(MATERIALS)


@pytest.mark.parametrize("name", sorted(MATERIALS))
def test_each_material_renders_square(name):
    image = MATERIALS[name](size=96, seed=0)
    assert image.shape == (96, 96, 3)
    assert image.dtype == np.uint8


@pytest.mark.parametrize("name", sorted(MATERIALS))
def test_each_material_is_deterministic(name):
    a = MATERIALS[name](size=64, seed=5)
    b = MATERIALS[name](size=64, seed=5)
    np.testing.assert_array_equal(a, b)


def test_generate_material_dispatch_and_error():
    image = generate_material("asphalt", size=64, seed=0)
    np.testing.assert_array_equal(image, MATERIALS["asphalt"](size=64, seed=0))
    with pytest.raises(KeyError, match="unknown material"):
        generate_material("lava")


def test_material_textures_subset():
    textures = material_textures(64, 0, ["grass", "water"])
    assert set(textures) == {"grass", "water"}
    assert textures["grass"].shape == (64, 64, 3)


def test_fill_patch_clips_material():
    from matplotlib import patches
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(1, 1), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_facecolor("#ff0000")
    budget = ShapeBudget()
    patch = patches.Rectangle((0.25, 0.25), 0.5, 0.5)
    texture = np.zeros((4, 4, 3), dtype=np.uint8)
    texture[..., 2] = 255
    fill_patch(ax, patch, texture, budget=budget)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    image = np.asarray(canvas.buffer_rgba())[..., :3]
    assert budget.count == 1
    assert tuple(image[50, 50]) == (0, 0, 255)
    assert tuple(image[10, 10]) != (0, 0, 255)
