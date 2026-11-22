import numpy as np
import pytest

from mplsyn.satellite import TEXTURES, generate_satellite_texture
from mplsyn.textures import MAX_SHAPES, ShapeBudget


def test_registry_has_ten_textures():
    assert len(TEXTURES) == 10


@pytest.mark.parametrize("name", sorted(TEXTURES))
def test_each_texture_renders_square(name):
    image = TEXTURES[name](size=128, seed=0)
    assert image.shape == (128, 128, 3)
    assert image.dtype == np.uint8


@pytest.mark.parametrize("name", sorted(TEXTURES))
def test_each_texture_is_deterministic(name):
    a = TEXTURES[name](size=96, seed=7)
    b = TEXTURES[name](size=96, seed=7)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", sorted(TEXTURES))
def test_each_texture_respects_shape_cap(name):
    image = TEXTURES[name](size=600, seed=0)
    assert image.shape == (600, 600, 3)


def test_generate_satellite_texture_dispatch():
    image = generate_satellite_texture("grassland", size=64, seed=0)
    np.testing.assert_array_equal(image, TEXTURES["grassland"](size=64, seed=0))


def test_unknown_texture_raises():
    with pytest.raises(KeyError, match="unknown texture"):
        generate_satellite_texture("not_a_texture")


def test_shape_budget_counts_and_caps():
    budget = ShapeBudget(max_shapes=2)
    assert budget.remaining == 2
    assert not budget.full
    budget.count = 2
    assert budget.full
    assert budget.remaining == 0


def test_shape_budget_rejects_invalid():
    with pytest.raises(ValueError):
        ShapeBudget(max_shapes=0)
    assert MAX_SHAPES == 10_000
