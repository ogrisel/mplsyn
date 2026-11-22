import numpy as np
import pytest

from mplsyn.textures import MAX_SHAPES, generate_texture, save_texture


def test_output_shape_and_dtype():
    image = generate_texture(size=64, n_shapes=20, seed=0)
    assert image.shape == (64, 64, 3)
    assert image.dtype == np.uint8


def test_deterministic_given_seed():
    a = generate_texture(size=64, n_shapes=50, seed=42)
    b = generate_texture(size=64, n_shapes=50, seed=42)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ():
    a = generate_texture(size=64, n_shapes=50, seed=1)
    b = generate_texture(size=64, n_shapes=50, seed=2)
    assert not np.array_equal(a, b)


def test_shape_cap_enforced():
    with pytest.raises(ValueError, match="MAX_SHAPES"):
        generate_texture(size=64, n_shapes=MAX_SHAPES + 1, seed=0)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        generate_texture(size=0, n_shapes=1, seed=0)
    with pytest.raises(ValueError):
        generate_texture(size=64, n_shapes=-1, seed=0)


def test_save_texture(tmp_path):
    image = generate_texture(size=32, n_shapes=5, seed=0)
    path = save_texture(image, tmp_path / "texture.png")
    assert path.exists()
    assert path.stat().st_size > 0
