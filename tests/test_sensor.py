import numpy as np
import pytest

from mlpsyn.scene import generate_scene
from mlpsyn.sensor import (
    EFFECTS,
    MAX_GAUSSIAN_SIGMA,
    MIN_POISSON_SCALE,
    apply_sensor,
    atmosphere,
    clouds,
    gaussian_noise,
    poisson_noise,
)


@pytest.fixture
def image():
    return generate_scene("fields", size=96, seed=0).image


def test_effects_registry_keys():
    assert set(EFFECTS) == {
        "clean",
        "gaussian",
        "poisson",
        "blur",
        "haze",
        "clouds",
    }


@pytest.mark.parametrize("name", sorted(EFFECTS))
def test_each_effect_preserves_shape_and_dtype(name, image):
    out = EFFECTS[name](image, np.random.default_rng(0))
    assert out.shape == image.shape
    assert out.dtype == np.uint8


def test_effects_are_deterministic_given_seed(image):
    a = gaussian_noise(image, np.random.default_rng(3))
    b = gaussian_noise(image, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)
    c = gaussian_noise(image, np.random.default_rng(4))
    assert not np.array_equal(a, c)


def test_clouds_darken_and_brighten(image):
    out = clouds(image, np.random.default_rng(0), coverage=0.5)
    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, image)


def test_apply_sensor_pipeline(image):
    out = apply_sensor(image, np.random.default_rng(0), kind="poisson", blur_radius=0.8)
    assert out.shape == image.shape
    assert out.dtype == np.uint8


def test_apply_sensor_applies_atmosphere_before_noise(image):
    out = apply_sensor(
        image, np.random.default_rng(0), kind="gaussian", haze_strength=0.3, cloud_coverage=0.4
    )
    rng = np.random.default_rng(0)
    expected = gaussian_noise(atmosphere(image, rng, 0.3, 0.4), rng)
    np.testing.assert_array_equal(out, expected)


def test_noise_is_mild_and_bounded(image):
    for kind in ("gaussian", "poisson"):
        out = apply_sensor(image, np.random.default_rng(1), kind=kind)
        std = (out.astype(float) - image.astype(float)).std()
        assert 0.0 < std < MAX_GAUSSIAN_SIGMA + 3.0


def test_gaussian_and_poisson_have_comparable_strength(image):
    gauss = apply_sensor(image, np.random.default_rng(2), kind="gaussian")
    poisson = apply_sensor(image, np.random.default_rng(2), kind="poisson")
    g_std = (gauss.astype(float) - image.astype(float)).std()
    p_std = (poisson.astype(float) - image.astype(float)).std()
    assert 0.5 < p_std / g_std < 2.0


def test_noise_parameters_are_clamped(image):
    heavy_gauss = gaussian_noise(image, np.random.default_rng(0), sigma=1000.0)
    assert (heavy_gauss.astype(float) - image.astype(float)).std() <= MAX_GAUSSIAN_SIGMA + 1.0
    clamped = poisson_noise(image, np.random.default_rng(0), scale=1.0)
    at_min = poisson_noise(image, np.random.default_rng(0), scale=MIN_POISSON_SCALE)
    np.testing.assert_array_equal(clamped, at_min)
    assert (clamped.astype(float) - image.astype(float)).std() <= 16.0