import pytest
from matplotlib.figure import Figure

from mplsyn.diagrams import plot_unet


def _texts(ax):
    return [text.get_text() for text in ax.texts]


def test_plot_unet_reports_the_feature_map_shapes():
    ax = Figure().subplots()
    assert plot_unet(base=20, in_channels=3, n_classes=13, size=64, ax=ax) is ax
    texts = " ".join(_texts(ax))
    # channel counts (including the concatenated skip connections) and spatial sizes
    for expected in (
        "image\n3",
        "enc4\n80",
        "160\n(80+80)",
        "80\n(40+40)",
        "40\n(20+20)",
        "logits\n13",
    ):
        assert expected in _texts(ax), expected
    for expected in ("64x64", "32x32", "16x16", "8x8"):
        assert expected in texts


def test_plot_unet_follows_the_base_width_and_image_size():
    ax = Figure().subplots()
    plot_unet(base=8, n_classes=5, size=32, ax=ax)
    texts = _texts(ax)
    assert "enc2\n16" in texts
    assert "64\n(32+32)" in texts and "32\n(16+16)" in texts
    assert "logits\n5" in texts and "4x4" in texts


def test_plot_unet_rejects_sizes_that_are_not_a_multiple_of_8():
    with pytest.raises(ValueError):
        plot_unet(size=60, ax=Figure().subplots())
