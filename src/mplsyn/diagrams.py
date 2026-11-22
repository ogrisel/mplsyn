"""Matplotlib diagrams of network architectures.

:func:`plot_unet` draws the compact U-Net of the segmentation notebook in the
style of the original U-Net paper: one box per feature map (width proportional to
the number of channels, height proportional to the spatial resolution), the
number of channels above each box, the spatial size below it, and arrows for the
operations between feature maps.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle

_FEATURE_COLOR = "#9ecae1"
_UPSAMPLED_COLOR = "#a1d99b"
_SKIP_COLOR = "#ffffff"
_LOGITS_COLOR = "#fdae6b"

# kind -> (colour, linestyle, legend label)
_ARROWS = {
    "conv": ("#08519c", "-", "double 3x3 convolution (stride 1)"),
    "down": ("#cb181d", "-", "double 3x3 convolution, stride 2 (halves height and width)"),
    "up": ("#238b45", "-", "upsampling x2 (nearest neighbour)"),
    "skip": ("#525252", "--", "skip connection (copy, then concatenation)"),
    "head": ("#e6550d", "-", "1x1 convolution (one logit per class)"),
}

_COLUMN_SPACING = 1.2


def plot_unet(base=20, in_channels=3, n_classes=13, size=64, ax=None):
    """Draw the U-Net architecture with the shape of every feature map.

    The diagram follows the ``SmallUNet`` of the segmentation notebook: four encoder
    blocks (the last three with stride 2), three decoder blocks that upsample,
    concatenate the skip connection and convolve, and a ``1x1`` convolution head.

    Parameters
    ----------
    base : int, default=20
        Number of channels of the first encoder block; the following blocks use
        ``2 * base``, ``4 * base`` and ``4 * base`` channels.
    in_channels : int, default=3
        Number of channels of the input image.
    n_classes : int, default=13
        Number of output channels (one logit map per class).
    size : int, default=64
        Height and width of the (square) input image, a multiple of 8.
    ax : matplotlib.axes.Axes, default=None
        Axes to draw on. A new wide figure is created when ``None``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes holding the diagram.
    """
    if size % 8:
        raise ValueError(f"size must be a multiple of 8, got {size}")
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 6))

    channels = [base, 2 * base, 4 * base, 4 * base]
    sizes = [size // 2**level for level in range(4)]

    def box_height(spatial_size):
        return max(2.4 * spatial_size / size, 0.35)

    def box_width(n_channels):
        return max(0.006 * n_channels, 0.08)

    # Vertical position of the centre of each resolution level (level 0 on top).
    centers = [0.0]
    for level in range(3):
        gap = (box_height(sizes[level]) + box_height(sizes[level + 1])) / 2 + 0.45
        centers.append(centers[-1] - gap)

    boxes = {}

    def add_box(name, column, level, parts, label):
        """Draw one feature map made of one or more (channels, colour, hatch) parts."""
        height = box_height(sizes[level])
        widths = [box_width(n_channels) for n_channels, _, _ in parts]
        left = column * _COLUMN_SPACING - sum(widths) / 2
        x = left
        for (_, color, hatch), width in zip(parts, widths, strict=True):
            ax.add_patch(
                Rectangle(
                    (x, centers[level] - height / 2),
                    width,
                    height,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.8,
                    hatch=hatch,
                )
            )
            x += width
        center_x = column * _COLUMN_SPACING
        ax.text(
            center_x,
            centers[level] + height / 2 + 0.08,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.text(
            center_x,
            centers[level] - height / 2 - 0.08,
            f"{sizes[level]}x{sizes[level]}",
            ha="center",
            va="top",
            fontsize=8,
            color="dimgray",
        )
        boxes[name] = {
            "left": left,
            "right": x,
            "y": centers[level],
            "height": height,
        }

    def add_arrow(source, target, kind, source_offset=0.0, target_offset=0.0):
        """Arrow from the right edge of ``source`` to the left edge of ``target``."""
        color, linestyle, _ = _ARROWS[kind]
        start = (
            boxes[source]["right"],
            boxes[source]["y"] + source_offset * boxes[source]["height"],
        )
        end = (
            boxes[target]["left"],
            boxes[target]["y"] + target_offset * boxes[target]["height"],
        )
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                color=color,
                linestyle=linestyle,
                linewidth=1.4,
                shrinkA=2,
                shrinkB=2,
            )
        )

    feature = (_FEATURE_COLOR, None)
    # Encoder, left to right and top to bottom.
    add_box("image", 0, 0, [(in_channels, *feature)], f"image\n{in_channels}")
    for index in range(4):
        name = f"enc{index + 1}"
        add_box(name, index + 1, index, [(channels[index], *feature)], f"{name}\n{channels[index]}")
    add_arrow("image", "enc1", "conv")
    for index in range(1, 4):
        add_arrow(f"enc{index}", f"enc{index + 1}", "down")

    # Decoder, bottom to top and left to right: concatenation box, then decoder output.
    upsampled_from = "enc4"
    upsampled_channels = channels[3]
    for step, level in enumerate((2, 1, 0)):
        skip_channels = channels[level]
        out_channels = {2: 2 * base, 1: base, 0: base}[level]
        concat_name, decoder_name = f"cat{level + 1}", f"dec{level + 1}"
        concat_column = 5 + 2 * step
        total = upsampled_channels + skip_channels
        add_box(
            concat_name,
            concat_column,
            level,
            [
                (upsampled_channels, _UPSAMPLED_COLOR, None),
                (skip_channels, _SKIP_COLOR, "///"),
            ],
            f"{total}\n({upsampled_channels}+{skip_channels})",
        )
        add_box(
            decoder_name,
            concat_column + 1,
            level,
            [(out_channels, *feature)],
            f"{decoder_name}\n{out_channels}",
        )
        add_arrow(upsampled_from, concat_name, "up", target_offset=-0.25)
        add_arrow(f"enc{level + 1}", concat_name, "skip", source_offset=0.25, target_offset=0.25)
        add_arrow(concat_name, decoder_name, "conv")
        upsampled_from, upsampled_channels = decoder_name, out_channels

    add_box("logits", 11, 0, [(n_classes, _LOGITS_COLOR, None)], f"logits\n{n_classes}")
    add_arrow("dec1", "logits", "head")

    handles = [
        Line2D([], [], color=color, linestyle=linestyle, linewidth=1.6, label=label)
        for color, linestyle, label in _ARROWS.values()
    ]
    handles += [
        Patch(facecolor=_FEATURE_COLOR, edgecolor="black", label="feature map"),
        Patch(facecolor=_UPSAMPLED_COLOR, edgecolor="black", label="upsampled map (decoder)"),
        Patch(facecolor=_SKIP_COLOR, edgecolor="black", hatch="///", label="skip connection map"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, ncol=2, frameon=True)

    ax.set_xlim(-0.8, 11 * _COLUMN_SPACING + 0.8)
    ax.set_ylim(centers[3] - 2.3, box_height(size) / 2 + 0.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"U-Net: number of channels above each box, height x width below "
        f"(input {in_channels}x{size}x{size})",
        fontsize=10,
    )
    return ax
