"""Internal render-mode flag shared by the drawing modules.

When semantic segmentation masks are requested, scenes are re-rendered with flat
class colours instead of material textures. The drawing helpers consult this flag
to skip effects that would blend colours (relief overlays, low-alpha fills) or
add fine detail (crop rows) that would corrupt the label map.
"""

from __future__ import annotations

MASK_MODE = {"on": False}


def mask_mode() -> bool:
    """Return True while scenes are being rendered in flat mask colours."""
    return MASK_MODE["on"]