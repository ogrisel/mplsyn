"""mlpsyn: synthetic image generation for computer-vision training."""

from mlpsyn.annotations import (
    category_index,
    coco_dataset,
    scene_to_coco,
    train_val_test_split,
    yolo_lines,
)
from mlpsyn.materials import MATERIALS, generate_material, material_textures
from mlpsyn.metrics import (
    diversity_report,
    label_histogram,
    objects_per_scene,
    spatial_coverage,
)
from mlpsyn.object import OBJECTS, draw_object
from mlpsyn.satellite import (
    TEXTURE_DESCRIPTIONS,
    TEXTURES,
    agricultural_fields,
    crop_rows,
    dense_forest,
    desert,
    generate_satellite_texture,
    grassland,
    road_network,
    snow_ice,
    sparse_woodland,
    urban_buildings,
    water_river,
)
from mlpsyn.scene import SCENES, Scene, SceneObject, generate_scene
from mlpsyn.segmentation import (
    CLASS_COLORS,
    MATERIAL_SEG_CLASS,
    SEGMENTATION_CLASSES,
    semantic_labels,
    semantic_mask,
)
from mlpsyn.sensor import EFFECTS, apply_sensor
from mlpsyn.textures import MAX_SHAPES, generate_texture

__all__ = [
    "EFFECTS",
    "MATERIALS",
    "MATERIAL_SEG_CLASS",
    "MAX_SHAPES",
    "OBJECTS",
    "SCENES",
    "SEGMENTATION_CLASSES",
    "TEXTURE_DESCRIPTIONS",
    "TEXTURES",
    "CLASS_COLORS",
    "Scene",
    "SceneObject",
    "agricultural_fields",
    "apply_sensor",
    "category_index",
    "coco_dataset",
    "crop_rows",
    "dense_forest",
    "desert",
    "diversity_report",
    "draw_object",
    "generate_material",
    "generate_satellite_texture",
    "generate_scene",
    "generate_texture",
    "grassland",
    "label_histogram",
    "material_textures",
    "objects_per_scene",
    "road_network",
    "scene_to_coco",
    "semantic_labels",
    "semantic_mask",
    "snow_ice",
    "sparse_woodland",
    "spatial_coverage",
    "train_val_test_split",
    "urban_buildings",
    "water_river",
    "yolo_lines",
]
__version__ = "0.1.0"
