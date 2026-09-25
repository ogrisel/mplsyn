# mlpsyn

Synthetic image generation for computer-vision training, built on matplotlib.

`mlpsyn` renders 600x600 images with fully controlled geometry and produces
matching annotations at the image, object and pixel level. It is designed to
mimic satellite/aerial and medical imagery without needing large asset
collections.

See [ROADMAP.md](ROADMAP.md) for the current plan.

## Development

This project uses [pixi](https://pixi.sh) to manage the environment.

```bash
pixi install -e dev        # set up the dev environment
pixi run -e dev test       # run the test suite
pixi run -e dev lint       # run ruff
pixi run -e dev format     # auto-format with ruff
pixi run -e dev lab        # launch JupyterLab on the notebooks
```

## Examples

`mlpsyn` builds images in four layers, each reusing the one below:

1. **materials** (`mlpsyn.materials`) — low-level surface textures: grainy
   asphalt, grass, foliage, soil, water, bark, roof;
2. **backgrounds** (`mlpsyn.satellite`, `mlpsyn.textures`) — land-cover patterns
   built on materials;
3. **objects** (`mlpsyn.object`) — geometric objects filled with materials
   (tree, building, field plot, car, ...);
4. **scenes** (`mlpsyn.scene`) — a material ground populated with
   material-filled objects, plus object-level bounding-box annotations.

See [`notebooks/`](notebooks/) for one notebook per level:

- [`01_texture_generation.ipynb`](notebooks/01_texture_generation.ipynb) — materials
- [`02_object_generation.ipynb`](notebooks/02_object_generation.ipynb) — objects
- [`03_scene_generation.ipynb`](notebooks/03_scene_generation.ipynb) — scenes
- [`04_scene_diversity.ipynb`](notebooks/04_scene_diversity.ipynb) — scene sampling + annotations

Training notebooks (PyTorch, Apple MPS):

- [`05_train_scene_classification.ipynb`](notebooks/05_train_scene_classification.ipynb) — image-level scene labels
- [`06_train_semantic_segmentation.ipynb`](notebooks/06_train_semantic_segmentation.ipynb) — per-pixel class masks
- [`07_train_cloud_occlusion.ipynb`](notebooks/07_train_cloud_occlusion.ipynb) — cloud occlusion masks
- [`08_train_object_detection.ipynb`](notebooks/08_train_object_detection.ipynb) — anchor detection + tree instance segmentation

The training notebooks need PyTorch and scikit-learn, which are installed with
pip (no cp314 conda wheels) into the dev environment:

```bash
pixi run -e dev install-torch      # PyTorch (Apple MPS)
pixi run -e dev install-sklearn    # scikit-learn (anchor k-means)
```

Unlike the demo notebooks (each cell under 10 s), the training notebooks run for
a few minutes; notebook 08 exposes a `FAST` flag for a quick smoke run.

## Results at a glance

Held-out scores after the scene-generation / sensor changes. The **validation**
split selects the checkpoint; the **test** split is reported once (they are
close, so the models are not tuned to validation).

| notebook | task (64 px unless noted) | metric | validation | test |
|---|---|---|---|---|
| 05 | scene classification (7 classes) | accuracy | 1.00 | 1.00 |
| 06 | semantic segmentation (12 classes) | pixel acc / mIoU | 0.78 / 0.54 | 0.78 / 0.53 |
| 07 | cloud occlusion | IoU / Dice | 0.92 / 0.96 | 0.92 / 0.96 |
| 08 | object detection @96 px (6 classes) | mAP@0.5 | 0.46 | 0.50 |
| 08 | tree instance masks (per-anchor head) | instance IoU / count MAE (MAPE) | 0.57 / 4.3 (27 %) | 0.57 / 4.6 (32 %) |

All tasks run on **light sensor noise with ~20 % partial cloud cover**, so the
scores are deliberately below a clean-data benchmark. That is the lesson, spelled
out in each notebook's "Interpreting these numbers" section.

**How to improve, and where the ceiling is**

- **Resolution** is the main lever for small objects (a car is ~4 px even at
  96 px): a 128 px run is the natural next experiment.
- **Anchor shapes** matter as much as resolution: clustering the box shapes in
  **log space** gives small objects their own anchor and lifted car/rock/tree AP
  noticeably (see notebook 08).
- **Sensor quality / clouds bound the recoverable accuracy** — no architecture
  can recover pixels that were never measured.
- **Priors help**: temporal stacks, multi-scale features, or a dedicated
  small-object level; on the instance task a box-relative per-anchor mask head
  counts crowns accurately but merges the most heavily occluded ones.

## Quick start

Low-level materials:

```python
from mlpsyn.materials import MATERIALS, generate_material

generate_material("asphalt", size=600, seed=0)
MATERIALS.keys()
# dict_keys(['grass', 'foliage', 'asphalt', 'concrete', 'soil', 'sand',
#            'water', 'crop', 'rock', 'snow', 'bark', 'roof'])
```

Satellite land-cover backgrounds (built on materials):

```python
from mlpsyn.satellite import TEXTURES, generate_satellite_texture

generate_satellite_texture("dense_forest", size=600, seed=0)
TEXTURES.keys()
# dict_keys(['dense_forest', 'sparse_woodland', 'grassland',
#            'agricultural_fields', 'crop_rows', 'urban_buildings',
#            'road_network', 'water_river', 'desert', 'snow_ice'])
```

Objects and scenes (with annotations):

```python
from mlpsyn.scene import generate_scene

scene = generate_scene("urban", size=600, seed=0)
scene.image.shape       # (600, 600, 3)
scene.labels()          # ['building', 'tree', 'car', ...]
scene.objects[0].bbox   # normalised (x0, y0, x1, y1)
```
