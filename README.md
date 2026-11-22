# mplsyn

Synthetic image generation for computer-vision training, built on matplotlib.

`mplsyn` renders 600x600 images with fully controlled geometry and produces
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

`mplsyn` builds images in four layers, each reusing the one below:

1. **materials** (`mplsyn.materials`) — low-level surface textures: grainy
   asphalt, grass, foliage, soil, water, bark, roof;
2. **backgrounds** (`mplsyn.satellite`, `mplsyn.textures`) — land-cover patterns
   built on materials;
3. **objects** (`mplsyn.object`) — geometric objects filled with materials
   (tree, building, field plot, car, ...);
4. **scenes** (`mplsyn.scene`) — a material ground populated with
   material-filled objects, plus object-level bounding-box annotations;
5. **datasets** (`mplsyn.datasets`) — numpy-only, reusable dataset builders used
   by the modeling notebooks (classification mixtures, segmentation / cloud
   pairs, detection images + objects, forest instances); anchors use
   scikit-learn. Keeping dataset generation here means the notebooks train, not
   preprocess, and `import mplsyn` stays free of PyTorch.

Notebooks are split into two folders:

**[`notebooks/dataset_qa/`](notebooks/dataset_qa/)** — building and inspecting the
synthetic dataset (one notebook per level):

- [`01_texture_generation.ipynb`](notebooks/dataset_qa/01_texture_generation.ipynb) — materials
- [`02_object_generation.ipynb`](notebooks/dataset_qa/02_object_generation.ipynb) — objects
- [`03_scene_generation.ipynb`](notebooks/dataset_qa/03_scene_generation.ipynb) — scenes
- [`04_scene_diversity.ipynb`](notebooks/dataset_qa/04_scene_diversity.ipynb) — scene sampling + annotations

**[`notebooks/modeling/`](notebooks/modeling/)** — training and evaluating models
(PyTorch; runs on CUDA / MPS / XPU / CPU):

- [`05_train_scene_classification.ipynb`](notebooks/modeling/05_train_scene_classification.ipynb) — image-level scene labels
- [`06_train_semantic_segmentation.ipynb`](notebooks/modeling/06_train_semantic_segmentation.ipynb) — per-pixel class masks
- [`07_train_cloud_occlusion.ipynb`](notebooks/modeling/07_train_cloud_occlusion.ipynb) — cloud occlusion masks
- [`08_train_object_detection.ipynb`](notebooks/modeling/08_train_object_detection.ipynb) — anchor-based object detection in urban scenes
- [`09_train_instance_segmentation.ipynb`](notebooks/modeling/09_train_instance_segmentation.ipynb) — tree counting and instance segmentation in forests (mask head per anchor)

What the **modeling** notebooks teach (objectives, exercises and prerequisites)
is collected in [`LEARNING_OBJECTIVES.md`](LEARNING_OBJECTIVES.md).

The modeling notebooks need PyTorch (plus `torchinfo` for the model summaries) and
scikit-learn, which are installed with
pip (no cp314 conda wheels) into the dev environment:

```bash
pixi run -e dev install-torch      # PyTorch + torchinfo (auto device: CUDA/MPS/XPU/CPU)
pixi run -e dev install-sklearn    # scikit-learn (anchor k-means)
```

The modeling notebooks load their data with `mplsyn.datasets.load_*`, which read a
pregenerated `.npz` from the local cache (`~/.cache/mplsyn`, or `$MPLSYN_DATA_HOME`),
else download it from `$MPLSYN_DATA_URL` (default: the
[`v1` release](https://github.com/ogrisel/mplsyn/releases/tag/v1)), else generate it once and cache it.
To build the artifacts for every notebook into `./staging/` (git-ignored):

```bash
pixi run -e dev stage-datasets
MPLSYN_DATA_HOME=staging pixi run -e dev lab    # try them locally
```

Unlike the demo notebooks (each cell under 10 s), the training notebooks run for
a few minutes; notebook 08 exposes a `FAST` flag for a quick smoke run.

## Results at a glance

Held-out scores after the scene-generation / sensor changes. The **validation**
split selects the checkpoint; the **test** split is reported once (they are
close, so the models are not tuned to validation).

| notebook | task (64 px unless noted) | metric | validation | test |
|---|---|---|---|---|
| 05 | scene classification (7 classes, 50 % ambiguous mixtures) | accuracy | 0.79 | 0.74 |
| 06 | semantic segmentation (12 classes) | pixel acc / mIoU | 0.78 / 0.49 | 0.80 / 0.50 |
| 07 | cloud occlusion | IoU / Dice | 0.82 / 0.90 | 0.80 / 0.89 |
| 08 | object detection @96 px (6 classes) | mAP@0.5 | 0.53 | 0.57 |
| 09 | tree instance masks (per-anchor head) | instance IoU / count MAE (MAPE) | 0.72 / 0.9 (5 %) | 0.73 / 0.8 (6 %) |

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
from mplsyn.materials import MATERIALS, generate_material

generate_material("asphalt", size=600, seed=0)
MATERIALS.keys()
# dict_keys(['grass', 'foliage', 'asphalt', 'concrete', 'soil', 'sand',
#            'water', 'crop', 'rock', 'snow', 'bark', 'roof'])
```

Satellite land-cover backgrounds (built on materials):

```python
from mplsyn.satellite import TEXTURES, generate_satellite_texture

generate_satellite_texture("dense_forest", size=600, seed=0)
TEXTURES.keys()
# dict_keys(['dense_forest', 'sparse_woodland', 'grassland',
#            'agricultural_fields', 'crop_rows', 'urban_buildings',
#            'road_network', 'water_river', 'desert', 'snow_ice'])
```

Objects and scenes (with annotations):

```python
from mplsyn.scene import generate_scene

scene = generate_scene("urban", size=600, seed=0)
scene.image.shape       # (600, 600, 3)
scene.labels()          # ['building', 'tree', 'car', ...]
scene.objects[0].bbox   # normalised (x0, y0, x1, y1)
```

## License

Released under the **BSD 3-Clause License** — see [`LICENSE`](LICENSE).
