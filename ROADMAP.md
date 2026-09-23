# Roadmap

`mlpsyn` generates synthetic images with detailed annotations to train
computer-vision models. It targets two visual domains that share a need for
large volumes of cheap, perfectly-labelled data:

- **Satellite / aerial imagery**: land cover, buildings, roads, fields, water.
- **Medical imagery**: tissue, lesions, organs, microscopy-like structures.

Images are rendered with matplotlib drawing primitives, which keeps the
dependency footprint small and the geometry fully controllable. Every scene is
described by a seed + spec, so a dataset can be regenerated deterministically.

## Design constraints

- **Image size**: 600x600 pixels by default (configurable, but the default is
  what the pipelines are tuned for).
- **Hard cap of 10 000 matplotlib shapes per image** (`MAX_SHAPES`) to bound the
  CPU/memory cost of rendering a single scene.
- Deterministic output from a seed.
- Annotations are produced alongside the raster, never inferred from it.

## Milestones

### 1. Textures from matplotlib primitives

Foundational layer: render abstract textures using only simple drawing
primitives, with no domain semantics yet.

- [x] `generate_texture(size=600, n_shapes=..., seed=...)` returning an RGB array.
- [x] Random placement/sizing/rotation/colour of `Rectangle`, `Circle`,
      `Polygon`, `Line2D` and `PathPatch` primitives.
- [x] Colour sampling from a configurable matplotlib colormap.
- [x] Enforce `MAX_SHAPES = 10_000`; raise on larger requests.
- [ ] Save helpers (PNG) and a tiling/border mode so textures can be stitched.
- [x] Tests covering determinism, bounds, dtype and the shape cap.

### 1b. Materials (low-level surface textures)

- [x] Material registry (`mlpsyn.materials`): grass, foliage, asphalt, concrete,
      soil, sand, water, crop, rock, snow, bark, roof, gravel, mud.
- [x] `fill_patch` reuses a material texture as a clipped fill for any shape.
- [x] Materials reused by objects (fills) and scene grounds.

### 2. Image-level annotations

- [ ] Scene descriptors (seed, spec) serialised to JSON next to each image.
- [x] Image-level labels derived from the generator (e.g. dominant class); scene classification notebook.
- [x] Train/val/test split helpers (`mlpsyn.annotations.train_val_test_split`).

### 3. Object-level annotations

- [x] Bounding boxes with class ids, emitted as the primitives are drawn
      (`mlpsyn.object` returns bboxes; `mlpsyn.scene` collects `SceneObject`s).
- [ ] Support for rotated boxes.
- [x] Export to COCO / YOLO formats (`mlpsyn.annotations`).

### 4. Pixel-level annotations

- [ ] Per-class semantic segmentation masks rendered in parallel with the RGB
      image (one canvas per class, composited).
- [ ] Instance masks for panoptic-style targets.
- [ ] Guarantee exact pixel alignment between image and masks.

### 5. Satellite domain

- [x] Land-cover texture generators across environments (forest, woodland,
      grassland, agriculture, crop rows, urban, roads, river, desert, snow).
- [x] Object generators assembled from textured shapes (trees, conifers, bushes,
      buildings, field plots, ponds, rocks, cars, trucks, silos, pools,
      hedgerows, boats, parking lots, plazas, orchards, greenhouses, roads,
      rivers).
- [x] Scene composition (`urban`, `forest`, `fields`, `coastal`, `mountain`,
      `arid`, `pasture`, `wilderness`) from background + objects.
- [x] Boundary / transition scenes between two regions (`forest_fields`,
      `urban_fringe`, e.g. woodland next to farmland or a city next to fields,
      optionally separated by a river).
- [x] Sensor and atmospheric post-processing (`mlpsyn.sensor`): Gaussian/Poisson
      noise, optical blur, haze and cloud cover with ground shadows.
- [x] Scene-diversity sampling notebook: 25 random scene types with varying
      post-processing and cloud coverage (at least 70 % clear frames).
- [ ] Seasonal / illumination variation (snow cover, day/night, sun angle).
- [ ] Land-cover segmentation classes.

### 6. Medical domain

- [ ] Scene grammar: tissue background, vessels, lesions, cells.
- [ ] Modality-inspired palettes (grayscale ultrasound, H&E-like stains).
- [ ] Lesion detection boxes and tissue segmentation masks.

### 7. Dataset tooling

- [ ] CLI to generate a dataset from a config file.
- [ ] Parallel/batched generation with reproducible seeding per sample.
- [ ] Performance profiling of the renderer (shapes/second, peak memory).

### 8. Packaging & docs

- [ ] User guide with rendered examples per domain.
- [ ] API reference.
- [ ] Benchmark gallery comparing domains and annotation overlays.

### 9. Visual QA and diversity

Rendering quality is driven by an explicit visual-inspection loop.

- [x] `VISUAL_QA.md`: consolidated findings, severity-tagged defects and a
      prioritised backlog, refreshed every round.
- [x] `JOURNAL.md`: timestamped log of each iteration.
- [x] Example notebooks render **5 seeds per type** (materials, objects,
      scenes) on grids (5 columns, per AGENTS); each grid must stay below 5 MB
      (checked in the notebooks via `check_figure_size`) and each cell must run
      in under 10 s.
- [x] `notebooks/04_scene_diversity.ipynb`: 25 random scene types with mixed
      sensor noise, haze and variable cloud cover (>= 70 % clear) to assess
      composition diversity across seeds.
- [x] Automatic diversity metrics (`mlpsyn.metrics`: label histogram, object
      counts, spatial coverage, entropy) to complement the human inspection.
- [ ] Defect tracking for the ever-growing scene vocabulary.

## Implementation plan — Round 6

Two user-requested workstreams. One item per iteration, each ending with
re-rendered notebooks and a refreshed `VISUAL_QA.md` / `JOURNAL.md`.

### A. Sensor / atmosphere pipeline

1. **R6-1 — order**: atmosphere (haze, cloud occlusion) must be applied
   *before* sensor noise and blur, i.e. the physical path is
   ground → haze → clouds → optics (blur) → detector (noise). Provide a single
   `apply_sensor(...)` that owns the whole order and accepts `haze_strength` and
   `cloud_coverage`, so callers cannot accidentally add noise first. Keep the
   individual `EFFECTS` entries for the per-effect demo grid. **Done**.
2. **R6-2 — consistent, bounded noise**: use one moderate noise magnitude for
   every image (fixed default Gaussian `sigma` and Poisson full-well `scale`,
   chosen so both give a comparable standard deviation), clamp user values to a
   hard cap, and stop the diversity notebook from randomly toggling noise on/off
   or using wildly different strengths. Noise must stay subtle at 1:1. **Done**.

### B. Agricultural layout realism

Current `fields_scene` is a lattice of axis-aligned rectangles each surrounded
by grass, with no road frontage. Replace it with a **field mosaic**:

3. **R6-3 — Voronoi field mosaic**: partition the frame into irregular convex
   cells (Voronoi of jittered sites, half-plane clipped). Cells *tile* the
   ground (no grass gaps between plots), so every plot has a natural,
   non-rectangular boundary. Assign each cell a use (crop / soil / fallow grass
   / woodland). **Done**.
4. **R6-4 — boundaries are real features**: draw shared cell edges as country
   roads (asphalt) and dirt tracks (gravel), connect them into a road network
   with access to every plot, and put hedgerows along a subset of the remaining
   edges. A river may follow a chain of edges or cut across; fields then reach
   the bank. Silos/woodland sit along the network. **Done**.
5. **R6-5 — reuse in boundary scenes**: `forest_fields` and `urban_fringe`
   place their crop parcels with the same mosaic/road-frontage logic instead of
   free-floating rectangles. **Done**.
6. **R6-6 — docs/QA**: update notebooks, `VISUAL_QA.md` and `JOURNAL.md`; add
   regression tests (pipeline order, bounded noise, mosaic tiling, field
   frontage). **Done**.
7. **R6-7 — speed**: small-material cache (`_material_cached`), fixed axes
   autoscale, and 5-seed coverage grids keep every notebook cell under 10 s.
   **Done**.

## Implementation plan — Round 7

User-requested realism items. One item per iteration, each ending with
re-rendered notebooks and refreshed `VISUAL_QA.md` / `JOURNAL.md`.

### A. Urban network correctness and amenities

1. **R7-1 — no roads to nowhere; no buildings on roads**: build the urban
   street network as an explicit connected graph (grid + arterials) and check
   it: every road segment must join the network or leave the frame, and no
   dangling interior stubs. Track road corridors (centreline + width) and reject
   any building footprint that overlaps a corridor (with a small margin).
   **Done** — corridors are tracked and buildings are sized to the local road
   clearance; grid + arterial all span the frame.
2. **R7-2 — parking lots, plazas and parks**: add dedicated open-space objects
   and place them in urban blocks — parking lots (asphalt pad + row of stalls +
   a few cars), plazas (paved square with a fountain/planting), and more parks
   (grass + trees). These fill the empty concrete blocks and diversify land use.
   **Done** — `draw_parking_lot` / `draw_plaza`; every urban seed now has at
   least one parking or plaza plus 2–4 park blocks.

### B. Wild and pasture regions

3. **R7-3 — dense forest / wild / pasture scene(s)**: add scene types that
   showcase wilderness: a dense-forest scene (continuous canopy, no roads) and
   a pasture / meadow scene (open grassland with grazing clearings, scattered
   trees, hedgerows, a farm track). Ensure the vocabulary can naturally appear
   in satellite imagery. **Done** — `wilderness` and `pasture` added to
   `SCENES` (10 scene types).

### C. Elevation-driven generation

4. **R7-4 — shared elevation map**: introduce a per-seed elevation field
   (`_elevation_field`, already used by `mountain`) as a first-class scene input.
   Derive a hillshade overlay, slope and flow (downhill) direction from it.
   **Done** — `_terrain(rng) -> (height, slope, hillshade)` with a fixed
   upper-left light; `_relief_overlay` draws it.
5. **R7-5 — layout consistent with elevation**: place water in local minima /
   along valley flow lines, forests on wetter/upper slopes, fields and pasture
   on flatter ground, and route roads to follow contours / gentle grades.
   Reuse the elevation map across the terrain scenes (mountain, forest, fields,
   forest_fields, pasture, wild). **Partly done** — water follows low ground
   (`_low_valley_path`), mountain snow/rocks/water use height/slope, fields pick
   crop/woodland/fallow from height/slope, and forest/wilderness/pasture/fields
   get a relief overlay. Contour-following roads remain open.
6. **R7-6 — contour-following / gentle-grade roads**: route roads and tracks
   along gentle ground. **Done** — the field/boundary road spanning tree is a
   minimum spanning tree weighted by boundary height difference, and
   forest/pasture tracks use `_gentle_path` (lowest-slope candidate meander).

Each item is one code change plus a re-render and inspection; tests and docs are
updated as the round progresses.

## Implementation plan — Round 8

User-requested: fix roads that lead nowhere, add segmentation/cloud mask
notebook sections, and add three small PyTorch training notebooks (MPS).

1. **R8-1 — no dead-end roads**: `_close_network` adds an unused incident edge
   to every degree-1 cell of the field/boundary road spanning tree, so rural
   roads no longer stop inside a field. **Done**.
2. **R8-2 — pixel semantic segmentation**: `mlpsyn.segmentation` re-renders a
   scene with flat per-material class colours (mask mode in `mlpsyn._mode`),
   maps pixels to 13 classes (0 = unlabeled) and exposes `semantic_mask`.
   **Done**.
3. **R8-3 — cloud occlusion mask**: `mlpsyn.sensor.cloud_alpha` and
   `clouds(..., return_mask=True)` expose the soft occlusion field used by the
   atmosphere. **Done**.
4. **R8-4 — notebook 04 sections**: four rows per image (original, +boxes,
   semantic labels, cloud mask) for five varied scenes. **Done**.
5. **R8-5 — three training notebooks** (`05_train_object_detection`,
   `06_train_semantic_segmentation`, `07_train_cloud_occlusion`): small
   from-scratch PyTorch models on the **MPS** device, each under 10 minutes,
   with task metrics (precision/recall/F1/mIoU, pixel accuracy/mIoU, IoU/Dice).
   **Done**.
   - Note: unlike the visual-demo notebooks, the training notebooks contain
     cells that legitimately run for a few minutes (dataset generation and
     training); the <10 s per-cell rule is not applied to them.
   - Torch is not a pixi dependency (no cp314 wheel matching pixi's macOS tag);
     install it with `pixi run -e dev install-torch`. MPS needs
     `KMP_DUPLICATE_LIB_OK=1` and `OMP_NUM_THREADS=1`, which the notebooks set.

## Implementation plan — Round 9

User-requested classification, multi-class detection and a segmentation legend.

1. **R9-1 — scene-classification notebook** (`08_train_scene_classification`):
   7 classes (arid, coastal, cropland, forest, mountain, pasture, urban) derived
   from the scene vocabulary, one label per image. The ambiguous boundary scenes
   `forest_fields` (-> cropland) and `urban_fringe` (-> urban) are included and
   labelled by the dominant area. Balanced sampling; small CNN; accuracy and
   confusion matrix. **Done** (val accuracy 0.93).
2. **R9-2 — multi-class detection + mAP@0.5**: notebook 05 reworked to 6
   detection classes (building, field, vegetation, rock, water, boat) with a
   per-class centre heatmap; evaluation now computes the precision-recall curve
   per class and **mAP@0.5 IoU** (all-point interpolation), plus a per-class
   legend on the prediction figure. **Done** (mAP@0.5 = 0.35).
3. **R9-3 — segmentation colour legend**: notebook 06 shows the class colour
   swatches (0 = unlabeled). Seeded training for reproducibility; mIoU 0.64.
   **Done**.

## Implementation plan — Round 10

Documentation of the learning objectives in the training notebooks.

1. **R10-1 — spaces, losses and metrics**: each training notebook (05 detection,
   06 segmentation, 07 cloud, 08 classification) now states the prediction and
   target spaces, gives the mathematical definition of its loss and metrics, and
   computes loss/metric values on a few hand-built prediction/ground-truth pairs
   **before any model is trained** (verifying the formulas against PyTorch).
   **Done**.

## Implementation plan — Round 11

Code readability in the training notebooks.

1. **R11-1 — explicit variable names**: renamed cryptic names in notebooks
   05–08 (`cx`→`center_x`, `ci`→`class_idx`, `ix/iy`→`cell_x/cell_y`,
   `gx/gy`→`grid_x/grid_y`, `X/Y`→`images/labels`, `tp/fp`→
   `true_positives/false_positives`, ...). Short names are kept only where
   explicit ones would make dense arithmetic unreadable (e.g. the `dx/dy`
   Gaussian loop offsets), each with an explanatory comment above its
   definition. **Done**.

## Implementation plan — Round 12

Rework object detection and add tree instance segmentation (notebook 05).

1. **R12-1 — anchors per cell (YOLO style)**: 16x16 grid, **5 anchors/cell**
   chosen by k-means on training box shapes; per-slot objectness + class + box
   `(tx,ty,tw,th)` targets; decode with sigmoid/exp + NMS; mAP@0.5 per class.
   **Done** (mAP@0.5 = 0.26; overfits after ~30 epochs).
2. **R12-2 — instance segmentation of overlapping tree canopies**: self-contained
   forest generator with exact instance masks (soil material + irregular green
   crowns, later crown wins on overlap); two-headed U-Net (canopy + centre
   heatmap); decode by heatmap peaks + nearest-peak (watershed) labelling;
   metrics = count MAE, per-instance IoU, detection rate. **Done**
   (count MAE 0.00, instance IoU 0.98, detection 301/301).
3. **R12-3 — next**: add a 6th large anchor / more field samples to lift the
   low *field* AP (0.08), and check whether overlapping objects in the same cell
   are actually recovered by different anchors.

## Implementation plan — Round 13

1. **R13-1 — reorder training notebooks**: `08_train_scene_classification`
   moved into position 05 (it is the simplest training task / first learning
   step), shifting the others down. New order:
   `05` scene classification, `06` object detection (+ instance segmentation),
   `07` semantic segmentation, `08` cloud occlusion. README lists the full
   index. **Done**.

2. **R13-2 — object detection last**: `06_train_object_detection` -> `08_...`;
   `07` semantic segmentation -> `06`, `08` cloud occlusion -> `07`. New order:
   `05` classification, `06` semantic segmentation, `07` cloud occlusion,
   `08` object detection + instance segmentation. **Done**.

## Implementation plan — Round 14

Instance segmentation as a mask head on the detector (notebook 08).

1. **R14-1 — multi-head detector + per-anchor mask head**: replaced the
   two-headed U-Net + watershed with a YOLACT-style model — objectness head,
   box head, **mask head per anchor** (`P` coefficients per `(cell, anchor)`),
   and a **prototype branch with upscaling layers** (`16 -> 32 -> 64`) that
   renders full-resolution instance masks. Decode = score + NMS, then
   `sigmoid(coeff . prototypes)` cropped to the box and painted in score order.
   **Done** (count MAE 1.28, instance IoU 0.67, detection 0.83).
2. **R14-2 — next**: the masks are blobby and counts are off by one on crowded
   scenes; try ROI-cropped mask logits or a higher-resolution prototype branch.
   The training is also sensitive — keep an eye on mask collapse.

## Implementation plan — Round 15

Improve detection mAP and instance IoU (notebook 08).

1. **R15-1 — augmentation + more data + best checkpoint**: random horizontal /
   vertical flips with consistently flipped targets; detection data 240 -> 360
   and **6 anchors**; forest data 220 -> 360; 140 / 120 epochs; keep the best
   checkpoint by mAP / instance IoU. **Done** — detection mAP@0.5 0.26 -> **0.41**,
   instance IoU 0.67 -> **0.82**, detection 0.83 -> **0.99**, count MAE 1.12.
2. **R15-2 — bug fixed**: `flip_forest_batch` must flip the instance-id tensor
   too, otherwise the mask targets were mismatched on flipped batches and the
   mask head collapsed. Now flips images, targets, ids and masks together.
3. **R15-3 — next**: vegetation/rock AP is still low (few-pixel objects at
   64px); try a finer stride or an extra small anchor, and reduce runtime.

## Implementation plan — Round 16

Compare prototype vs per-anchor instance masks (notebook 08).

1. **R16-1 — per-anchor mask head**: added a second instance model that predicts
   one `16 x 16` mask per `(cell, anchor)` (GT mask cropped to its box as target;
   prediction resized to the predicted box), with no prototype basis. Same
   detector, data, flips and 120 epochs. **Done**.
2. **R16-2 — tradeoff table + figure**: side-by-side params, training time,
   inference ms/image, instance IoU, detection and count MAE, plus a qualitative
   figure (GT / prototype / per-anchor).
   Result: prototype 0.816 IoU / 0.99 det / 109.6 s; per-anchor 0.687 IoU /
   0.92 det / 69.9 s (faster, similar inference, lower count MAE).
3. **R16-3 — next**: raise per-anchor mask resolution or add ROIAlign; or feed
   the prototype coefficients a small spatial bias.

## Implementation plan — Round 17

Refocus the detection task on urban / coastal classes (notebook 08).

1. **R17-1 — six urban/coastal classes**: building, rock, tree, car, boat,
   parking. Scene mix mostly `urban` + `urban_fringe` + `coastal (beach)`.
   **Done**.
2. **R17-2 — photometric jitter + more data**: per-image brightness/contrast
   jitter on top of flips lifted mAP@0.5 from 0.41 to 0.45 at 64 px.
3. **R17-3 — 96 px detection**: the classes are tiny (cars ~3 px at 64 px), so
   the detection section runs at `96 x 96` / `24 x 24` grid while the forest
   section stays at 64 px (`FOREST_SIZE`/`FOREST_GRID`). mAP@0.5 0.45 -> **0.59**
   (building 0.96, parking 0.93, boat 0.67, tree 0.53, rock 0.37, car 0.09).
4. **R17-4 — next**: try 128 px for cars; the notebook now runs ~7 min.

## Implementation plan — Round 18

Documentation, coastal-city generation and field boundaries.

1. **R18-1 — docstrings**: short docstrings added to the notebook helper
   functions (make_data, encode_targets, decode_*, forest_*, crop_resize, ...)
   and to the key `mlpsyn` helpers. **Done**.
2. **R18-2 — coastal cities**: new `coastal_city` scene with a rubble
   **breakwater**, **piers** on pilings and a sheltered **marina** (finger docks
   + moored boats), plus a coast road, waterfront buildings, parking, cars and a
   park. Inspected in notebook 03 (5 seeds) and added to the detection scenes.
   **Done**.
3. **R18-3 — field boundaries**: `forest_fields` no longer insets its field
   mosaic (fields reach the top/bottom/right edges) and its woodland half is
   filled with canopy instead of bare soil. Inspected in notebook 03. **Done**.
4. **R18-4 — sklearn anchors**: `kmeans_anchors` now uses scikit-learn's
   `KMeans(n_init=10)` instead of a hand-rolled single-init k-means. **Done**.
5. **R18-5 — inline comments**: 1-5 line comments added to the important
   sections of notebooks 05-08 (training loops, decode, mask targets).
6. **R18-6 — qualitative grids**: detection and classification now show a
   `3 x 5` grid of validation images; segmentation/cloud/instance grids already
   have >= 3 rows x 5 columns.
7. **R18-7 — detection with harbours**: `coastal_city` added to the detection
   dataset (breakwater->rock, pier->building, marina->parking); mAP@0.5 0.59 ->
   **0.56** on the richer mix (car 0.12, boat 0.70, building 0.84).

## Implementation plan — Round 25 (prioritized)

Consolidated from the `VISUAL_QA.md` backlog review (stale pre-R17 entries
removed). Order is by teaching value, not by mAP.

1. **E1 — test split + protocol**: add a held-out test set and document
   model-selection-on-val / report-on-test in 05-08 (currently train/val only).
2. **E2 — failure galleries**: add per-model error galleries to 05 (misclassified
   scenes), 06 (worst-IoU regions) and 07 (missed clouds), matching notebook 08.
3. **E3 — results table + guidance**: one cross-notebook table of metrics plus a
   short "how to improve / where the ceiling is" note.
4. **D1 / D2 — dataset realism**: make the mountain scene less washed and the
   pasture/arid scenes less sparse (visual inspection in 03/04).
5. **M1 — resolution ablation**: show detection/instance metrics at 64/96/128 px
   as an explicit lesson on the resolution ceiling.
6. **M2 / D3 / L1 / L2**: dense-crown mask quality, cross-scene scale/density
   consistency, duplicate boxes, and prototype count stability.
