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

## Implementation plan — Round 32 (educational value of the model notebooks)

Review of notebooks 05–08: they already state objectives, prediction/target
spaces, losses and metrics, verify the formulas on toy data, keep a held-out
test split and show failure galleries. The gaps are all about **reading and
interpreting** the results, so the round adds five teaching aids, one per
iteration.

1. **Edu-1 — training curves everywhere.** Notebook 05 already plots training
   loss and validation accuracy; plot the same (loss + the task metric vs
   epoch, with the best epoch marked) in 06, 07 and both 08 models, so learners
   can *see* the fit and the point of diminishing returns.
2. **Edu-2 — trivial baselines.** Put every headline score next to a naive
   baseline: majority-class accuracy (05), all-background mIoU (06),
   all-clear / all-cloud IoU (07) and a no-object detection baseline (08), plus
   the mean tree count for the instance task. A metric is only meaningful
   relative to what "doing nothing" scores.
3. **Edu-3 — operating point for clouds (07).** Sweep the probability threshold
   and plot IoU / Dice / precision / recall plus the precision-recall curve, to
   show that 0.5 is a convention and that the operating point trades recall for
   precision.
4. **Edu-4 — class prevalence vs per-class IoU (06).** Bar/annotate class
   frequency next to per-class IoU so the learner sees why rare classes drag
   mIoU down and why pixel accuracy hides them.
5. **Edu-5 — per-class precision-recall curves (08).** Plot the PR curve for
   each detection class and mark the AP, explaining what mAP averages and why a
   rare class's AP is noisy.

Each item is one code change plus a re-run/inspection of the affected notebook,
with the plots described in the notebook and the plan tracked here.

## Implementation plan — Round 33 (experiments and metric behaviour)

Round 32 taught how to *read* a score. This round teaches how to *run* an
experiment and how the metrics behave under the hood.

1. **Edu-6 — augmentation gallery (08).** Show one training image and its
   horizontal flip, vertical flip and brightness/contrast jitter with the boxes
   transformed, proving the labels stay valid. Augmentation is used but never
   shown; seeing it is what makes the "why" click.
2. **Edu-7 — seed variance (05).** Retrain the classifier with three seeds on a
   short schedule and report validation accuracy as mean ± std, so learners see
   that a single number hides run-to-run noise and judge differences against it.
3. **Edu-8 — class-weighted loss (06).** Train the same U-Net with a
   class-frequency-weighted cross-entropy and compare per-class IoU and mIoU to
   the unweighted run, connecting the prevalence plot to a concrete remedy.
4. **Edu-9 — detection metric behaviour (08).** Compute mAP at IoU 0.5 / 0.65 /
   0.75 (what @0.5 hides about box tightness) and sweep the score threshold,
   plotting precision / recall / count, to show the operating-point trade-off.
5. **Edu-10 — probability calibration (07).** Reliability diagram and expected
   calibration error for the cloud probabilities, then temperature scaling fit
   on validation, showing that sigmoid output is not a calibrated probability.

One item per iteration, each ending with the affected notebook re-run and the
new figures described in the notebook.

### Round 33 — results

- **Augmentation gallery (08):** original / horizontal flip / vertical flip /
  brightness-contrast of one training image with transformed boxes; labels stay
  valid by construction.
- **Seed variance (05):** validation accuracy over 3 seeds = [0.843, 0.971,
  0.700], **mean 0.838 ± 0.111** — a concrete reminder that single-run
  differences below ~0.1 are noise.
- **Class-weighted loss (06):** plain CE mIoU **0.465** vs class-weighted
  **0.564**; the rarest classes gain most (gravel 0.03 -> 0.15, wetland
  0.17 -> 0.29).
- **Detection metric behaviour (08):** mAP@0.50 **0.472**, @0.65 **0.247**,
  @0.75 **0.060** — @0.5 hides very loose localisation; the score-threshold
  sweep shows the precision/recall/count trade-off.
- **Calibration (07):** ECE **0.117 -> 0.006** with temperature scaling
  (T = 0.30), so the raw sigmoid is over/under-confident and a one-parameter fix
  recalibrates it without touching accuracy.

## Implementation plan — Round 34 (review-driven)

Fresh inspection of the notebook figures and a re-render of all scenes / objects.
Three concrete issues, plus remaining backlog.

1. **R34-1 — arid wadi reads as a road.** The dry riverbed draws a single gravel
   ribbon with a dark centre line (`#6f6349`, the road convention). Rework it
   into a **braided** dry channel: two or three narrower sandy/gravel threads,
   light gravel bars between them, pebble speckle and no centre line.
2. **R34-2 — greenhouse reads as white blobs.** `draw_greenhouse` draws a row of
   plain light ellipses. Replace with rectangular **glazed ranges**: translucent
   pale-green panels, a frame ridge grid, per-range tone variation and a soft
   shadow, so it reads as glasshouses from above.
3. **R34-3 — instance quality by density (08).** The instance metrics are single
   averages; report **count MAE / IoU / detection by regime** (sparse / medium /
   dense / avenue) plus an IoU histogram, so the dense-canopy failure mode is
   quantified rather than asserted.

Remaining backlog (not this round): mountain still washes toward flat grey; the
transition-scene field mosaic keeps a straight inner edge on some seeds; urban
roads are uniformly dark. These stay on the backlog for a later round.

### Round 34 — results

- **R34-1 wadi:** arid scenes now show a braided sandy dry channel (pale apron,
  two-three gravel threads, pebbles, no centre line) instead of a road-like
  ribbon.
- **R34-2 greenhouse:** reads as parallel gabled glazed ranges with frame ribs
  and a shadow, not white ellipses.
- **R34-3 instance by density (08):** count MAE **2.29 / 3.54 / 4.33** and mean
  IoU **0.596 / 0.592 / 0.554** for sparse / medium / dense — the failure mode is
  now quantified.
- Generator read-outs in 02 (conifer, greenhouse, parking) and 03 (arid wadi)
  refreshed to match the current renders.

## Implementation plan — Round 35 (clarity of the model notebooks)

The model notebooks cover the mechanics and the metrics; this round makes the
abstract encodings **visible** and the exercises **self-checkable**.

1. **Clar-1 — anchor-assignment figure (08).** The detection targets are
   `(cell, anchor)` slots but this is only shown numerically. Draw a few training
   images with the GT boxes, the grid cell each object falls in and the anchor
   index it matched, so the encoding becomes concrete.
2. **Clar-2 — box-relative mask target figure (08).** Show one forest instance:
   image, instance mask, its bounding box, and the **cropped + resized
   `MASK_SIZE` target** that the mask head actually learns, explaining the
   Mask R-CNN-style target.
3. **Clar-3 — key-terms box per notebook (05-08).** A short glossary of the
   jargon each notebook introduces (logits, softmax, CE; IoU/mIoU, ignore_index;
   Dice, BCE-with-logits; anchor, objectness, NMS, mAP, ECE), so a reader never
   has to leave the notebook to decode a term.
4. **Clar-4 — exercise self-checks (05-08).** Append a "what to expect / how to
   check" note to each exercise so learners can verify their own experiments
   instead of guessing whether they got it right.

Each item is one change plus a re-run of the affected notebook, with the new
figure/box described in the notebook.

### Round 35 — results

- **Clar-1 anchor figure (08):** three training images with GT boxes, the target
  grid cell and the matched anchor index (`aK`) drawn, plus the positive count.
- **Clar-2 mask-target figure (08):** image / instance map / one instance with
  its box / the `24x24` crop+resize target — the box-relative mask the head
  learns.
- **Clar-3 key terms:** a compact glossary cell added to 05, 06, 07 and 08.
- **Clar-4 self-checks:** a "what to expect" note appended to every exercise set
  (05-08) so learners can verify their experiments.

## Round 36 — validation loss and less noisy curves

Requested: plot validation loss alongside the validation metrics, enlarge the
training/validation/test sets to reduce metric oscillation, and adjust epochs so
runtimes do not grow much.

- **Validation loss** added to every training curve: 05/06/07 (single loss panel
  + metric panel) and 08 (detection and instance models, loss panels on val).
- **Larger splits**: 05 N_PER_CLASS 46->72 / val 10->16 / test 8->12; 06 and 07
  N_SAMPLES 240->360, val 48->80, test 40->60; 08 detection 360->480 (val
  64->80, test 48->64) and forest 320->400 (val 40->56, test 32->44).
- **Epochs reduced** to compensate: 05 40->30 (and lr 2e-3 -> 1e-3, which also
  removed a BatchNorm transient), 06 30->16, 07 20->12, 08 detection 150->100,
  forest 100->70.
- **Result**: 05-08 total runtimes 54 / 87 / 44 / 188 s (was ~38 / 83 / 39 /
  190 s); validation curves are much smoother (06 mIoU monotonic; detection
  mAP monotonic and train/val losses track). Test detection mAP rose to 0.50 and
  instance test IoU to 0.57.

## Round 37 — harder scene classification (ambiguous scenes)

Requested: make scene classification harder with ambiguous scenes.

- **New transitional scene types** in `mlpsyn.scene`: `savanna` (arid ground,
  nearly forest-dense trees), `woodland_pasture` (meadow gone half-woodland),
  `scrubland` (rocky and dry, no trees). Registered in `SCENES`, rendered in 03,
  sampled in 04, and used in the classification palette (savanna/scrubland ->
  arid, woodland_pasture -> pasture, plus `coastal_city` -> coastal).
- **Mixture dataset (05)**: half the images blend two scene types with a soft
  wavy boundary at a near-equal share. The rival class is drawn from a
  hand-picked **confusable** set, and light blur/noise removes fine-texture
  shortcuts.
- **Irreducible ambiguity**: a near-balanced mixture has no single correct label,
  so it is assigned to the target scene with probability 0.65 and to the rival
  otherwise, independent of area.
- **Result**: classification is no longer trivial - validation accuracy
  **0.74** (was 1.00) with an off-diagonal confusion matrix (coastal<->urban,
  mountain<->arid, forest<->pasture/cropland); the ceiling (~0.75) is explained
  in the notebook as irreducible.

## Round 38 — wadi colour, concrete/urban richness, more clouds

Three requested changes:

1. **Wadi closer to sand (arid)** — the apron now uses the sand base tone, the
   braided gravel threads are drawn at 50 % alpha (blended toward the sand) and
   the pebbles are sandy, so the dry bed no longer stands out as a grey ribbon.
2. **Concrete + urban richness** — the `concrete` material no longer draws
   large expansion-joint lines (they read as oversized cracks); it is now
   textured only at the aggregate scale. Three new small objects -
   `crossing` (zebra), `pavement` (tiled sidewalk apron) and `planter` (planted
   bed) - are drawn into `urban`, together with more trees and a higher chance
   of a second building per block, so blocks are no longer bare concrete.
3. **Semi-transparent clouds, more often** — `clouds`/`cloud_alpha` gained an
   `opacity` parameter (default **0.6**, previously capped at 0.9), so the ground
   stays visible through the cloud. Cloud frequency rose to **~40 %** in the
   segmentation/detection data (was 20 %) and to **48 %** in the diversity
   notebook (was 28 %).

**Result** (all notebooks re-run): classification 0.78/0.82; segmentation mIoU
0.49; cloud IoU 0.72/0.76 (harder with translucent clouds); detection
0.46/0.47; instance 0.57 IoU. 172 tests pass.

## Round 39 — vehicles, muted materials, semi-arid diversity, notebook quality

Requested batch:

- **Vehicles**: separate `_VEHICLE_COLORS` palette (neutrals + a few saturated
  colours) for cars and trucks, plus a new `bus` object placed in urban scenes;
  detection maps `bus -> car`.
- **Materials**: `sand`, `grass`, `foliage` and `crop` now pick a **per-seed,
  muted base tone** from small palettes, so backgrounds vary across seeds and are
  less saturated.
- **Arid diversity**: a per-scene `vegetation_level` drives arid scenes from bare
  desert to **semi-arid** (grass patches, many more bushes, scattered trees).
  Fixed a bug where a missing `"grass"` material made `_ground_patches` draw
  matplotlib's default **blue** polygon.
- **Notebooks**: `pick_device()` now selects cuda -> mps -> xpu -> cpu; a
  "Why BCE for a multi-class head?" section explains binary cross-entropy applied
  per class (vs softmax CE); every bounding-box figure carries a **colour legend**;
  a short **figure read-out is placed directly below each figure** (the end-of-
  notebook duplicates were removed).

**Result** (all notebooks re-run): 05 0.80/0.71; 06 mIoU 0.41 (muted tones lower
class separability); 07 IoU 0.82/0.85; 08 detection mAP 0.52/0.50, instance IoU
0.56. 173 tests pass.

## Round 40 — cloud opacity: opaque cores with a quarter semi-transparent

Requested: raise the median cloud opacity so most clouds have a **fully opaque
core**, while keeping ~25 % of cloudy scenes partially transparent.

- `cloud_alpha`/`clouds`/`atmosphere`/`apply_sensor` now take `opacity=None`; a
  per-scene value is sampled by `_sample_opacity`: **75 %** draw an opaque core
  (0.95-1.0), **25 %** stay semi-transparent (0.45-0.6). Measured over 400
  samples: ~72 % opaque-core, ~28 % partial.
- Notebooks 06/08 no longer force `cloud_opacity=0.55`, so the sensor samples
  the mix.

**Result** (03/04/06/07/08 re-run): 06 mIoU 0.50, 07 IoU 0.84/0.85, **detection
drops to 0.41/0.39** because opaque clouds hide the small objects entirely
(instance/forest unchanged, no clouds there).

## Round 41 — confusion matrices and class histograms

- **Segmentation (06)**: pixel-level confusion matrix (row-normalised) + log
  class histogram, exposing systematic confusions the scalar mIoU hides.
- **Classification (05)**: class-count histogram (true vs predicted) beside the
  confusion matrix.
- **Detection (08)**: per-class ground-truth histogram beside the score-threshold
  sweep, to read AP asymmetries against data frequency.

## Round 42 — review of the model notebooks (planned)

**Review.** Metrics after the last rounds: 05 classification 0.80/0.71 (val/test);
06 segmentation mIoU 0.50/0.51; 07 cloud IoU 0.84/0.85; 08 detection mAP@0.5
0.41/0.39, instance IoU 0.56. Figures and read-outs are in good shape and no
read-out cites a stale number. Defects found:

- 07: the evaluation **figure read-out sits above the Evaluation section** (a
  marker matched the training loop's `evaluate(...)` unpacking, not the eval
  cell); 06 and 07 also lack an explicit **evaluation-protocol** note.
- 06 has 6 figure cells but 5 read-outs (the class-colour legend); 08 has 9
  figure cells but 8 read-outs (`colorize_instances`).
- 06/07/08 titles still say "**PyTorch + MPS**" although the device is now
  auto-selected (cuda/mps/xpu/cpu).
- 05's seed-variance result is a print-only cell with no figure/read-out.
- Detection mAP is now the weakest metric because ~40 % opaque cloud cover hides
  small objects.

**Prioritized plan (for the next implementation rounds).**

1. **P1 — fix the read-out defects**: move 07's evaluation read-out below the
   figure, add the missing read-outs (06 colour legend, 08 `colorize_instances`),
   add a one-line evaluation-protocol note to 06/07, and retitle 06/07/08 to
   "PyTorch (CUDA / MPS / XPU / CPU)".
2. **P1 — 05 seed-variance figure**: plot the three seed accuracies with their
   mean ± std and add a read-out (turns a print into a teaching figure).
3. **P2 — training-recipe box** per notebook: learning rate, epochs, batch size,
   augmentation, split sizes, parameter count, one compact table at the top of
   the training section.
4. **P2 — detection error taxonomy (08)**: split the false positives into
   background/localisation/duplicate/class-confusion (using the existing greedy
   matching) and show a small breakdown, so the low `car`/`rock` AP is
   explained rather than just reported.
5. **P2 — resolution ablation (08)**: a short 64 px vs 96 px detector comparison
   (the ROADMAP M1 item) to demonstrate the resolution ceiling.
6. **P3 — cross-notebook learning path**: a short "how the notebooks build on
   each other" map in the README (classification -> segmentation -> cloud ->
   detection/instance).
7. **P3 — segmentation rare-class strategy**: the class-weighted experiment
   exists; extend it with a short note/table of what more (dice loss, class
   sampling) would change.

### Round 42 — results

- **Read-out/title fixes**: 07's evaluation read-out moved below its figure; 06/07
  gained protocol notes; 06/08 gained their missing read-outs; 06/07/08 retitled
  to "PyTorch, automatic device".
- **05 seed variance**: plotted the three seed accuracies with mean ± std
  (0.685 ± 0.055) and a read-out.
- **Recipe tables** added to 05-08 (model/params, input, loss, optimiser, epochs,
  batch, augmentation, splits); 08 has detection and instance tables.
- **Detection error taxonomy (08)**: false positives split by cause - on
  validation **localisation 524, background 252, duplicates 75, class confusion
  2** - so box tightness, not clutter, is the main error.
- **Resolution ablation (08)**: same architecture at 64 px scores **0.13 val /
  0.12 test** mAP@0.5 vs **0.41 / 0.39** at 96 px, demonstrating the resolution
  ceiling.
