# Journal

Iteration log. Newest at the top. Each entry: item addressed, what changed,
what the re-render showed, and follow-ups.


### 2026-09-23 23:20 — Round 32 (educational value of the model notebooks)

Reviewed notebooks 05–08 and found the teaching gap was in **interpreting**
results, not in the mechanics. Implemented five items:

1. **Training curves everywhere** — 06, 07 and both 08 models now plot loss and
   the task metric vs epoch with the best epoch marked (05 already did).
2. **Trivial baselines** — 05 majority-class accuracy (0.14 vs model 0.97);
   06 most-frequent-class baseline (high accuracy, near-zero mIoU); 07 all-clear
   / all-cloud (all-clear scores 0.78 accuracy while detecting no cloud); 08
   no-object mAP 0 and mean-count MAE (5.66 vs model 3.00).
3. **Operating point for clouds (07)** — threshold sweep of IoU / Dice /
   precision / recall, showing 0.5 is a convention.
4. **Class prevalence vs per-class IoU (06)** — shows why rare classes dominate
   mIoU even when pixel accuracy looks high.
5. **Per-class precision-recall curves (08)** — refactored
   `average_precision` into `precision_recall_curve` and plotted AP per class,
   showing what mAP averages.

All notebooks re-run; 05–08 outputs refreshed; every new figure is described in
its notebook. 162 tests pass; ruff clean.

### 2026-09-23 18:25 — Round 31 (5-item review-driven iteration)

Fresh review of all scenes/objects/materials + notebooks, then five prioritized
items, one commit each:

1. **Conifer**: the spiky dahlia/star with a dark hole is replaced by a soft
   radial fir crown (concentric smooth blobs + batched needle streaks + small
   dark top). Used in forest/wilderness/mountain/pasture.
2. **Ground materials**: `soil` gains clods and dry cracks, `grass` gains blade
   ticks and a second tonal mottle, `crop` gains finer plough bands and grain,
   and `sand` gains a faint cross-ripple family plus finer grain.
3. **Rock + mountain**: rock slabs are smaller/denser with lighter facets and
   gravel grain; the mountain lowers its relief-overlay alpha (0.7 -> 0.45) and
   adds more grass/soil patches so it stops washing out to flat grey.
4. **Water/coast**: three shoreline harmonics, a two-band wet-sand gradient and
   foam broken into short segments; rivers lose the single bright centre line for
   faint dashed highlights.
5. **Detection anchors**: `kmeans_anchors` now clusters `(log w, log h)` for the
   varied detection boxes (`log_space=False` for the uniform forest crowns).
   Detection mAP@0.5 recovered 0.42 -> **0.47** val / 0.44 -> **0.45** test
   (rock 0.17 -> 0.29, car 0.18 -> 0.24, tree 0.32 -> 0.49).

All eight notebooks re-run; 162 tests pass; ruff clean. Results table in README
and Round-31 read-out/backlog in `VISUAL_QA.md` updated.

### 2026-09-23 12:10 — Round 30 (scale consistency, legend, boulders, avenues)

- **Scale consistency (R28-5)**: parking-lot cars were `size*0.09-0.12` (~0.006-0.012
  of the frame) vs roadside cars `0.04-0.056` — 4-6x too small. `draw_parking_lot`
  now scales cars to `size*0.24-0.30` with matching stall spacing, and pads were
  enlarged (urban `step*0.85-1.0`, coastal `0.12-0.18`). A close-up confirms lot
  cars now match the standalone road car.
- **Trade-off**: the larger pads shifted the k-means anchors toward bigger boxes,
  dropping detection mAP@0.5 (val 0.46 -> **0.42**, test 0.50 -> **0.44**; rock
  0.27 -> 0.17, car 0.22 -> 0.18). Annotating the car-sized lot cars was tested and
  *lowered* car AP further (0.18 -> 0.16) — packed overlapping boxes — so they stay
  drawn-but-unlabelled. Visual realism kept; the notebooks state the trade-off.
- **Legend**: the detection figures now carry an on-image legend (green = ground
  truth/prediction; the failure gallery adds yellow = missed).
- **Boulders**: sparse/medium forests in `make_forest_data` now scatter grey
  boulders under the canopy, not counted as instances.
- **Avenues**: a fourth `avenue` layout puts street trees in near-regular rows
  along a road (dashed centre line) — the order/discipline test. Instance IoU held
  (0.572 vs 0.576), MAPE 23 %.
- All notebooks re-run (`01`-`08`); 162 tests pass; ruff clean. Report updated in
  `VISUAL_QA.md` (Round 30) and the README results table.

### 2026-09-23 09:40 — Round 29 (dataset-generation profiling + 3x speedup)

- **Profiled** 96 px scene generation (`/tmp/time08.py`): 360 mixed scenes took
  **76 s** (~211 ms/scene), dominated by per-artist matplotlib bookkeeping and
  repeated texture synthesis.
- **Optimization 1 — material-texture cache warmth**: `_compose` now requests
  `material_textures(size, (seed or 0) % 8, needed_materials)` for all materials
  a scene needs, so the per-material LRU cache is populated once instead of
  re-synthesizing textures on each fill.
- **Optimization 2 — batched road dashes**: `draw_road` collects all centre-line
  dash segments and inserts them as a single `LineCollection` (new
  `ShapeBudget.add_line_collection`) instead of one `plot` per dash.
- **Optimization 3 — skip limit bookkeeping**: `_compose` no-ops
  `ax._update_patch_limits` / `_update_line_limits` / `_update_collection_limits`
  (autoscaling is irrelevant because the axes limits are set explicitly).
- **Result**: urban 242 -> **84 ms**, forest 210 -> **154 ms**, fields 189 ->
  **122 ms**, coastal_city 211 -> **128 ms**; 360 mixed scenes 76 s -> **25 s**
  (**~3x**). Rendered `/tmp/opt_quality.png` (8 scenes) and confirmed no visual
  regression. 162 tests pass; ruff clean.

### 2026-09-22 16:00 — Round 28 (traffic, harder forests, MAPE)

- **Traffic**: urban and coastal-city scenes now place many more cars/trucks on
  the roads, kerbside parking and occasional **traffic jams** (queues of cars).
  Car AP rose 0.15 -> **0.22**.
- **Forest instance task**: `make_forest_data` zoomed out (smaller crowns), more
  trees (up to ~32) with a wider density range, plus light blur/detector noise.
  The task is deliberately harder: instance IoU 0.64 -> 0.58.
- **MAPE**: tree-count evaluation now reports **MAPE** in addition to MAE
  (validation MAE 3.4 / MAPE 20 %; test 3.1 / 19 %).
- **Bug fix (critical)**: adding vehicles exposed a NaN divergence — edge-clipped
  cars produced **zero-size boxes** and `log(0)` in the box targets poisoned the
  detection loss (and the resulting NaNs made MPS ~35x slower). Fixed by
  bounds-checking vehicles in `_place_vehicle` and skipping degenerate boxes in
  `encode_targets`.
- **Parking-lot cars**: annotated them, but they are sub-pixel (pads are drawn
  too small), which *lowered* car AP to 0.11, so the annotation was reverted. The
  remaining item is a **scale-consistency** fix (larger parking pads).
- Final: detection mAP@0.5 val 0.46 / test 0.50; instance val IoU 0.58 / MAE 3.4
  (MAPE 20 %). All notebooks re-run; 162 tests pass; ruff clean.


### 2026-09-22 12:30 — Round 27 (cropland boundaries, simplify 08, notes)

- **Cropland boundary defect**: confirmed in notebook 06 that cropland stopped
  short of the frame. Root causes: (a) `urban_fringe` inset its field mosaic by
  6 %, so cropland touched **0 %** of the frame edges; (b) the `fields` scene
  biased land use to grass/woodland (low/steep cells) and filled woodland with
  the *soil* material (bare_soil), leaving large non-crop borders. Fixed: mosaic
  boxes span the full frame, `fields` defaults to cropland with rare grass /
  woodland, woodland filled with foliage. Edge cropland fraction: `urban_fringe`
  0 -> 0.73 (left); `fields` 0.35 -> ~0.75.
- **Simplified notebook 08**: removed the prototype (YOLACT) head, the learned
  prototype-mask display and the prototype-vs-per-anchor comparison; only the
  per-anchor instance head remains. Validation MAE 1.8 / IoU 0.64 / det 0.86;
  test MAE 1.5 / IoU 0.63 / det 0.84; detection mAP@0.5 val 0.50 / test 0.54.
- **Contextual notes**: rewrote "Interpreting these numbers" in 05/06/07 with
  task-relevant examples (no cars where cars are not predicted).
- Re-ran 03/04/05/06/07 and 08; 162 tests pass; ruff clean.


### 2026-09-22 09:00 — Round 26 (prioritized plan: E1/E2/E3, D1/D2)

Executed the Round-25 priorities:

- **E1 — test split + protocol** in 05-08 (train / validation / test; validation
  selects the checkpoint, test reported once). Val and test are close everywhere.
- **E2 — failure galleries**: 05 least-confident (boundary scenes), 06 worst
  pixel-error (clouded scenes dominate), 07 worst cloud-IoU (missed/over edges).
- **E3 — results table + guidance** added to the README ("Results at a glance").
- **D1 — mountain** less washed: hillshade alpha 0.7, more grass/soil patches,
  snow softer, and the rock material's bright facets toned down.
- **D2 — pasture/arid densified**: more hedgerows/tree clumps/plants; arid gets a
  wadi in 85 % of scenes plus more rocks, scrub and gravel patches.

Test metrics: 06 acc 0.79 / mIoU 0.54; 07 IoU 0.91 / Dice 0.96; 08 detection
mAP@0.5 0.50; prototype IoU 0.72 / MAE 6.47; per-anchor IoU 0.63 / MAE 1.78.
Notebooks 03/04 re-run; all notebooks clean; 162 tests pass; ruff clean.


### 2026-09-22 05:45 — Round 25 (backlog review + prioritization)

- Reviewed `VISUAL_QA.md`, removed stale backlog entries that predate the R17
  detection-class change, and consolidated the rest into a prioritized table.
- Priorities set by teaching value: **E1** test split + protocol, **E2** failure
  galleries for 05/06/07, **E3** results table/guidance, then dataset realism
  (D1 mountain, D2 pasture/arid), then a **resolution ablation** (M1) that
  demonstrates the resolution ceiling rather than chasing mAP.
- Recorded the plan in `ROADMAP.md` (Round 25).


### 2026-09-22 05:15 — Round 24 (interpreting noisy/low-res metrics)

- Added an **"Interpreting these numbers"** closing section to notebooks 05-08
  making the educational point explicit: high accuracy on clean renders can
  mislead, noise/clouds/low resolution bound recoverable accuracy, some errors
  are irreducible (~4 px cars, fully occluded crowns), and evaluation means
  choosing an operating point. Tailored to each task (64 vs 96 px, clouds,
  soft cloud masks, dense forests).
- Markdown-only change; no re-execution needed. 162 tests pass; ruff clean.


### 2026-09-22 04:30 — Round 23 (noise, dense forests, 5 steps)

Five top-priority, inspection-driven steps:

1. **Sensor noise in 06** — `apply_sensor` (light blur/haze + Poisson/Gaussian)
   plus ~20 % partial cloud cover; label masks unchanged. Segmentation acc 0.81,
   mIoU 0.58; the prediction grid segments through haze/clouds.
2. **Sensor noise in 08** — first pass was too strong (mAP 0.60 -> 0.43), so it
   was softened to "a small amount"; detection mAP lands at **0.54**.
3. **Diverse tree instance data** — `make_forest_data` now samples sparse (6-11),
   medium (11-20) and dense (16-28) regimes with soil/grass/foliage backgrounds
   and occasional autumn trees; 400 scenes, `top_k` 45.
4. **Visible-instance relabel** — fully occluded crowns are dropped so the count
   is over visible trees; prototype count MAE 5.9 -> 4.4.
5. **Per-anchor masks** — `MASK_SIZE` 16 -> 24 and decode NMS 0.5 -> 0.4 removed
   the over-fragmentation: per-anchor count MAE 8.7 -> **1.7**.

- Final: detection mAP@0.5 0.54 (noise + clouds); instance prototype IoU 0.73 /
  MAE 4.40, per-anchor IoU 0.64 / MAE 1.69.
- All notebooks execute; 162 tests pass; ruff clean.


### 2026-09-22 01:00 — Round 22 (educational improvements)

- **Learning objectives + exercises** added as a markdown box to notebooks 05-08
  (what each notebook teaches, plus 2-3 exercises).
- **Failure gallery** added to notebook 08: greedy TP/FP/FN matching, drawn as a
  3x5 grid (green = matched, red = false positive, yellow = missed). Re-ran the
  notebook (mAP@0.5 0.599; car 0.196). The gallery shows the model's real errors:
  false positives in dense urban blocks, missed buildings in coastal cities.
- **Fast mode**: a `FAST` flag in notebook 08 scales samples/epochs for a quick
  smoke run.
- **README**: documented the pip-installed `torch`/`scikit-learn` extras and the
  training-notebook runtime exception.
- All notebooks execute cleanly; 162 tests pass; ruff clean.


### 2026-09-21 23:30 — Round 21 (training-notebook inspection + educational review)

- Re-ran all training notebooks after the dataset-generation fixes.
  - 05 classification: val accuracy **0.80** (grid now spans all classes).
  - 06 segmentation: pixel accuracy **0.86**.
  - 07 cloud: IoU **0.91**, Dice **0.96**.
  - 08 detection: mAP@0.5 **0.60** (car 0.20, building 0.88, parking 0.88);
    instance prototype IoU **0.82** (det 0.98), per-anchor **0.69** (0.94).
- Fixed the classification qualitative grid (was dominated by arid/coastal):
  now samples round-robin across classes.
- Wrote an educational review in `VISUAL_QA.md`: missing objectives boxes,
  exercises, a test split, failure galleries, a fast mode, and README notes on
  the pip extras.


### 2026-09-21 18:30 — Round 20 (dataset-generation iterations)

Visual inspection of notebooks 01-04, then four fixes (one per iteration):

1. **Hillshade** — `_terrain` now emits a continuous grayscale relief (faded in
   mainly on slopes) instead of a binary white/black mottle; mountains read as
   ridges/valleys.
2. **Water** — new `_ripples` helper replaces the straight `_lines` streaks, so
   water no longer looks banded.
3. **Urban density** — a second, smaller building is placed in ~60 % of blocks
   (with road clearance), so blocks are no longer large bare-concrete slabs.
4. **Shadows** — object shadow alpha 0.30 -> 0.42 and building shadow
   0.14/0.24 -> 0.18/0.30, giving darker, more consistent drop shadows.

All four dataset notebooks re-executed; 162 tests pass; ruff clean.


### 2026-09-21 15:40 — Round 19 (inspection-driven, 3 iterations)

Visual inspection of notebooks 03 and 08 + code review, then three iterations:

1. **Coastal city realism** — buildings now sit on a jittered block grid kept
   clear of the coast road and cross streets (no slabs on roads); the marina
   basin is water-filled with a quay edge.
2. **Car realism / detectability** — cars and trucks are drawn at a realistic
   ~4-6 px (96 px frames) instead of ~3 px. Detection mAP@0.5 0.555 -> **0.593**
   and car AP 0.12 -> **0.18** (building 0.89, parking 0.84, boat 0.68, rock
   0.50, tree 0.46).
3. **Propagation + docs** — re-ran notebooks 03/04/06/07 with the new scenes and
   car sizes; updated the diversity analysis (eleven scene types incl.
   `coastal_city`) and the detection read-out; refreshed `VISUAL_QA.md`.

- Note: a `base=48` detection model reached 0.576 offline but made the notebook
  too slow (>20 min with data generation), so it was reverted to `base=32`.
- Next: class-frequency-weighted loss for rare classes; test-time flip TTA;
  trim detection runtime; 128 px for cars.


### 2026-09-21 11:30 — Round 18 (coastal cities, field boundaries, docs)

- **Coastal cities**: new `coastal_city` scene — sea + beach, a rubble
  **breakwater**, **piers** on pilings, a sheltered **marina** with finger docks
  and moored boats, plus a coast road, waterfront buildings, parking, cars and a
  park. Registered in `SCENES`, covered by a new test, inspected in notebook 03.
- **Field boundaries**: `forest_fields` now fills the agricultural half to the
  top/bottom/right frame edges (was inset by 6 %) and fills the woodland half
  with canopy instead of bare soil. Inspected in notebook 03.
- **Anchors**: notebook 08 `kmeans_anchors` uses scikit-learn `KMeans(n_init=10)`
  (added as an `install-sklearn` pixi task) instead of a hand-rolled k-means.
- **Docs/comments**: short docstrings added to notebook helpers and key `mlpsyn`
  helpers; 1-5 line inline comments added to the modelling notebooks.
- **Qualitative grids**: detection (08) and classification (05) now show a
  `3 x 5` validation grid; segmentation/cloud/instance grids already qualify.
- **Detection**: `coastal_city` added to the dataset (breakwater->rock,
  pier->building, marina->parking); mAP@0.5 **0.555** on the richer mix
  (building 0.84, parking 0.78, boat 0.70, tree 0.45, rock 0.44, car 0.12).
- All notebooks re-executed; 162 tests pass; ruff clean.


### 2026-09-21 09:05 — Round 17 (urban/coastal detection refocus)

- **Change**: notebook 08 detection task refocused on **building, rock, tree,
  car, boat, parking** with a dataset that is mostly dense `urban` blocks plus
  `urban_fringe` and `coastal (beach)` scenes (mapping silo->building,
  truck->car, conifer/bush/orchard->tree). Added per-image brightness/contrast
  jitter and raised the detection resolution to **96 x 96 (24 x 24 grid)**, while
  the forest instance section stays at 64 px via new `FOREST_SIZE`/`FOREST_GRID`.
- **Iterations / results**:
  - 64 px, new classes, 6 anchors, flips: mAP@0.5 **0.406**;
  - + photometric jitter + more data (600): **0.454**;
  - + 96 px (360 samples, 150 epochs): **0.594** (building 0.96, parking 0.93,
    boat 0.67, tree 0.53, rock 0.37, car 0.09).
  Cars stay hard even at 96 px (car ~4 px); 128 px is the next lever.
- **Also**: tried a stride-2 (32x32) head at 64 px — it did not train (mAP
  collapsed to ~0.03 even for buildings), and focal loss / 8 anchors were
  slightly worse than BCE with 6 anchors.
- **Cost**: notebook now runs ~7 min (detection data ~3 min + training ~2 min).


### 2026-09-20 21:10 — Round 16 (prototype vs per-anchor masks)

- **Change**: notebook 08 gained a second instance head, `AnchorMaskDetector`,
  that predicts one `16 x 16` mask per `(cell, anchor)` directly (no prototypes),
  trained on GT masks cropped to their box and decoded by resizing to the
  predicted box. Added a tradeoff table (params, train time, ms/image, IoU,
  detection, count MAE) and a GT/prototype/per-anchor figure.
- **Result** (same detector, data, flips, 120 epochs):
  - prototype masks: 144.8k params, 109.6 s train, 2.3 ms/img, IoU **0.816**,
    detection **0.99**, count MAE 1.12;
  - per-anchor masks: 153.6k params, 69.9 s train, 1.9 ms/img, IoU **0.687**,
    detection 0.92, count MAE **0.98**.
  The prototype head gives the best masks at ~1.6x training cost; the per-anchor
  head is cheaper and counts slightly better but masks overlapping crowns worse.
- **Note**: the per-anchor target/decode must both be box-relative, otherwise
  the model learns full-image masks and decode fails (a first attempt scored
  IoU 0.32).


### 2026-09-20 19:40 — Round 15 (detection + instance improvements)

- **Change**: notebook 08 improved with random flips (images and targets together),
  more data (detection 360, forest 360), 6 detection anchors, 140 / 120 epochs
  and best-checkpoint selection by mAP / instance IoU.
- **Bug fixed**: `flip_forest_batch` flipped objectness/boxes/masks but not the
  instance-id tensor, so mask targets were mismatched on flipped batches and the
  mask head collapsed (eval IoU 0). Flipping the ids too fixed it.
- **Result**: detection mAP@0.5 **0.41** (was 0.26; water 0.85, boat 0.60,
  building 0.39, field 0.28, vegetation 0.19, rock 0.14); instance **IoU 0.82**
  (was 0.67), detection **411/417 = 0.99**, count MAE **1.12**.
- **Follow-up**: vegetation/rock still weak (few-pixel objects at 64 px); runtime
  ~5 min, could be trimmed.


### 2026-09-20 18:05 — Round 14 (mask head on the detector)

- **Change**: notebook 08 instance section rebuilt. The two-headed U-Net +
  centre heatmap + watershed decoder is replaced by a **multi-head anchor
  detector**: objectness head, box head, an **extra mask head per anchor**
  (`P` coefficients per `(cell, anchor)`) and a **prototype branch with
  upscaling layers** (`16 -> 32 -> 64`). Instance mask =
  `sigmoid(sum_p coeff_p * proto_p)` cropped to the predicted box, painted in
  score order. Anchors are k-means tree boxes (3); `P = 16` prototypes.
- **Result**: count MAE **1.28** trees, instance IoU **0.67**, detection
  **234/281 = 0.83** (60 epochs, ~63 s). Detection head unchanged (mAP@0.5
  0.26).
- **Debug note**: an earlier version of the loop collapsed the mask branch
  (instance IoU 0.35); rewriting the batch/mask-target lines fixed it. Watch for
  mask collapse; consider ROI-cropped mask logits for sharper boundaries.

### 2026-09-20 16:45 — Round 13b (object detection to last position)

- **Change**: `06_train_object_detection.ipynb` -> `08_...` (last), so `07`
  semantic segmentation -> `06` and `08` cloud occlusion -> `07`. README index
  updated. No code/output changes.

### 2026-09-20 16:35 — Round 13 (renumber training notebooks)

- **Change**: `08_train_scene_classification.ipynb` -> `05_...` (simplest task
  first); the other training notebooks shift down: `05_object_detection` -> `06`,
  `06_semantic_segmentation` -> `07`, `07_cloud_occlusion` -> `08`. README now
  lists the full notebook index. No code/output changes.

<!-- entries below -->

### 2026-09-20 16:10 — Round 12 (anchor detection + tree instance segmentation)

- **Change**: notebook 05 reworked.
  - **R12-1 anchors**: 16x16 grid x 5 anchors/cell (k-means on training box
    shapes, shape-IoU distance); per-slot objectness/class/box targets
    (`tx,ty,tw,th`); loss `obj + 2*class + 2*box`; decode with sigmoid/exp + NMS;
    mAP@0.5 per class.
  - **R12-2 instance segmentation**: new section on overlapping forest canopies.
    Self-contained generator (soil material + irregular crowns, later crown wins
    on overlap) with exact instance masks; two-headed U-Net predicting canopy +
    centre heatmap; decode = top-hat peaks + nearest-peak (watershed) labelling;
    metrics = count MAE, per-instance IoU, detection rate; GT/pred colour figures.
- **Result**: detection mAP@0.5 = 0.261 (peaks ~0.266 at epoch 29, overfits
  after); water/boat strong (0.60/0.30), field weak (0.08). Instance
  segmentation: count MAE 0.00, instance IoU 0.98, detection 301/301.
- **Follow-up (R12-3)**: add a larger anchor / more field samples; more epochs
  worsen mAP, so consider mild augmentation or best-checkpoint selection.

### 2026-09-20 15:25 — Round 11 (explicit variable names in training notebooks)

- **Change**: notebooks 05–08 rebuilt with explicit names:
  `cx/cy`→`center_x/center_y`, `ci`→`class_idx`, `ix/iy`→`cell_x/cell_y`,
  `gx/gy`→`grid_x/grid_y`, `w/h`→`width/height`, `X/Y`→`images/labels`,
  `tp/fp`→`true_positives/false_positives`, `dets/gts`→
  `detections/ground_truths`, `mrec/mpre`→`recall_curve/precision_curve`,
  `perm/idx`→`permutation/batch_indices`, `cm`→`confusion`, and descriptive
  encoder/decoder names. Short names kept only where dense arithmetic needs them
  (`dx/dy` Gaussian offsets, `enc1..enc4`/`dec1..dec3`) with comments above.
- **Verified**: all four notebooks re-executed with no errors and equivalent
  metrics (detection mAP@0.5 0.35, segmentation acc 0.84 / mIoU 0.60, cloud IoU
  0.93 / Dice 0.97, classification acc 0.93).
- **Housekeeping**: ROADMAP Round 11 and VISUAL_QA updated.

### 2026-09-20 14:57 — Round 10 (loss/metric math + pre-training demos)

- **Change**: each training notebook (05 detection, 06 segmentation, 07 cloud,
  08 classification) gained a markdown section describing the **prediction and
  target spaces** and the **mathematical definition of the losses and metrics**,
  followed by a self-contained code cell that computes loss/metric values on
  toy prediction/ground-truth pairs *before any training*.
  - 05: CenterNet focal heatmap loss + L1 size/offset, IoU and mAP@0.5
    (all-point interpolation). Demo: focal 0.009 (correct cell) vs 5.544 (wrong
    cell), IoU 0.818 vs 0.0, AP 1.0.
  - 06: cross-entropy with `ignore_index=0`; pixel accuracy, per-class IoU and
    mIoU. Demo: CE 3.185 = manual, perfect logits -> 0, acc/mIoU 1.0.
  - 07: BCE-with-logits; IoU, Dice, pixel accuracy. Demo: BCE 0.273 = manual.
  - 08: categorical cross-entropy; accuracy, per-class accuracy, confusion
    matrix. Demo: CE 0.687 = manual.
- **Housekeeping**: all four notebooks re-executed with no errors; a small
  `gather` bug in the segmentation demo was fixed. Docs updated
  (ROADMAP Round 10, VISUAL_QA).

### 2026-09-20 14:01 — Round 9 (classification, multi-class detection, legend)

- **R9-1**: new `08_train_scene_classification.ipynb` — 7 scene classes
  (arid, coastal, cropland, forest, mountain, pasture, urban), one label per
  image; ambiguous `forest_fields` -> cropland and `urban_fringe` -> urban
  (documented). Balanced sampling (44/class), small CNN, cross-entropy; val
  accuracy 0.93 + per-class accuracy + confusion matrix. ~55 s.
- **R9-2**: reworked `05_train_object_detection.ipynb` to 6 classes (building,
  field, vegetation, rock, water, boat) with a per-class heatmap; evaluation is
  now **mAP@0.5 IoU** (precision-recall per class, all-point interpolation) plus
  a per-class colour legend on the predictions. mAP@0.5 = 0.35 (~100 s).
- **R9-3**: `06_train_semantic_segmentation.ipynb` gained a class colour legend
  and a manual seed + 30 epochs: pixel accuracy 0.87, mIoU 0.64.
- **Housekeeping**: all notebooks re-executed with no errors; `src`/`tests` still
  pass; ROADMAP/VISUAL_QA updated. Note: an initial detection run hung in
  nbconvert (a one-off MPS/kernel stall); re-running completed in 1m42s.

### 2026-09-20 12:26 — Round 8 (roads, segmentation, training notebooks)

- **R8-1 dead-end roads**: added `_close_network(tree, edges)` (raise every
  degree-1 cell to degree >= 2 using an unused edge) and used it in
  `fields_scene` and `_field_mosaic`. Test added.
- **R8-2/8-3 masks**: new `mlpsyn._mode` render flag; `generate_scene(..., mask=True)`
  draws flat semantic-class colours (relief/soft-alpha/rows suppressed);
  `mlpsyn.segmentation` maps the flat render to 13 classes (0 = unlabeled).
  `mlpsyn.sensor.cloud_alpha` + `clouds(..., return_mask=True)` expose the
  occlusion field. 5 new tests.
- **R8-4 notebook**: nb04 gained a 4-row figure (original, +boxes, semantic
  labels, cloud mask) for five scenes (4.63 MB, still < 5 MB).
- **R8-5 training notebooks**: installed torch 2.14 (pip, cp314 macOS arm64
  wheel; `pixi run -e dev install-torch`) and added three MPS notebooks:
  - nb05 object detection — class-agnostic CenterNet-lite; P 0.59 / R 0.52 /
    F1 0.55 / matched mIoU 0.62 (~100 s). Fixed a channel-broadcast bug in the
    heatmap MSE that had made the first run meaningless.
  - nb06 semantic segmentation — small U-Net; pixel acc 0.81, mIoU 0.56
    (~100 s).
  - nb07 cloud occlusion — small U-Net; IoU 0.94, Dice 0.97, acc 0.99 (~65 s).
  - Env needs `KMP_DUPLICATE_LIB_OK=1` and `OMP_NUM_THREADS=1` (set in the
    notebooks) to avoid a libomp crash.
- **Housekeeping**: 159 tests pass, ruff clean on `src`/`tests`; `pyproject`
  gains `pip` and the `install-torch` task; ROADMAP/VISUAL_QA updated.

### 2026-09-20 03:50 — Round 7, Iter 22 (R7-8 automatic diversity metrics)

- **Change**: new `mlpsyn.metrics` with `label_histogram`, `objects_per_scene`,
  `spatial_coverage` and `diversity_report` (label entropy + mean spatial
  coverage); exported from `mlpsyn`. nb04 gained a section printing a 30-scene
  report.
- **Result**: over 30 random scenes the vocabulary is well covered — 16 labels
  present, 7–61 objects/scene, label entropy 1.8, mean 4×4 spatial coverage
  0.71.
- **Tests**: 4 new metric tests (153 pass), ruff clean. nb04 max cell 7.0 s.
- **Next**: pixel-level segmentation masks (R5-13) — the remaining tooling item.

### 2026-09-20 03:30 — Round 7, Iter 21 (R5-12 COCO/YOLO export + split)

- **Change**: new `mlpsyn.annotations` with `category_index`, `yolo_lines`,
  `scene_to_coco`, `coco_dataset` and `train_val_test_split`; exported from
  `mlpsyn`. Boxes convert from the scene's bottom-left normalised format to
  pixel top-left (COCO) and normalised top-left (YOLO).
- **Notebook**: nb03 gained an "Exporting annotations and splitting a dataset"
  section printing COCO images/categories/annotations, a YOLO line and the split.
- **Tests**: 7 new annotation tests (149 pass), ruff clean. Coverage grid reduced
  to size 100 / dpi 60 so nb03 stays well under 10 s/cell (now 7.6 s max).
- **Next**: pixel-level segmentation masks (R5-13).

### 2026-09-20 03:05 — Round 7, Iter 20 (R7-7 orchard + greenhouse)

- **Change**: added `draw_orchard` (a soil block planted with a regular grid of
  small trees) and `draw_greenhouse` (a row of pale polytunnels) to `object.py`,
  `__all__` and `OBJECTS` (`orchard`, `greenhouse`). `fields_scene` now always
  plants one orchard on its flattest eligible cell (guaranteed appearance,
  elevation-consistent); `pasture_scene` places a greenhouse row ~45 % of the
  time.
- **Verified**: fields grid — the orchard reads as a planted block on level
  ground; pasture greenhouses read as rows of polytunnels. Object notebook now
  covers both.
- **Housekeeping**: 142 tests pass, ruff clean; roadmap object list updated.
- **Next**: tooling (COCO/YOLO export, segmentation masks) or more vocabulary.

### 2026-09-20 02:45 — Round 7, Iter 19 (R7-6 gentle-grade roads)

- **Item**: R7-6 — route roads/tracks along gentle ground.
- **Change**: `_spanning_tree(n, edges, weight=None)` now sorts edges by weight
  (minimum spanning tree). `fields_scene` and `_field_mosaic` weight by boundary
  height difference; `_field_mosaic` gained a `terrain` argument and the two
  boundary scenes pass their terrain. Added `_gentle_path` (lowest-slope
  candidate meander); forest tracks and the pasture farm track use it. Rivers
  already use `_low_valley_path`.
- **Verified**: 3×5 grid — forest/pasture tracks run along valleys and gentle
  ground; field roads cross level boundaries. New test for weighted MST and the
  terrain/valley/gentle helpers; 139 tests pass, ruff clean.
- **Next**: tooling (COCO/YOLO export, segmentation masks) or vocabulary.

### 2026-09-20 02:20 — Round 7, Iter 18 (R7-4/R7-5 elevation map)

- **Change**: replaced `_hillshade` with `_terrain(rng) -> (height, slope,
  hillshade)` and added `_sample`, `_pick_by_elevation`, `_low_valley_path` and
  `_relief_overlay`. The hillshade light is fixed upper-left so it agrees with
  the object drop shadows (shadow consistency).
- **Applied**: `mountain` places snow on high ground, conifers on mid slopes,
  boulders on steep slopes and ponds in low basins; `forest`/`wilderness`/
  `fields` route rivers along the lowest candidate meander; `fields` assigns
  crop/woodland/fallow from cell height/slope; `forest`, `wilderness`,
  `pasture` and `fields` draw a soft relief overlay.
- **Verified**: 4×5 terrain grid — rolling relief, valley-following streams and
  elevation-consistent mountain features. 139 tests pass.
- **Open**: roads still do not follow contours (R7-6).

### 2026-09-20 02:10 — Round 7, Iter 17 (R7-3 wilderness + pasture)

- **Change**: added `wilderness_scene` (foliage background, road-less continuous
  canopy, clearings, optional valley stream, high density) and `pasture_scene`
  (meadow tone variation, gravel farm track, hedgerows, tree clumps, farmstead
  buildings + silo, pond). Registered both in `SCENES` (10 types) and tests.
- **Verified**: 2×5 grid — wilderness reads as a continuous canopy; pasture as
  open farmland with a farm and hedges.
- **Housekeeping**: density floors later trimmed for notebook speed.

### 2026-09-20 02:00 — Round 7, Iter 16 (R7-2 parking, plazas, parks)

- **Change**: new `draw_parking_lot` (asphalt pad + stall lines + parked cars)
  and `draw_plaza` (paved square + fountain or planted centre + corner trees),
  added to `__all__` and `OBJECTS` (`parking`, `plaza`). Urban reserves 2–4 park
  blocks, 1–2 parking lots and 0–1 plaza per seed; rejected amenity cells fall
  through to buildings.
- **Verified**: 2×5 urban grid and close-ups — parking stalls with cars and
  fountains/planting are clearly readable; empty concrete blocks are reduced.

### 2026-09-20 01:50 — Round 7, Iter 15 (R7-1 buildings off roads)

- **Item**: buildings crossed roads and some roads looked like they led nowhere.
- **Change**: urban tracks each road as `(centreline, width)` in `corridors`;
  added `_distance_to_polyline`, `_road_clearance` and `_near_road`. Building,
  pool, parking and plaza sizes are derived from the local road clearance, so
  footprints fit inside blocks. The grid and arterial span the frame, so no road
  dead-ends inside it.
- **Verified**: 5-seed urban grid — no building crosses a street.
- **Next**: amenities (parking/plaza/parks).

### 2026-09-19 16:30 — Round 6, Iter 14 (R5-10 continuous wadi)

- **Item**: R5-10 — the arid wadi read as a chain of grey patches.
- **Change**: `arid_scene` now draws the wadi as one continuous feature: a pale
  bank ribbon (`width*1.6`), a full-opacity gravel bed (`width`, alpha 1) and a
  darker thalweg centreline; meander resolution raised (n=26) and orientation
  randomised. Removed the low-alpha gravel fill that let sand show through.
- **Verified**: 5-seed grid and a 420 px close-up — an unbroken dry riverbed
  with a bank; no patchiness.
- **Housekeeping**: nb03 arid analysis updated; `VISUAL_QA.md` R5-10 marked done.
- **Next**: R5-11 per-seed global parameters + shadow consistency, then tooling.

### 2026-09-19 16:15 — Round 6, Iter 13 (R6-6/R6-7 tests, speed, docs)

- **Tests**: added mosaic-tiling, Voronoi-area, spanning-tree-frontage, sensor
  ordering, mild/bounded-noise and clamping tests; removed the obsolete
  "trees not inside plots" test (fields now tile). 131 tests pass, ruff clean.
- **Speed**: the scene coverage cell had grown past 10 s. Added a small
  `_material_cached` LRU (size ≤ 256), disabled axes autoscale in `_compose`,
  reduced crop-row density (spacing 0.02–0.035, ≤ 45 rows, no rows on fallow
  grass), and switched all coverage grids to **5 columns** per `AGENTS.md`.
  Largest cell is now 6.2 s; notebooks: nb01 4.4 s, nb02 3.1 s, nb03 19.2 s,
  nb04 14.8 s.
- **Docs**: `ROADMAP.md` Round 6 plan marked done; `VISUAL_QA.md` Round 6
  backlog/progress/inspection notes added.
- **Next**: R5-10 continuous wadi, then R5-11 per-seed global parameters.

### 2026-09-19 16:05 — Round 6, Iter 12 (R6-5 boundary scenes)

- **Change**: extracted `_field_mosaic` (Voronoi cells + road spanning tree +
  hedgerows) and reused it for the farmland half of `forest_fields` and
  `urban_fringe`; removed their 2×3 rectangle grids and the now-unused
  `draw_field_plot` import and dead scene helpers.
- **Verified**: 5-seed grids — the farmland halves are irregular road-bounded
  mosaics abutting the soft region boundary.
- **Next**: tests, speed and docs.

### 2026-09-19 15:50 — Round 6, Iter 11 (R6-3/R6-4 fields mosaic)

- **Change**: rebuilt `fields_scene`. Jittered sites → `_voronoi_cells`
  (pure-numpy half-plane clipping) tile the frame; each cell is crop / soil /
  fallow grass / woodland with polygon crop rows (`draw_field_polygon`, new in
  `object.py`). `_shared_edges` + Kruskal `_spanning_tree` give a road network
  (asphalt country roads + gravel dirt tracks, `shoulder=False`) so every field
  has frontage; hedgerows line a subset of the remaining edges; a river crosses;
  silos sit by roads; woodland cells are tree-filled via `Path.contains_points`.
- **Why**: the old layout was a lattice of rectangles each surrounded by grass,
  with no road access.
- **Verified**: 5-seed and 420 px close-ups — irregular parcels that tile the
  ground, bounded by roads/tracks/hedges/river/woodland. No rectangle-in-grass.
- **Next**: reuse the mosaic in the boundary scenes.

### 2026-09-19 15:35 — Round 6, Iter 10 (R6-1/R6-2 sensor order + bounded noise)

- **Change**: `mlpsyn.sensor` reworked. `apply_sensor` now owns the physical
  chain **atmosphere (haze → clouds) → optics (blur) → detector (noise)** and
  accepts `haze_strength`/`cloud_coverage`, so noise is never applied before
  atmospheric effects. Added `atmosphere()`. Noise defaults are fixed and
  clamped: `GAUSSIAN_SIGMA=8`, `MAX_GAUSSIAN_SIGMA=14`, `POISSON_SCALE=600`,
  `MIN_POISSON_SCALE=220` (both give a comparable ~7–8/255 std). Notebook 04 now
  applies one mild noise to every frame via a single `apply_sensor` call.
- **Verified**: ordering test (`apply_sensor == gaussian(atmosphere(...))`),
  mild/bounded and comparable-strength tests, clamping test. Visual: fine grain
  sits on top of cloud/haze in the zoom; no frame over-noised.
- **Next**: agricultural layout realism.

### 2026-09-19 14:22 — Round 5, Iter 9 (R5-9 mountain elevation shading)

- **Item**: R5-9 — `mountain` read as a flat rocky plateau; the thin ridge
  shadow/highlight bands were almost invisible.
- **Change**: added `_elevation_field` (summed random sinusoids + a few Gaussian
  hills) and `_hillshade`, a Lambertian shaded-relief RGBA overlay (white on lit
  slopes, black on shaded ones, alpha growing with deviation from the mean).
  The overlay is drawn at zorder 1 over the rock ground; ground patches moved to
  zorder 2. Removed the old ridge-band loop.
- **Verified**: 5-seed and 10-seed mountain rows show broad lit/shaded slopes
  with soft transitions; light azimuth varies per seed. No hard banding.
- **Housekeeping**: nb03 mountain analysis updated; all four notebooks
  re-executed (nb03 3.33 MB, nb04 3.69 MB, every grid < 5 MB); 125 tests pass;
  ruff clean on `src`/`tests`.
- **Next**: R5-10 continuous wadi channel, then R5-11 per-seed global parameters.

### 2026-09-19 14:15 — Round 5, Iter 8 (R5-8 soft region boundaries)

- **Item**: R5-8 — the `forest_fields` and `urban_fringe` region boundaries were
  hard lines (a straight/rectangular fill edge).
- **Change**: added `_organic_edge` (the existing wavy split plus a smoothed
  high-frequency wobble) and `_soft_edge`, which draws three nested,
  increasingly wide bands of the region material on the neighbour side with
  alpha 0.45/0.25/0.10. All bands share one wobble field so they nest smoothly
  (independent wobbles produced visible contour stripes). Applied soil↔grass in
  `forest_fields` and concrete↔grass in `urban_fringe`.
- **Verified**: 5-seed and 400 px close-up renders — the boundary is now a
  mottled transition band, no hard line and no stepped contours. Trees sit in
  the transition, reading as woodland thinning into farmland.
- **Housekeeping**: nb03 scene analysis updated (both defects removed); nb01–nb04
  re-executed (nb03 3.3 MB, all grids < 5 MB); 125 tests pass; ruff clean on
  `src`/`tests` (pre-existing notebook import-order warnings remain).
- **Next**: R5-9 mountain ridge/elevation shading, then R5-10 continuous wadi.

### Round 5 (cont.) — scenes, 10-seed grids, diversity notebook

- **Sensor**: removed `vignette` and `jpeg` (camera artefacts, not satellite);
  `EFFECTS` now {clean, gaussian, poisson, blur, haze, clouds}.
- **New scenes**: `coastal` (shore, foam, boats, rocks), `mountain` (rock ground,
  snow patches, conifers), `arid` (sand, gravel wadi, rocks, scrub) and two
  boundary scenes `forest_fields` (woodland next to farmland) and `urban_fringe`
  (city next to fields, optionally split by a river). `SCENES` now has 8 types.
- **Notebooks**: coverage grids use **10 seeds** per type; added a
  `check_figure_size` helper (prints and asserts PNG < 5 MB) called before every
  `plt.show()`. Fixed a single-row subplot indexing bug (13 objects now).
- **Notebook 04**: new `04_scene_diversity.ipynb` — 5x5 grid of 25 random scene
  types with random sensor treatment and controlled cloud cover (7/25 cloudy,
  72 % clear).
- **Roadmap**: added milestone 9 (Visual QA & diversity); updated materials,
  objects and scene lists.
- **Visual findings**: boundary scenes have hard straight region edges (R5-8,
  medium); mountain lacks ridge shading (R5-9, medium); arid wadi is a chain of
  patches (R5-10, low).
- **Sizes**: nb01 1.85 MB, nb02 1.03 MB, nb03 3.28 MB, nb04 1.85 MB; every grid
  < 5 MB. 125 tests pass, ruff clean.

### Iter 3 — new materials (R5-3)

- **Change**: added `gravel` (dense small rounded stones on a grey-brown base)
  and `mud` (dark wet soil, bent cracks, pale puddle patches) to `materials.py`,
  `MATERIALS`, `MATERIAL_DESCRIPTIONS` and `__all__` (14 materials total).
- **Verified**: 5-seed grids — gravel reads as loose stones; mud as wet ground
  with cracks/puddles. A first mud version showed faint coloured specks; clod/
  mottle/speckle jitter was reduced.
- **Housekeeping**: nb01 gallery made dynamic (5 columns) to fit 14 materials;
  analyses updated; 116 tests pass; ruff clean; notebooks re-executed.

### Iter 2 — new objects (R5-2)

- **Change**: added `draw_silo`, `draw_pool`, `draw_truck`, `draw_hedgerow` to
  `object.py` and `OBJECTS` (now 12 objects). Integrated into scenes: `fields`
  gets 1–2 silos and 1–2 bush hedgerows on the margins (plus `concrete` as a
  material); `urban` gets backyard `pool`s (6% of blocks) and `truck`s (22% of
  vehicles).
- **Verified**: object grid (3 figures of 4 objects × 5 seeds) — truck reads as a
  box truck, silo as a ribbed circular structure with a domed roof (rebuilt from
  the first concentric-ring version), pool as a water rectangle with a tiled
  deck, hedgerow as a line of bushes. Scenes show silos/hedgerows on field
  margins and trucks/pools in the city.
- **Housekeeping**: updated the scene label test; nb02 materials list gained
  `concrete`/`asphalt`; nb02/nb03 analyses updated; 112 tests pass, ruff clean;
  notebooks re-executed.

## Round 5

### Iter 1 — sensor + atmosphere module (R5-1)

- **Change**: new `src/mlpsyn/sensor.py` with `gaussian_noise`, `poisson_noise`,
  `blur`, `vignette`, `jpeg`, `haze`, `clouds`, `apply_sensor` and an `EFFECTS`
  registry. Exported from `mlpsyn.__init__`. Schelling clouds use a multi-octave
  field, soft alpha and an offset ground shadow.
- **Notebook**: added a "Sensor and atmospheric effects" section to nb03 with an
  8-effect × 5-seed grid (fields scene) plus a visual analysis.
- **Verified**: grid inspected — poisson/blur/haze/clouds obvious; gaussian,
  vignette, jpeg subtle but present. Notebook size kept to ~5 MB by using
  size=120/dpi=70.
- **Housekeeping**: 108 tests pass (12 new in `tests/test_sensor.py`), ruff clean.

### Iter 12 — snow relief + asphalt cracks (P2)

- **Change**: added `_cracks` (short bent random-walk polylines) and used it for
  `asphalt` instead of long `_lines`. Added a second low-alpha blue-grey mottle
  to `snow` for relief.
- **Verified**: 5-seed grids — asphalt no longer has edge-long scratches; snow is
  no longer featureless white.
- **Housekeeping**: nb01 asphalt/snow analyses updated; tests/ruff clean;
  notebooks re-executed.

### Round 4 summary (start 08:26, ~45 min)

12 iterations, one item each; notebooks re-executed and analyses refreshed along
the way. All P0 (rotated-clip, rock material, rock object, conifer), P1
(tree/bush highlights, sand, field rows, forest clearings, urban variety) and P2
(pond sheen, building shadow, snow, asphalt) items are done. 96 tests pass, ruff
clean. Only roadmap items remain (linear-feature masks, seamless materials,
satellite background alignment).

### Iter 11 — building shadow softening (P2)

- **Change**: `draw_building` shadow is now two offset layers (scale 1.0 α0.24,
  scale 1.07 α0.14) instead of one hard α0.35 rectangle.
- **Verified**: 5-seed grid — softer shadow fringe; buildings still read cleanly.

### Iter 10 — pond sheen softening (P2)

- **Change**: `draw_pond` sheen replaced by 2–3 overlapping translucent ellipses
  (α0.35) instead of a single hard-edged light polygon.
- **Verified**: 5-seed grid — sheen reads as soft reflected light, no floating
  sheet.
- **Housekeeping**: nb02 pond/building analyses updated; notebooks re-executed;
  removed a now-unused `_blob` import in `object.py` (ruff); tests/ruff clean.

### Iter 9 — urban building variety / greenery (P1#9)

- **Change**: `urban_scene` building size range widened to 0.32–0.62·step, tree
  probability 0.12→0.16, park blocks now 1–3 at 0.75 probability.
- **Verified**: 5-seed grid — more varied building sizes/colours and more
  greenery; seed 3 is park-heavy (9 objects) but still plausible.
- **Housekeeping**: nb03 urban analysis updated.

### Iter 8 — soft forest clearings + density (P1#8)

- **Change**: added `alpha` to `fill_patch`; `_ground_patches` now nests three
  low-alpha (0.18) fills per clearing so edges fade into the ground. Forest base
  density raised to `size²/4600` (clamp 45–120) and stands to 4–7.
- **Verified**: 5-seed forest grid — counts 77–86 at 600px (consistent), soft
  clearings, no paint-splotch edges.
- **Housekeeping**: nb03 forest analysis updated; a first ruff run flagged a long
  line in `_ground_patches`, fixed; tests/ruff clean; notebooks re-executed.

### Iter 7 — `field_plot` row contrast + rotated clip (P1#7)

- **Change**: row colour in `draw_field_plot` is now the fill colour darkened to
  55% (instead of a jittered `_CROP_COLORS` entry), and rows are clipped with
  `TransformedPatchPath(plot)` so they stay inside rotated plots.
- **Verified**: crop and grass fills at 25° all show clear parallel rows; no
  rows spilling outside the footprint.
- **Housekeeping**: nb02/nb03 field analyses updated; tests/ruff clean; notebooks
  re-executed.

### Iter 6 — `sand` full-tile ripples (P1#6)

- **Change**: extended the ripple parameter range in `materials.sand` from
  `t = linspace(-0.8, 0.8)` to `t = linspace(-1.2, 1.8)`, so each ripple spans
  the whole projected width of the tile.
- **Verified**: 5-seed grid — ripples now cover the full tile at every seed; the
  plain quadrant/left-half boundary is gone.
- **Housekeeping**: notebook sand analysis updated; tests/ruff clean; notebooks
  re-executed.

### Iter 5 — `tree`/`bush` highlights (P1#5)

- **Change**: removed the single large bright highlight from `draw_tree`; two
  loops now add 3–5 medium and 2–3 small foliage highlights. In `draw_bush`,
  satellite blobs shrank to 0.08–0.14·size and 2–3 small highlights were added.
- **Verified**: 5-seed grid — canopies are mottled with small highlights instead
  of one flat bright blob.
- **Housekeeping**: notebook tree/bush analysis updated.

### Iter 4 — `conifer` object (P0#4)

- **Change**: `draw_conifer` now draws three overlapping jagged foliage rings
  (14–20 vertices, alternating outer/inner radius, inner ≈ 0.5–0.66·outer) over
  a base canopy, plus mid/dark centre discs. Replaced the thin-tine rosette.
- **Verified**: 5-seed object grid — reads as a dense fir crown, not a black
  spiky starburst. Scenes show jagged dark-green crowns in the forest.
- **Housekeeping**: tests/ruff clean; notebooks re-executed; nb02 conifer/rock
  analyses updated; `VISUAL_QA.md` P0 cleared.

### Iter 3 — `rock` object (P0#3)

- **Change**: `draw_rock` rebuilt as a rounded `_smooth_blob` boulder with a
  shadow ellipse, a lit facet (top-left), a shaded facet (bottom-right) and a
  subtle outline.
- **Verified**: 5-seed object grid — plausible boulder, no folded-paper look or
  grey wedges.

### Iter 2 — `rock` material (P0#2)

- **Change**: `rock` material rebuilt from small confetti plates to large
  overlapping angular slabs (n ≈ size²/500, radius 0.11–0.26) with lit/shaded
  facet polygons; removed the long straight crevice lines.
- **Verified**: 5-seed material grid — reads as a rocky/scree face in muted
  brown/grey. (First attempt with many small 0.07–0.18 slabs still looked like
  shattered glass; enlarged slabs fixed it.)
- **Housekeeping**: notebook gallery + coverage rock analyses updated.

## Round 4

### Iter 1 — `fill_patch` rotated clip (P0#1)

- **Change**: `materials.fill_patch` now clips the rotated texture with
  `TransformedPatchPath(patch)` instead of `set_clip_path(patch)`.
- **Why**: the texture image was rotated about the object centre but the clip
  stayed axis-aligned, so material spilled outside rotated footprints
  (axis-aligned octagon) and the true footprint outline appeared as an "extra"
  rotated rectangle. Visible on buildings/plots at 20°/55°.
- **Verified**: rotation demo at 0/20/55° — fill, outline and bbox now coincide.
- **Housekeeping**: 96 tests pass, ruff clean, all three notebooks re-executed,
  the nb02 rotation/coverage analyses updated, `VISUAL_QA.md` backlog updated.
