# Learning objectives

What each notebook teaches, in the order they are meant to be read. The first
four build the synthetic-data generator bottom-up (materials -> objects ->
scenes -> sampling/degradation); the last four train and evaluate models on
that data. They live in `notebooks/dataset_qa/` (01-04) and
`notebooks/modeling/` (05-08).

## Learning path at a glance

| # | Notebook | Level | Builds on | Main outcome |
|---|---|---|---|---|
| 01 | `01_texture_generation` | materials | — | render and reuse surface textures |
| 02 | `02_object_generation` | objects | 01 | draw annotated objects from materials |
| 03 | `03_scene_generation` | scenes | 01, 02 | compose and annotate full scenes |
| 04 | `04_scene_diversity` | sampling | 03 | add realistic sensor/atmosphere variation |
| 05 | `05_train_scene_classification` | image classification | 03, 04 | train a CNN, read a confusion matrix |
| 06 | `06_train_semantic_segmentation` | per-pixel classification | 03, 04 | train a U-Net, read IoU / mIoU |
| 07 | `07_train_cloud_occlusion` | binary segmentation | 04, 06 | segment clouds, read IoU / Dice |
| 08 | `08_train_object_detection` | detection + instances | 02, 03, 04 | train a detector and a mask head, read mAP / IoU / count error |

---

## 01 — Material textures (low level)

**Role.** The foundation: a *material* is a dense, abstract surface texture
(grass, foliage, asphalt, soil, sand, water, crop, rock, ...) with no object
semantics, reused by everything above it.

**By the end you can:**
- generate a material deterministically from a seed and explain why the same
  material varies but stays plausible across seeds;
- reuse a material as a clipped fill on arbitrary geometry with `fill_patch`
  (a roof, a canopy, a pond, a triangle);
- reason about the renderer's hard limits (the `MAX_SHAPES` shape budget) and
  about determinism (same seed -> same image).

**Key concepts:** material abstraction, texture grain, clipped fills, seeded
determinism, shape budget.

---

## 02 — Objects from materials

**Role.** Objects are geometry filled with a material, with a shadow and a
rotation. They are the annotated primitives that scenes are built from.

**By the end you can:**
- use the object contract
  `draw_x(ax, rng, budget, cx, cy, size, materials=None, rotation=None)` and the
  normalised `(x0, y0, x1, y1)` bounding box it returns;
- build an object from geometry + material fill (plus wheel/cabin/shadow
  details) and keep it readable at satellite scale;
- explain the **"no axis-aligned prior"** principle: objects are drawn at
  arbitrary orientation so a model cannot cheat on upright alignment;
- recover tight axis-aligned boxes for object-level annotations.

**Key concepts:** geometry + material fill, rotation-invariant drawing,
bounding-box contract, top-down (plan) view.

---

## 03 — Scenes: material ground + objects

**Role.** A scene composes a material ground with placed, annotated objects and
returns a `Scene` (image + objects).

**By the end you can:**
- compose a scene from a ground material plus semantic objects, with placement
  and orientation that respect the layout rules (roads, field mosaics, water);
- read `Scene.objects` and `SceneObject.pixel_bbox` (top-left origin) and export
  annotations to **COCO / YOLO** with a deterministic train/val/test split
  (`mlpsyn.annotations`);
- distinguish annotated point objects from unannotated background features
  (roads, rivers, parks);
- recognise the **fourteen scene types**, including the deliberately ambiguous
  transitional ones (`savanna`, `scrubland`, `woodland_pasture`, `coastal_city`,
  `forest_fields`, `urban_fringe`).

**Key concepts:** scene composition, annotation origins (scene vs pixel),
COCO/YOLO formats, dataset splits, ambiguous/transitional scenes.

---

## 04 — Scene diversity and sensor realism

**Role.** Sample the scene vocabulary and apply a physically ordered degradation
chain, then measure the diversity.

**By the end you can:**
- apply the **atmosphere -> optics -> detector** chain with `apply_sensor`
  (haze and cloud occlusion, then blur, then bounded Gaussian/Poisson noise) and
  explain why the order matters;
- describe how clouds (mostly opaque cores, ~25 % semi-transparent), haze, blur
  and noise differ and when each degrades a task most;
- quantify variety with `mlpsyn.metrics` (label histogram, objects per scene,
  spatial coverage, entropy) instead of eyeballing alone;
- verify that the RGB image, the semantic mask and the cloud mask are
  **pixel-aligned** because they come from the same deterministic spec.

**Key concepts:** degradation order, atmosphere vs optics vs detector,
cloud occlusion and transparency, pixel alignment, diversity metrics.

---

## 05 — Scene classification (image-level labels)

**Role.** The simplest model task: one label per image.

**By the end you can:**
- turn rendered scenes into a **class-balanced, image-level** dataset, including
  deliberate two-scene **mixtures** and a confusable-rival rule;
- build a small CNN with **global average pooling** and train it with
  **cross-entropy**;
- read a **confusion matrix** together with a **class-count histogram** and
  explain that many off-diagonal entries are **irreducible label ambiguity**
  (a 50/50 mixture has no single correct label), not modelling failure;
- quantify run-to-run noise with a **seed-variance** experiment and refuse to
  over-interpret differences smaller than the spread.

**Exercises.** Relabel the boundary scenes and watch the confusion move; add a
flip augmentation and explain the (small) gain.

**After this notebook:** you can build a classifier and, more importantly, judge
whether an accuracy number actually means something.

---

## 06 — Semantic segmentation (per-pixel labels)

**Role.** Move from one label per image to one label per pixel.

**By the end you can:**
- build paired **(image, semantic-mask)** data with `mlpsyn.segmentation`;
- implement a compact **U-Net** with skip connections;
- train with **cross-entropy**, correctly using `ignore_index` for the unlabeled
  class 0;
- read **per-class IoU** and **mean IoU (mIoU)**, and use a pixel-level
  **confusion matrix** + class histogram to see *which* classes confuse;
- explain why a **class-weighted loss** lifts rare classes and raises mIoU, and
  why pixel accuracy alone hides rare-class failure.

**Exercises.** Explain `ignore_index=0`; find the lowest-IoU classes and relate
them to class frequency.

**After this notebook:** you can train and honestly evaluate a segmentation model
and diagnose its per-class behaviour.

---

## 07 — Cloud-occlusion segmentation (binary)

**Role.** A focused binary-segmentation task on the atmosphere itself.

**By the end you can:**
- simulate cloud occlusion as a soft field with `mlpsyn.sensor.clouds`;
- train a U-Net for **binary segmentation** of the occlusion mask;
- read **IoU, Dice and pixel accuracy**, and explain why accuracy misleads when
  clouds are the minority class;
- treat the 0.5 threshold as a **convention**: a threshold sweep shows the
  precision/recall operating point;
- check **calibration** with a reliability diagram and fix over/under-confidence
  with **temperature scaling** (ECE before/after).

**Exercises.** Why Dice over accuracy here; widen the coverage range and describe
the effect on IoU / Dice.

**After this notebook:** you understand overlap metrics, operating points and
probability calibration for a binary mask.

---

## 08 — Object detection and instance segmentation

**Role.** The capstone: localise *and* classify objects, then separate
overlapping instances.

**By the end you can:**
- encode objects as **anchor-based detection targets** (objectness, one-hot
  class, box), choosing anchors by **k-means in log space** over box shapes;
- explain **why BCE-with-logits is used on the multi-class head** (binary CE
  applied independently per class; softmax CE is the mutually-exclusive
  alternative);
- train a multi-task detector (objectness + class + box) and evaluate it with
  **mAP@0.5** (score threshold, per-class NMS, greedy matching), including
  per-class **precision-recall curves**, mAP at several IoU thresholds, and a
  **score-threshold sweep**;
- explain the detector's **error taxonomy** (localisation vs background vs
  duplicate vs class confusion) and read the **resolution ablation** (96 px vs
  64 px);
- extend the detector with a **per-anchor mask head** (box-relative,
  crop-and-resize target) and evaluate instance masks with **IoU** and
  **count MAE / MAPE**, stratified by canopy density.

**Exercises.** Vary the NMS threshold; lower the score threshold and watch count
vs IoU; raise `MASK_SIZE` and weigh the accuracy gain against cost.

**After this notebook:** you can build a small detector + instance head, and you
know how to read mAP, PR curves, calibration and error breakdowns rather than a
single number.

---

## Educational-value review

### Strengths

- **Clear, bottom-up progression** (materials -> objects -> scenes -> sampling ->
  classification -> segmentation -> clouds -> detection), so every layer builds
  on the previous one.
- **The model notebooks state objectives, exercises and self-check notes**, and
  place a **read-out below every figure**.
- **Honest evaluation**: three disjoint splits and a reported-once test set;
  trivial baselines (majority class, all-background, all-clear, no-object,
  mean-count); training/validation curves with validation loss; confusion
  matrices with class histograms; PR curves; calibration; error taxonomy.
- **Deliberate difficulty is explained**: ambiguous scene mixtures with an
  irreducible label ceiling, semi-transparent/opaque clouds, the resolution
  ceiling and the few-pixel car.
- **Reproducibility**: seeded rendering and training, deterministic splits, and
  automatic device selection (CUDA / MPS / XPU / CPU).

### Gaps and recommendations

1. **01-04 have no explicit objectives or exercises.** This file supplies derived
   objectives; consider adding a short "Learning objectives / Try this" cell to
   each generator notebook (e.g. "change the sand ripple wavelength and predict
   the effect on realism").
2. **Fix small stale strings** (done for 01/03 in this round: the intro now says
   fourteen scene types, a duplicated "Ten scene types" read-out was removed, and
   "three seeds" became "five"). Keep watching for drift.
3. **Add a cross-notebook learning path** to the README, pointing here.
4. **08 is long (~50 cells, ~4-5 min).** A one-paragraph "contents / how to skim"
   box near the top, or splitting detection and instance notebooks, would help.
5. **More "why" depth in 07**: relate Dice to IoU explicitly and note that
   temperature scaling does not change the argmax / accuracy.
6. **Generator notebooks lack failure galleries**: a "when does this look wrong?"
   grid (e.g. overlapping conifers, clipped shadows) would teach inspection.
7. **Prerequisites / time budget** per notebook would let a reader plan a course
   (roughly: 01-04 a few minutes each, 05-07 ~1-2 min, 08 ~5 min).
