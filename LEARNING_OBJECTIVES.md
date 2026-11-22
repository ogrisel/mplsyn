# Learning objectives — modeling notebooks

What the four modeling notebooks in `notebooks/modeling/` teach, in the order
they are meant to be read. They all train and evaluate PyTorch models on the
synthetic data built by the generator notebooks; the dataset-building notebooks
live in `notebooks/dataset_qa/` and are out of scope here.

Every notebook runs on CUDA / MPS / XPU when available and falls back to CPU
(`pick_device()`), uses three disjoint train/validation/test splits, and reports
the test metric once.

## Learning path at a glance

| # | Notebook | Task | Builds on | Main outcome |
|---|---|---|---|---|
| 05 | `05_train_scene_classification` | image classification | scene data | train a CNN; read a confusion matrix honestly |
| 06 | `06_train_semantic_segmentation` | per-pixel classification | scene + mask data | train a U-Net; read IoU / mIoU and per-class errors |
| 07 | `07_train_cloud_occlusion` | binary segmentation | cloud masks | read IoU / Dice, operating points and calibration |
| 08 | `08_train_object_detection` | object detection | object annotations | train an anchor detector; read mAP, PR, error causes |
| 09 | `09_train_instance_segmentation` | instance segmentation | 08 + forest instances | add a mask head per anchor; read instance IoU and count error |

---

## 05 — Scene classification (image-level labels)

**Role.** The simplest model task: one label per image, on a deliberately
ambiguous dataset.

**By the end you can:**
- turn rendered scenes into a class-balanced image-level dataset, including
  two-scene **mixtures** whose rival is drawn from a confusable set;
- build a small CNN with **global average pooling** and train it with
  **cross-entropy**, watching the **training and validation loss** curves;
- read a **confusion matrix together with a class-count histogram**, and explain
  that the off-diagonal entries are largely **irreducible label ambiguity**
  (a 50/50 mixture has no single correct label), not a modelling failure;
- quote an accuracy against a **majority-class baseline**, and quantify run-to-run
  noise with a **seed-variance** experiment (differences smaller than the spread
  are noise).

**Key concepts:** image-level labels, logits/softmax, cross-entropy, global
average pooling, confusion matrix, class histogram, label ambiguity, seed
variance.

**Exercises.** Relabel the boundary scenes and watch the confusion move; add a
flip augmentation and explain the (small) gain.

**After this notebook:** you can build a classifier and, more importantly, judge
whether an accuracy number actually means something.

---

## 06 — Semantic segmentation (per-pixel labels)

**Role.** From one label per image to one label per pixel.

**Prerequisites.** Scenes plus pixel-aligned semantic masks (`mplsyn.segmentation`).

**By the end you can:**
- build paired **(image, semantic-mask)** data, with masks that are pixel-aligned
  to the image by construction;
- implement a compact **U-Net** with skip connections;
- train with **cross-entropy**, correctly using `ignore_index` for the unlabeled
  class 0;
- read **per-class IoU** and **mean IoU (mIoU)**, and use a pixel-level
  **confusion matrix** plus a class histogram to see *which* classes confuse
  (e.g. `rock` -> `gravel`, `road` <-> `built`);
- show why a **class-weighted loss** lifts the rare classes and raises mIoU, and
  why pixel accuracy alone hides rare-class failure.

**Key concepts:** semantic segmentation, softmax over classes per pixel,
`ignore_index`, IoU / mIoU, pixel confusion matrix, class imbalance.

**Exercises.** Find the lowest-IoU classes and relate them to class frequency and
to the weighted-loss comparison.

**After this notebook:** you can train and honestly evaluate a segmentation
model and diagnose its per-class behaviour.

---

## 07 — Cloud-occlusion segmentation (binary)

**Role.** A focused binary-segmentation task on the atmosphere itself.

**Prerequisites.** Semantic segmentation with U-Nets.

**By the end you can:**
- simulate cloud occlusion as a soft field, including a mix of opaque cores and
  semi-transparent veils;
- train a U-Net for **binary segmentation** of the occlusion mask;
- read **IoU, Dice and pixel accuracy**, and explain why accuracy misleads when
  clouds are the minority class;
- treat the 0.5 threshold as a **convention**: a threshold sweep shows the
  precision/recall operating point, and explain why Dice is usually preferred to
  accuracy for this target;
- check **calibration** with a reliability diagram and fix over/under-confidence
  with **temperature scaling** (ECE before/after), noting that calibration does
  not change accuracy.

**Key concepts:** binary segmentation, BCE-with-logits, sigmoid, IoU vs Dice vs
accuracy, operating point, calibration / ECE, temperature scaling.

**Exercises.** Why Dice over accuracy here; widen the coverage range and describe
the effect on IoU / Dice.

**After this notebook:** you understand overlap metrics, operating points and
probability calibration for a binary mask.

---

## 08 — Object detection (urban scenes)

**Role.** Localise *and* classify objects: an anchor-based, YOLO-style detector.

**Prerequisites.** Object boxes (generator notebooks 02-03), plus the
classification and segmentation ideas from 05-06.

**By the end you can:**
- encode objects as **anchor-based detection targets** (objectness, one-hot
  class, box), choosing anchors by **k-means in log space** over box shapes and
  reading the assignment visually;
- explain **why BCE-with-logits is used on the multi-class head** (binary CE
  applied independently per class; softmax CE is the mutually-exclusive
  alternative) and why the head supports multi-label targets;
- train a multi-task detector (objectness + class + box) and evaluate it with
  **mAP@0.5** (score threshold, per-class NMS, greedy matching), including
  per-class **precision-recall curves**, mAP at stricter IoU thresholds, and a
  **score-threshold sweep**;
- explain the detector's **error taxonomy** (localisation vs background vs
  duplicate vs class confusion) and read the **resolution ablation** (96 px vs
  64 px);
- separate unavoidable limits (a ~4 px car) from things a model change could fix.

**Key concepts:** anchors, objectness, box encoding/decoding, NMS, AP / mAP,
precision-recall, localization quality, resolution ceiling, error taxonomy.

**Exercises.** Vary the NMS threshold; raise the score threshold and watch
precision, recall and mAP; explain why the 64 px run loses the small classes.

**After this notebook:** you can build a small detector and you know how to read
mAP, PR curves and error breakdowns rather than a single number.

---

## 09 — Instance segmentation (trees in forests)

**Role.** Separate overlapping instances: count trees and segment each crown by
extending the detector of 08 with a mask head per anchor.

**Prerequisites.** Notebook 08 (anchor targets, objectness and box losses, NMS)
and the segmentation ideas from 06-07 (IoU, Dice).

**By the end you can:**
- recall the **anchor target encoding** and why box offsets are regressed directly;
- extend the detector with a **per-anchor mask head** (box-relative,
  crop-and-resize target) and train it with **objectness + box + mask BCE +
  soft Dice**;
- evaluate instance masks with **instance IoU**, **detection rate** and
  **count MAE / MAPE** against a mean-count baseline, stratified by canopy
  density;
- read **torchinfo** to see that the mask head dominates both the outputs and
  the parameters;
- separate unavoidable limits (a fully occluded crown) from things a model change
  could fix.

**Key concepts:** box-relative masks, mask head per anchor, soft Dice, instance
IoU, detection rate, count error, density stratification.

**Exercises.** Explain why Dice complements BCE; lower the score threshold and
watch count vs IoU; raise `MASK_SIZE` and weigh the accuracy gain against cost.

**After this notebook:** you can build a small instance-segmentation model and
read count and mask quality separately.

---

## Educational-value review (modeling notebooks)

### Strengths

- **Five tasks of increasing structure** (classification -> segmentation ->
  binary mask -> detection -> instances), each with a clear objective, exercises
  and self-check notes.
- **A read-out sits directly below every figure**, and each figure is designed to
  teach a specific idea (curves, confusion matrix + class histogram, PR curves,
  calibration, error taxonomy, resolution ablation).
- **Honest evaluation throughout**: three splits with a report-once test set;
  trivial baselines (majority class, most-frequent class, all-clear, no-object,
  mean-count); training/validation loss curves; per-class breakdowns; and explicit
  discussion of the irreducible ceiling (label ambiguity, clouds, resolution).
- **Reproducibility**: seeded training, deterministic splits and automatic device
  selection; a `FAST` flag gives a quick smoke run for notebook 08.
- **Metacognition**: the seed-variance figure and the "what to expect" notes teach
  learners to treat single numbers with the right skepticism.

### Gaps and recommendations

1. **Deepen the "why" notes in 07** (Dice vs IoU; temperature scaling leaves the
   argmax unchanged).
2. **Add prerequisites and a rough time budget** to each modeling notebook.
3. **A cross-notebook learning path** in the README (this file provides the
   content) so a reader knows 05 -> 06 -> 07 -> 08 -> 09 builds up.
4. **Consider a small "common pitfalls" box per notebook** (e.g. evaluating on
   the test set, comparing across seeds, reading mAP without PR curves).
5. **For 08, a dedicated vehicle/bus class** (or a truck/bus breakdown) would let
   learners measure the new vehicles rather than folding them into `car`.
