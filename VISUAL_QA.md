# Visual QA

Assessment of the images actually embedded in the three example notebooks
(`01_texture_generation`, `02_object_generation`, `03_scene_generation`), each
of which renders every material / object / scene with three seeds.

Method: extracted the embedded `image/png` outputs from the executed notebooks
and inspected them at native resolution, plus close-up renders of individual
materials and objects.

Levels under review:

- **L1 materials** (`mlpsyn.materials`) — low-level surfaces
- **L2 objects** (`mlpsyn.object`) — geometry filled with materials
- **L3 scenes** (`mlpsyn.scene`) — material ground + objects

## Overall verdict

| Area | Verdict | Notes |
|---|---|---|
| Material variety and grain | Mixed (Round 3) | most read well; `rock` still confetti-like, `sand` partly covered, `snow` flat |
| Material reuse (`fill_patch`) | Good | clipping works; fills now rotate with the object |
| Object readability | Mixed (Round 3) | building/car/pond/field good; `conifer` spiky, `rock` weak, tree/bush highlight flat |
| Scene layout (no axis prior) | Good | urban rotated/sheared, fields rotated, forest organic |
| View-angle consistency | Good | all point objects are plan view (iter 4/10) |
| Annotations | Good | `pixel_bbox` (top-left origin) aligns with `imshow` |
| 3-seed coverage | Good (Round 2) | scenes now vary composition (roads/rivers/woodland/parks), not just counts/rotation |
| Edge handling | Good | centres kept inside a margin; policy documented |
| Scene realism | Good (Round 2) | curvy roads, rivers, woodland, clearings, parks; grounds no longer flat |

## Round 2 — scene diversity and realism

Second inspection pass, again starting from the images embedded in
`notebooks/03_scene_generation.ipynb` (3 scenes × seeds 0/1/2), then the object
and material notebooks.

### Findings from notebook 03 (scenes)

**Urban**

- The street network is a near-perfect rotated/sheared **lattice**: every road
  has the same width and is effectively straight (the ±0.02 bow is invisible),
  so the block structure is too regular.
- Buildings are all similar in size and mostly grey; little roof/height variety.
- No green space (parks/squares); trees appear only as rare single cells.
- Cars are present but tiny and low-contrast, hard to spot.
- Seeds differ mainly in rotation and density, not in composition.

**Forest**

- The soil ground is a **flat, uniform brown** across the whole frame.
- Trees/conifers are scattered uniformly, with no clumping into stands, no
  clearings and no undergrowth; it does not read as a canopy.
- Conifers still read as dark **starbursts** rather than needle crowns.
- No paths/tracks and no water.

**Fields**

- Trees sit **on top of crop plots** (not on margins), which is the biggest
  realism defect; they should read as hedgerows along boundaries.
- The grass ground is flat and the plots form a regular, near-axis-aligned
  lattice; spacing/size are too uniform.
- The pond can sit very close to the frame edge.
- No rural road, river or adjacent woodland.

**Across seeds** — composition (presence of roads, water, woodland, parks) is
constant; only counts and rotation change. This is the main diversity gap.

### Added criteria

8. **Diversity across seeds** — seeds should vary *composition*, not just counts
   and orientation (e.g. presence/absence of a river, road, woodland patch,
   park).
9. **Realism / plausibility** — features should be placed where they make sense
   (trees on margins not crops, roads connecting, rivers meandering) and read at
   scene scale.

### Round 2 backlog

| # | Item | Type | Status |
|---|---|---|---|
| R1 | Trees land inside field plots → constrain vegetation to margins/hedgerows | defect | **Done (iter 14)** |
| R2 | Pond can sit at the frame edge → edge policy | defect | **Done (iter 14)** |
| R3 | Flat forest ground → ground variation, stands, clearings, undergrowth | defect/realism | **Done (iter 18)** |
| R4 | Conifer reads as a dark starburst → needle/canopy gradient | defect | **Done (iter 21)** |
| R5 | Perfect urban street lattice → curvy, warped streets | defect/realism | **Done (iter 15)** |
| R6 | Flat fields ground + regular mosaic → organic parcel variation | defect/realism | **Done (iter 22)** |
| R7 | **Curvy urban roads** (arterials + warped grid) | realism | **Done (iter 15)** |
| R8 | **Countryside roads**: rural road in fields, winding track in forest | realism | **Done (iter 16)** |
| R9 | **Rivers**: meandering linear water feature in some scenes | realism | **Done (iter 17)** |
| R10 | **Forest close to fields**: woodland patches/hedgerows in the fields scene | realism | **Done (iter 19)** |
| R11 | **Per-seed composition variants** (river/lake/road/woodland/park optional) | diversity | **Done (iter 19–20)** |
| R12 | Urban green space (parks/squares), roof/height variety, roundabout | realism | **Done (iter 20)** |
| R13 | Annotation + tests for the new linear features | correctness | **Partly (iter 23)** — object tests added; linear scene annotations deferred to masks |


## Evaluation criteria

For every future pass, judge each level against:

1. **Grain / material realism** — does the surface read as the intended material?
2. **Object readability** — is the semantic object recognisable at scene scale?
3. **No axis-aligned prior** — no layout *or texture* aligned to the image axes.
4. **View-angle consistency** — every object must use the same camera model.
   The project is top-down (plan view), so trees must be crowns seen from above
   (no trunks, no triangular elevation silhouettes), buildings must be roof
   footprints, etc. A side/elevation view is a defect.
5. **Annotation accuracy** — boxes align with the rendered geometry.
6. **Coverage** — three seeds should show meaningful variation.
7. **Material reuse** — objects and grounds share the material layer.

## Notebook 01 — materials (L1)

**Looks good**

- `grass`, `foliage`, `asphalt`, `concrete`, `soil`, `water`, `snow`, `bark`
  have convincing grain and tonal variation.
- `fill_patch` demo correctly clips roof/water/asphalt/crop into a rectangle,
  circle and triangle.
- 3-seed coverage: `grass`, `foliage`, `asphalt`, `soil`, `water`, `snow`,
  `bark`, `rock` vary; layouts/tones differ convincingly.

**Defects**

| # | Defect | Severity |
|---|---|---|
| M1 | `crop` is extremely regular: dense horizontal stripes, almost identical across seeds, strong axis-aligned banding / moiré | **Fixed (iter 5)** — plain grainy surface + soft bands |
| M2 | `rock` fragments are faint coloured specks (green/pink/purple) that read as confetti/litter, not rock; the angular fragments are barely visible | **Fixed (iter 6)** — larger outlined plates + crevice lines |
| M3 | `concrete` and `roof` are nearly identical across seeds (very low coverage); `roof` is a sterile perfect grid | **Fixed (iter 7)** — concrete gets random joints/potholes; roof switches shingles/seams |
| M4 | Tiny chromatic specks (red/green/blue dots) in `soil`, `grass`, `sand`, `water`, `bark` from the per-channel chroma term — visible as litter up close | **Fixed (iter 3)** |
| M5 | `sand` ripples are short straight scratches, not dune-like | **Fixed (iter 8)** — coherent parallel wavy ripples |
| M6 | Materials are not seamless-tileable (edges do not wrap) | Low |

## Notebook 02 — objects (L2)

**Looks good**

- `tree` and `conifer`: material-filled canopies read as foliage; size/shape
  vary across seeds.
- `pond`: organic shoreline, water material and sheen look natural.
- `building`: roof material + panels + rooftop units is plausible.
- `draw_road` with `curve` and asphalt fill looks good; shoulders + dashes read.
- Rotation is exposed and clearly demonstrated.
- Bounding boxes align with the objects.

**Defects**

| # | Defect | Severity |
|---|---|---|
| O1 | **Materials do not rotate with the object.** `fill_patch` clips an axis-aligned world-space material field, so a building or field plot rotated 20–55° shows axis-aligned roof grid / crop rows → crosshatch, moiré, and an axis-aligned texture prior | **Fixed (iter 2)** |
| O2 | **`draw_car` windshield placement bug**: window endpoints use absolute `cy` as the local y offset, so a stray blue line is drawn far from the car (clearly visible in the bbox demo) | **Fixed (iter 1)** |
| O3 | `rock` object is weak: mostly a brown angular blob + tan facet; the same-shape shadow still shows grey wedges | Medium |
| O4 | `bush` cluster is spread out; blobs detach from the main mass (seed 1 and 2) | **Fixed (iter 9)** — compact attached cluster |
| O5 | `car` reads as two stacked rectangles; cabin is as large as the body, no wheels, window line nearly invisible | **Fixed (iter 10)** — body + 4 wheels + smaller cabin + windshield |
| O6 | Building shadow is a hard offset rectangle that can look like a duplicate footprint | Low |
| O7 | **View angle**: `conifer` is a side elevation (trunk + upward triangles); `tree` also shows a trunk. Both are inconsistent with the top-down camera used by buildings/roads/fields/cars | **Fixed (iter 4)** — conifer is now a radial rosette, tree a trunkless crown; rosette still a little sparse |

## Notebook 03 — scenes (L3)

**Looks good**

- `urban`: rotated + sheared street grid, buildings aligned to the streets,
  covers the frame; no axis-aligned layout prior.
- `forest`: soil ground, organic tree placement, good variability across seeds.
- `fields`: rotated parcel mosaic with grass margins and a pond.
- Annotations: 16 boxes on the `urban` example align with buildings/trees/cars.
- Grounds are material textures; the land-cover reuse section shows
  `agricultural_fields` built on `soil` + `crop`.

**Defects**

| # | Defect | Severity |
|---|---|---|
| S1 | Crosshatch on rotated roofs and field plots, inherited from O1 (material field is axis-aligned) | **Fixed (iter 2)** |
| S2 | `urban` seed 0 has large empty concrete blocks; building footprints sometimes overlap roads; cars are barely visible | **Fixed (iter 11)** — denser grid (n=6–8), fewer tree cells, 3× cars; 27–36 objects/scene |
| S3 | `fields` plots overlap each other in places and the pond occasionally overlaps a parcel | **Fixed (iter 12)** — smaller rotation/size; pond placed by rejection sampling against field boxes (omitted if no gap) |
| S4 | Land-cover `agricultural_fields` crop-stripe banding is very strong (inherited from M1) | **Fixed (iter 5)** |
| S5 | Objects clipped at the frame edge are still annotated with a clipped box (e.g. a car at the top-left) — no documented policy | **Fixed (iter 13)** — centres kept inside a margin; policy documented |

## Cross-cutting root causes

1. **World-space material fills (O1/S1).** The single biggest issue: objects
   rotate, their texture does not. This also means rotated objects still carry
   an axis-aligned texture prior at pixel level, even though scene layout no
   longer does.
2. **`_color_noise` chroma bleed (M2/M4).** Independent per-channel jitter puts
   coloured specks on neutral materials (`rock`, `bark`, `soil`).
3. **`crop` material regularity (M1/S4).** One hard-coded row spacing repeated
   everywhere.
4. **Low-variance materials (M3).** `concrete`/`roof` need more distinct
   structure across seeds.
5. **Object construction shortcuts (O2/O3/O4/O5).** Car window bug; rock/bush
   assembled from a few raw blobs.

## Prioritised next steps

### P0 — correctness (small, high impact)

1. Fix `draw_car` windshield transform (use local offsets, rotate about cabin
   centre); add a regression test that no artist lands outside the car bbox.
2. Make material fills respect object rotation: apply the object's rotation to
   the `imshow` transform (or render the fill in the object's local frame), so
   roof/field textures rotate with the shape.
3. Reduce chroma in `_color_noise` (luminance-dominant, small or zero chroma) to
   remove coloured specks on neutral materials.

### P1 — material quality

4. Rework `crop`: randomized row spacing/orientation and less contrast; or make
   `crop` a plain grainy surface and let `field_plot` own the rows.
5. Add real structure to `rock` (contrast-boosted angular facets, less speckle).
6. Diversify `concrete` (cracks/joints) and `roof` (tiles/seams, colour variety)
   across seeds.
7. Give `sand` smooth, coherent sinusoidal ripples.

### P2 — object and scene quality

8. Tighten `bush` (smaller cluster spread; keep blobs attached).
9. Improve `car` proportions (smaller cabin, add wheels) and make it visible at
   scene scale.
10. `urban`: add cross-street variety, reduce empty blocks, keep buildings off
    roads, make cars more visible.
11. `fields`: resolve plot/pond overlaps; add a small overlap-avoidance pass.
12. Define an edge policy: drop objects whose visible fraction is below a
    threshold, or document that boxes are clipped.
13. Optional: seamless tiling for materials; pixel-level segmentation masks
    (roadmap milestone 4).

## Suggested acceptance checks for the next iteration

- No axis-aligned grid or rows visible on any rotated object at 3 seeds.
- No stray artists outside an object's bbox (car regression test).
- Neutral materials contain no saturated colour specks at 1:1 zoom.
- `crop`/`roof`/`concrete` visibly differ across 3 seeds.
- All object bboxes still align with `imshow` (top-left origin).
- Every object reads as top-down (no trunks or vertical silhouettes).

## Progress log

Items are executed one at a time. Each entry lists the fix, the visual result,
and any newly discovered issue. Previously-done items are re-checked for
regressions at every subsequent pass.

**Final status (13 iterations):** all P0 and P1 items plus the main P2 items are
done. Each iteration was rendered and inspected; a final regression pass over
all materials, objects and scenes (3 seeds) showed no regressions in the
previously completed items. All three notebooks were regenerated with the final
code and 92 tests pass.

| # | Item | Status | Notes |
|---|---|---|---|
| 0 | Add view-angle criterion + progress log | done | baseline report |
| 1 | Fix `draw_car` windshield bug | done | window now uses local cabin offset; verified at rotations 0/37/90. Regression test `test_car_artists_stay_within_bbox` added. Car shape still crude (item 10) |
| 2 | Rotation-aware material fills | done | `fill_patch(rotation=, center=)` applies an `Affine2D` rotation to the fill; roof grid and crop rows now rotate with building/plot (verified 0/25/55°). No crosshatch |
| 3 | Reduce `_color_noise` chroma | done | chroma term cut from 0.25·spread to 0.05·spread; neutral materials now brownish only (max saturation ≈0.36 on rock) |
| 4 | Top-down conifer / remove side-view trunks | done | conifer rewritten as a two-layer radial rosette (9–12 tines, wider after follow-up); tree trunk replaced by a dark crown centre. All point objects now top-down |
| 5 | Rework `crop` material | done | replaced dense hard rows with grain + soft bands; field rows now subtle and rotate with the plot, no banding |
| 6 | `rock` material structure | done | larger 5-gon plates with edge outlines + dark crevice lines; sampled max-saturation pixel is brown (84,66,49) |
| 7 | Diversify `concrete` / `roof` | done | concrete: random-angle expansion joints + potholes; roof: 3 styles (panel grid / seams / shingles) with per-seed angle. Max saturation <0.12 (greys) |
| 8 | Coherent `sand` ripples | done | parallel sinusoidal ripples with per-seed orientation; amplitude softened after seed 2 showed zig-zag |
| 9 | Tighten `bush` cluster | done | satellite blobs placed on a ring at 0.05–0.14·size so they always overlap the main mass; sand amplitude softened |
| 10 | `car` proportions + wheels | done | top-down body, four wheels peeking out, smaller cabin, windshield; verified 0/37/90° and bbox test still passes |
| 11 | `urban` empty blocks / car visibility | done | n 5–7→6–8, tree probability 0.18→0.12, cars n*2→n*3; empty blocks filled. Later edge policy trims truncated cars |
| 12 | `fields` plot / pond overlap | done | field size 0.9→0.86, rotation ±18→±15; pond rejected if it overlaps a field box. Minor: trees may still fall on plots (hedgerow reading) |
| 13 | Edge policy for clipped objects | done | car centres must be inside [0.05,0.95]; forest margin 0.06→0.08; documented in `scene.py`. Verified fields/forest have 0 clipped boxes, urban only corner buildings |

## Remaining (not addressed this round)

- **O3** `rock` object still reads as a brown angular blob + facet; the shadow
  can show grey wedges. Material is improved, object geometry is not.
- **O6** Building shadow is a hard offset rectangle; could be softened/feathered.
- **M6** Materials are not seamless-tileable (edges do not wrap).
- **Satellite backgrounds** `urban_buildings` and `road_network` remain
  axis-aligned (they are backgrounds, not composed scenes).
- **Trees on field plots** (hedgerow reading) are allowed but not intentional.
- **Conifer rosette** is stylised; a needle-density gradient would read better.
- Roadmap items: pixel-level segmentation masks, COCO/YOLO export, train/val
  splits.

## Round 2 progress log

Each item is a code change, a re-render of the three scenes (seeds 0–2) and an
inspection; the object/linear features were also checked close-up. Done items
are re-checked for regressions on every later render.

| # | Item | Status | Notes |
|---|---|---|---|
| 14 | R1/R2: hedgerow trees + pond edge | done | `_scatter_outside` rejects tree centres inside plot boxes (pad 0.015) so trees sit on margins; pond placed with a 0.04 frame margin. Trees no longer sit on crops |
| 15 | R5/R7: curvy urban roads | done | `_meander_points` + generalised `draw_road(points=, meander=)`; the grid is now warped and a curved arterial crosses ~70% of scenes. Lattice no longer perfect |
| 16 | R8: countryside roads | done | `fields` gets a meandering rural road (~60%), `forest` a winding track (~60%, no dashes). Both are drawn as explicit polylines |
| 17 | R9: rivers | done | new `draw_river` (bank ribbon + material-filled water + sheen); included in `forest` (~55%) and `fields` (~45%), meandering |
| 18 | R3: forest ground + stands | done | `_ground_patches` breaks up the flat soil with grass/soil clearings; `_forest_patch` clumps trees into woodland stands |
| 19 | R10/R11: woodland near fields + variants | done | fields places a woodland patch on a margin (~50%); optional features (road/river/woodland/pond) are now per-seed |
| 20 | R12: urban parks/roundabout/variety | done | green park blocks (~60%), a roundabout island (~45%), 12 roof colours and larger size range; cars 0.03–0.046 |
| 21 | R4: conifer crown | done | base canopy + 11–15 tines + mid-green centre; no longer a sparse dark starburst, reads as a needle crown at scene scale |
| 22 | R6: organic field mosaic | done | whole-mosaic rotation ±22°, per-plot rotation ±16°, size 0.72–0.92, 12% fallow cells |
| 23 | R13: tests | done | 4 new tests: road meander/polyline bbox, river bbox, `_meander_points` endpoints, fields trees not inside plots. 96 tests pass, ruff clean |
| 24 | Fields corridor + conifer tuning | done | linear features drawn *first* and field cells whose centre falls in the road/river corridor are skipped, so roads/rivers no longer cut across plots; pond rejects the corridor; conifer base shrunk so tines protrude |
| 25 | Regression + notebooks | done | all three notebooks re-executed with the final code; the scenes coverage now uses per-scene seed sets (urban [0,3,9], forest [0,2,5], fields [0,3,9]) so every row shows its road/river/woodland variants; final montage re-inspected |

### Round 2 verdict

- **Urban** — warped/curvy streets, a curved arterial, parks, a roundabout,
  varied roofs and visible cars. No longer a perfect lattice.
- **Forest** — clearings and ground variation, clustered stands, winding tracks
  and meandering streams. Reads as woodland rather than evenly scattered trees.
- **Fields** — hedgerow trees on margins, organic rotated mosaic, rural roads
  and rivers routed through corridors between plots, woodland patches and a
  pond. Trees no longer sit on crops.
- **Diversity** — composition now varies per seed (road/river/woodland/pond/
  park presence), not just counts and rotation.
- **Regression** — all Round 1 fixes still hold; 96 tests pass; ruff clean.

### Round 2 remaining

- **Field plot annotations** are the plot rectangles only; roads, rivers and
  woodland patches are not in `Scene.objects`. Pixel-level masks (roadmap) will
  cover linear features.
- **Rivers can still touch a plot corner** occasionally; the corridor skip uses
  centre distance, not full rectangle intersection.
- **O3/O6/M6** and satellite background alignment from Round 1 remain open.

## Round 3 — 5-instance grid inspection

The three notebooks now render **five instances in a row** (seeds 0–4) for every
material, object and scene. All resulting grids were inspected:

- materials: three figures of 4 materials × 5 seeds
  (`MATERIALS` gallery + coverage)
- objects: two figures of 4 objects × 5 seeds
- scenes: one figure of 3 scenes × 5 seeds

### Findings

**Materials**

- `rock` **still reads as coloured confetti**: tan ground with scattered small
  pink/green/tan polygon "leaves" plus a few long straight crevice lines. It
  does not read as a rock surface. (Round 1 M2 was only a partial fix.)
- `sand` ripples cover only part of the tile for some seeds (seed 1: left ~60%,
  seed 2: upper-left quadrant) leaving a plain region with a hard boundary.
- `snow` is very flat: near-white with sparse blue dashes; little relief.
- `asphalt` cracks are long straight scratches that reach the tile edge.
- `grass`, `foliage`, `soil`, `water`, `crop`, `concrete`, `bark`, `roof` all
  read well and vary plausibly across the five seeds.

**Objects**

- `conifer` is a **dark spiky starburst** standalone: long thin tines radiate
  from a small canopy, reading as a black urchin rather than a fir crown. (The
  Round 2 R4 fix improved the scene-scale look but not the object.)
- `tree` and `bush` have one large flat, bright highlight blob that reads as a
  blemish on the canopy.
- `rock` is weak: an angular polygon with a large flat tan facet and grey
  shadow wedges; the silhouette looks like folded paper, not a boulder.
- `field_plot` showed an extra rotated inner outline / diagonal line inside the
  plot. The rotation demo confirmed the root cause: **the material fill was not
  clipped by the rotated footprint**. `fill_patch` rotated the texture image but
  `image.set_clip_path(patch)` ignored the patch's rotation, so the fill spilled
  outside the footprint (an axis-aligned octagon) while the outline marked the
  true rotated rectangle. This affected every rotated material-filled object.
  **Fixed in Round 4** (`TransformedPatchPath`).
- `pond` sheen is a hard-edged light polygon that looks like a floating sheet.
- `building` reads well (roof material, panels, rotation, rooftop units); its
  shadow is still a hard offset rectangle.
- `car` reads well and varies in colour/orientation.

**Scenes**

- `urban` — warped streets, parks, roundabout and cars are present, but
  buildings are still fairly uniform in colour/size and there is little green
  space; some buildings still overlap roads.
- `forest` — clearings and stands read well, but `_ground_patches` have hard
  polygon edges (they look like flat paint splotches) and density varies a lot
  between seeds (seed 1 is sparse).
- `fields` — mosaic, hedgerows, roads/rivers and ponds read well, but some
  plots look empty because the crop rows are too low-contrast on green/soil
  fills; a pond sits close to the frame edge at seed 4.

### Next steps (reset after Round 3)

Prioritised; each item is one code change + re-render + inspection.

**P0 — clear defects**

1. ~~**`fill_patch` rotation clip**~~ — **DONE (Round 4 iter 1)**:
   `TransformedPatchPath(patch)`; fills now stay inside rotated footprints
   (verified at 0/20/55°).
2. ~~**`rock` material**~~ — **DONE (Round 4 iter 2)**: overlapping grey/brown
   slabs with lit/shaded facets; no more confetti.
3. ~~**`rock` object**~~ — **DONE (Round 4 iter 3)**: rounded boulder silhouette,
   lit facet + shaded side, subtle outline; no grey wedges.
4. ~~**`conifer` object**~~ — **DONE (Round 4 iter 4)**: three overlapping jagged
   foliage rings form a dense fir crown; no more starburst.

**P1 — realism**

5. ~~`tree`/`bush`~~ — **DONE (Round 4 iter 5)**: single big highlight replaced by
   several small foliage highlights; canopies read as mottled foliage.
6. ~~`sand`~~ — **DONE (Round 4 iter 6)**: ripple span extended (`t` −1.2…1.8) so
   ripples cover the whole tile at every seed.
7. ~~`field_plot` row contrast~~ — **DONE (Round 4 iter 7)**: row colour is now a
   darkened version of the fill colour (×0.55) and rows are clipped with
   `TransformedPatchPath`; rows are visible on crop/soil/grass fills at all
   rotations.
8. ~~`forest` ground patches / density~~ — **DONE (Round 4 iter 8)**:
   `_ground_patches` builds each clearing from three nested low-alpha fills
   (soft edge); forest base density raised (`size²/4600`, 45–120) and stands
   increased to 4–7, so counts are consistent (77–86 at 600px).
9. ~~`urban` building variety / greenery~~ — **DONE (Round 4 iter 9)**: building
   size range widened (0.32–0.62·step), tree probability 0.12→0.16 and parks
   0.75 probability with 1–3 blocks; more size/colour variety and greenery.

**P2 — polish**

10. ~~`pond` sheen~~ **DONE (iter 10)**; ~~building shadow~~ **DONE (iter 11)**;
    ~~`snow` relief~~ **DONE (iter 12)**; ~~`asphalt` crack shape~~ **DONE
    (iter 12)**. P2 polish is complete.
11. Linear-feature scene annotations / pixel masks (roadmap), seamless
    materials, satellite background alignment.

### Round 4 — iteration log

One item per iteration; notebooks re-executed and analyses refreshed each time.

| Iter | Item | Status | Result |
|---|---|---|---|
| 1 | P0#1 `fill_patch` rotated clip | done | `image.set_clip_path(TransformedPatchPath(patch))`. Re-rendered the rotation demo: fill, outline and bbox coincide at 0/20/55° with no spill. Buildings/plots in scenes are clean. 96 tests pass, ruff clean; notebooks re-executed. |
| 2 | P0#2 `rock` material | done | Rebuilt as overlapping angular slabs (`~size²/500`) with lit/shaded facets and no long scratch lines. The 5-seed grid reads as a rocky face, not confetti. |
| 3 | P0#3 `rock` object | done | Rounded boulder silhouette with a lit facet, a shaded side and a subtle outline; the 5-seed object grid reads as a boulder on a shadow. |
| 4 | P0#4 `conifer` | done | Replaced thin tines with three overlapping jagged star rings (14–20 vertices, inner/outer radii); reads as a dense fir crown from above. Rock/conifer analyses refreshed in the notebooks. |
| 5 | P1#5 `tree`/`bush` highlights | done | Removed the single large bright highlight; added 2–5 small highlights and shrank the satellite canopy blobs, so canopies read as mottled foliage. |
| 6 | P1#6 `sand` coverage | done | Extended the ripple span (`t` −1.2…1.8) so the whole tile is covered at every seed; the partial-coverage boundary is gone. |
| 7 | P1#7 `field_plot` rows | done | Row colour is a darkened version of the fill (×0.55) and rows are clipped with `TransformedPatchPath`, so rows are visible on crop/soil/grass fills and stay inside rotated plots. |
| 8 | P1#8 forest clearings/density | done | `_ground_patches` now builds each clearing from 3 nested low-alpha fills (soft edge); forest density raised to `size²/4600` (45–120) with 4–7 stands, giving consistent counts (77–86 at 600px). |
| 9 | P1#9 urban variety/greenery | done | Building sizes 0.32–0.62·step, tree probability 0.12→0.16, parks 1–3 blocks at 0.75 probability. |
| 10 | P2 pond sheen | done | Sheen replaced by 2–3 soft translucent ellipses (alpha 0.35) instead of one hard-edged polygon. |
| 11 | P2 building shadow | done | Two-layer offset shadow (scale 1.0/1.07, alpha 0.24/0.14) softens the previous hard rectangle. |
| 12 | P2 snow relief + asphalt cracks | done | Added a second low-alpha blue-grey mottle to `snow` for relief; `asphalt` cracks are now short bent random-walk lines (`_cracks`) instead of edge-long scratches. |

P0/P1/P2 complete (only the roadmap items remain: linear-feature masks, seamless
materials, satellite alignment). A near-full notebook regeneration was run after
each iteration; 96 tests pass; ruff clean.

### Round 2 remaining (superseded by the list above)

- Field plot annotations cover plots only; roads/rivers/woodland are not in
  `Scene.objects` (see P2).
- Rivers can occasionally touch a plot corner (corridor uses centre distance).
- **O3/O6/M6** and satellite background alignment are folded into P0–P2 above.
## Round 5 — realism, atmosphere and vocabulary

All P0/P1/P2 defects from Rounds 1–4 are fixed. Work now shifts to the AGENTS
"step back" goals: richer object/material/scene vocabulary and atmospheric /
sensor realism.

### Backlog

| # | Item | Type | Status |
|---|---|---|---|
| R5-1 | Sensor + atmosphere module: Gaussian/Poisson noise, blur, haze, clouds (vignette/JPEG removed — camera, not satellite) | realism | **Done** |
| R5-2 | New objects: silo, pool, truck, hedgerow, boat | vocabulary | **Done** |
| R5-3 | New materials: gravel, mud | vocabulary | **Done** |
| R5-4 | New scenes: coastal, mountain, arid, and boundary scenes `forest_fields` / `urban_fringe` | vocabulary | **Done** |
| R5-5 | Grids use 10 seeds; image-size check (`check_figure_size`) keeps every grid < 5 MB | quality | **Done** |
| R5-6 | Random-scene diversity notebook (25 scenes, mixed noise, >=70 % clear) | diversity | **Done** |
| R5-7 | Roadmap updated for the above | docs | **Done** |
| R5-8 | Soft/organic region boundaries in `forest_fields` / `urban_fringe` | defect | **Done** |
| R5-9 | Mountain ridge/elevation shading | defect | **Done** |
| R5-10 | Arid wadi as a continuous channel | defect | **Done** |
| R5-11 | Per-seed global parameters + shadow consistency | diversity | pending |
| R5-12 | Annotation export (COCO/YOLO) + train/val split | tooling | pending |
| R5-13 | Pixel-level segmentation masks | tooling | pending |

### R5-1 — sensor and atmospheric effects

New `mlpsyn.sensor` module applied to the final image; the scenes notebook
renders an 8-effect × 5-seed grid (`EFFECTS`).

**Looks good**: every effect runs, is deterministic per seed and composable via
`apply_sensor`. Poisson shot noise, blur, haze and clouds are convincing; clouds
include an offset ground shadow; haze has a natural top-to-bottom gradient.

**Defects**: Gaussian noise is subtle at the default sigma (low); the clouds are
quite opaque/white in the densest patches and a little blobby (low); blur at the
default radius is fairly strong (low).


### Round 5 progress

| Iter | Item | Status | Result |
|---|---|---|---|
| 1 | Sensor + atmosphere module | done | `mlpsyn.sensor` (gaussian, poisson, blur, haze, clouds, `apply_sensor`, `EFFECTS`); vignette/JPEG removed as camera artefacts. |
| 2 | New objects | done | silo, pool, truck, hedgerow (+ boat for the coast); integrated into scenes. |
| 3 | New materials | done | gravel, mud (14 materials total). |
| 4 | New scenes | done | coastal, mountain, arid, forest_fields, urban_fringe (8 scene types). |
| 5 | 10-seed grids + size checks | done | all coverage grids use 10 seeds; `check_figure_size` prints and asserts <5 MB. Scene coverage grid 1.10 MB, sensor 0.65 MB, random scenes 1.39 MB. |
| 6 | Diversity notebook | done | `notebooks/04_scene_diversity.ipynb`: 25 random scenes, mixed noise/haze, 7/25 cloudy (72 % clear). |
| 7 | Roadmap | done | new milestone 9 (Visual QA & diversity) and updated satellite/objects/materials lists. |
| 8 | R5-8 soft/organic region boundaries | done | New `_organic_edge` (extra smoothed high-frequency wobble) and `_soft_edge` (three nested, increasingly wide bands of the region material drawn on the neighbour side with alpha 0.45/0.25/0.10, sharing one wobble field so they nest). Applied to both `forest_fields` (soil↔grass) and `urban_fringe` (concrete↔grass). Inspection: the hard line is replaced by a mottled transition; no contour banding. 125 tests pass, ruff clean (src/tests); all four notebooks re-executed. |
| 9 | R5-9 mountain elevation shading | done | Replaced the faint ridge-line polygons with `_elevation_field` + `_hillshade`: a smooth random height field (summed sinusoids + Gaussian hills) shaded by a Lambertian light, overlaid as a translucent white/black RGBA image. The 10-seed grid now reads as ridges/valleys rather than a flat plateau; shading is per-seed (light azimuth varies). 125 tests pass; notebooks re-executed. |

**R5-8 — boundary scenes, inspection notes**

- `forest_fields` seeds 0–4: the soil/grass edge is now a soft mixed band; trees
  sit in the transition, which reads as woodland thinning into farmland.
- `urban_fringe` seeds 0–4: concrete fades into grass over a mottled band
  (seed 0 shows it clearly where no river splits the frame); the band nests
  smoothly, with no stepped contour lines.
- Diversity across seeds is good: boundary shape, river presence and field
  layout all vary.

**R5-9 — mountain, inspection notes**

- The 10-seed mountain row shows broad lit/shaded slopes with soft transitions;
  no hard banding. Snow patches and green scrub sit on the relief. Ponds are
  occasional, as before. The relief is somewhat soft/blurry at scene scale
  (low-severity, could be sharpened with more octaves).

**Next (R5-10 defect):** the arid wadi is a chain of grey patches; make it a
continuous gravel channel. Then R5-11 per-seed global parameters and the tooling
items (COCO/YOLO export, segmentation masks). *(R5-10 done in Round 6, iter 14.)*

## Round 6 — sensor ordering, bounded noise and agricultural layout

Two user-requested workstreams; see the Round 6 plan in `ROADMAP.md`.

### Backlog

| # | Item | Type | Status |
|---|---|---|---|
| R6-1 | Atmosphere (haze, clouds) applied before sensor noise/blur in one pipeline | correctness | **Done** |
| R6-2 | Consistent, bounded noise on every image (no random off, hard caps) | quality | **Done** |
| R6-3 | Voronoi field mosaic: fields tile the ground, non-rectangular boundaries | realism | **Done** |
| R6-4 | Field boundaries are roads/dirt tracks/hedgerows/rivers; every field has frontage | realism | **Done** |
| R6-5 | Same mosaic logic in `forest_fields` / `urban_fringe` | realism | **Done** |
| R6-6 | Tests + notebook/QA updates | tooling | **Done** |
| R6-7 | Keep every notebook cell under 10 s (5-seed grids, material cache, fixed autoscale) | quality | **Done** |
| R5-10 | Arid wadi as a continuous channel | defect | **Done** |
| R5-11 | Per-seed global parameters + shadow consistency | diversity | pending |
| R5-12 | Annotation export (COCO/YOLO) + train/val split | tooling | pending |
| R5-13 | Pixel-level segmentation masks | tooling | pending |

### Round 6 progress

| Iter | Item | Status | Result |
|---|---|---|---|
| 10 | R6-1/R6-2 sensor order + bounded noise | done | `apply_sensor` now owns the chain atmosphere → blur → noise and accepts `haze_strength`/`cloud_coverage`; `atmosphere()` composes haze then clouds. Noise defaults are fixed and clamped (`GAUSSIAN_SIGMA=8`, `MAX_GAUSSIAN_SIGMA=14`, `POISSON_SCALE=600`, `MIN_POISSON_SCALE=220`). Notebook 04 applies one mild noise to every frame. Tests: order, mild/bounded, comparable Gaussian/Poisson strength, clamping. |
| 11 | R6-3/R6-4 fields mosaic | done | `fields_scene` rebuilt: jittered sites → `_voronoi_cells` (half-plane clipped) tile the frame; each cell is crop/soil/fallow/woodland with polygon crop rows. A Kruskal spanning tree over shared edges is drawn as country roads / gravel dirt tracks (every field gets frontage); hedgerows on other edges; a river crosses; silos along roads; trees in woodland. |
| 12 | R6-5 boundary scenes | done | `_field_mosaic` reused for the farmland half of `forest_fields` and `urban_fringe`; their rectangle grids are gone. |
| 13 | R6-6/R6-7 tests, speed, docs | done | New tests for mosaic tiling, Voronoi area, spanning-tree frontage, sensor order and bounded noise (131 pass). Material cache + fixed autoscale; coverage grids use 5 columns (AGENTS) and every cell runs < 10 s. |
| 14 | R5-10 continuous wadi | done | The arid wadi is now a single continuous channel: a pale bank ribbon, a full-opacity gravel bed and a darker thalweg line, instead of a low-alpha gravel ribbon that read as grey patches. Close-up confirms an unbroken dry riverbed across the frame. |

### Round 6 inspection notes

- **fields (5 seeds)** — parcels tile the ground with irregular Voronoi
  boundaries; dark country roads and tan dirt tracks run along parcel edges and
  connect; hedgerows, woodland blocks and a river bound the other sides. No
  rectangle-in-grass remains. Crop-row direction varies per parcel. The road
  shoulder was removed for field roads (it read as a grey ribbon); roads are now
  clean ribbons.
- **forest_fields / urban_fringe** — farmland halves are now irregular mosaics
  with road frontage, not rectangle grids; the soft region boundary is
  preserved.
- **sensor grid** — every diversity frame carries a mild, comparable noise
  (Gaussian σ=8 or Poisson full-well 600); haze and clouds are visibly applied
  under the grain. No frame is over-noised.
- **Speed** — nb01 4.4 s, nb02 3.1 s, nb03 19.2 s total with the largest cell
  (5-seed scene coverage) at 6.2 s, nb04 14.8 s with cells at ~6.3/6.6 s; all
  grids < 5 MB.
- **arid (5 seeds)** — the wadi (seed 0) is an unbroken gravel channel with a
  pale bank; the gravel stones are fairly large at high resolution (low
  severity, general material scale).

**Next:** R5-11 per-seed global parameters, then the tooling items.

## Round 7 — urban correctness/amenities, wild/pasture and elevation

User-requested realism items; see the Round 7 plan in `ROADMAP.md`.

### Backlog

| # | Item | Type | Status |
|---|---|---|---|
| R7-1 | Roads lead somewhere; buildings never cross roads | correctness | **Done** |
| R7-2 | Parking lots, plazas and more parks in urban settings | realism | **Done** |
| R7-3 | Dense-forest (`wilderness`) and pasture/meadow scenes | vocabulary | **Done** |
| R7-4 | Shared per-seed elevation map (`_terrain`) as a scene input | architecture | **Done** |
| R7-5 | Layout consistent with elevation (water low, snow high, fields flat) | realism | **Partly done** |
| R7-6 | Contour-following / gentle-grade roads | realism | **Done** |
| R7-7 | Orchard and greenhouse vocabulary | vocabulary | **Done** |
| R5-12 | Annotation export (COCO/YOLO) + train/val split | tooling | **Done** |
| R7-8 | Automatic diversity metrics (`mlpsyn.metrics`) | tooling | **Done** |
| R5-13 | Pixel-level segmentation masks | tooling | pending |

### Round 7 progress

| Iter | Item | Status | Result |
|---|---|---|---|
| 15 | R7-1 buildings off roads | done | Urban tracks every road as a `(centreline, width)` corridor; each building/pool is sized from `_road_clearance` at its block centre, so footprints never overlap a road. The grid + arterial all span the frame, so no road dead-ends inside. |
| 16 | R7-2 parking / plaza / parks | done | New `draw_parking_lot` (asphalt pad, stall lines, parked cars) and `draw_plaza` (paved square with fountain or planted centre and corner trees); urban now reserves 2–4 park blocks and 1–2 parking plus 0–1 plaza per seed. |
| 17 | R7-3 wilderness + pasture | done | `wilderness_scene` (road-less continuous canopy, clearings, valley stream) and `pasture_scene` (meadow, hedgerows, scattered trees, gravel farm track, farmstead, pond). `SCENES` now has 10 types. |
| 18 | R7-4/R7-5 elevation map | done | `_terrain(rng) -> (height, slope, hillshade)` with a fixed upper-left light; `_relief_overlay` and `_low_valley_path`. Mountain snow/rocks/water, fields crop/woodland/fallow, and forest/wilderness/pasture/fields relief all derive from the same per-seed map. |
| 19 | R7-6 gentle-grade roads | done | `_spanning_tree` accepts a `weight`; the field and boundary mosaics build a minimum spanning tree weighted by boundary height difference. Forest and pasture tracks use `_gentle_path` (lowest-slope of several candidate meanders). New test covers weighted MST, `_terrain`, `_low_valley_path` and `_gentle_path`. |
| 20 | R7-7 orchard + greenhouse | done | New `draw_orchard` (grid of small trees on a soil block) and `draw_greenhouse` (row of pale polytunnels); registered in `OBJECTS`. Every `fields` scene now has one orchard on its flattest cell; `pasture` sometimes gets greenhouses. Tests parametrised over `OBJECTS` now cover both (142 pass). |
| 21 | R5-12 COCO/YOLO export + split | done | New `mlpsyn.annotations`: `category_index`, `yolo_lines`, `scene_to_coco`, `coco_dataset` (pixel top-left boxes) and `train_val_test_split` (deterministic). Shown in nb03; 7 new tests (149 pass). |
| 22 | R7-8 diversity metrics | done | New `mlpsyn.metrics`: `label_histogram`, `objects_per_scene`, `spatial_coverage`, `diversity_report`. nb04 prints a 30-scene report (16 labels present, entropy 1.8, spatial coverage 0.71); 4 new tests (153 pass). |

### Round 7 inspection notes

- **urban (5 seeds)** — parking lots (stalls + cars), plazas (fountain/planting)
  and park blocks now fill the former empty concrete; buildings sit in the
  blocks and no longer cross streets. Some blocks remain plain concrete
  (low severity); the arterial still crosses at an angle (fine).
- **wilderness (5 seeds)** — dense, near-continuous canopy with occasional
  clearings and a stream; clearly distinct from `forest`.
- **pasture (5 seeds)** — rolling meadow with a soft relief overlay, hedgerows,
  tree clumps, a gravel track, a farmstead, ponds and occasional rows of
  greenhouses; reads as farmland.
- **fields orchard / greenhouse** — the guaranteed orchard sits on the flattest
  cell and reads as a planted block; in `pasture` the occasional greenhouse rows
  read as pale polytunnels.
- **elevation consistency** — in mountain, snow prefers high ground and ponds
  the low basins; rivers in forest/wilderness/fields follow the lowest of
  several candidate meanders; forest/pasture tracks follow the gentlest ground
  and field roads use a minimum spanning tree weighted by boundary height
  difference; fields turn to woodland on steep cells and meadow on low cells.
- **Speed** — scene coverage cell now 7.6 s raw (10 types × 5 seeds) after
  trimming the wilderness/forest density floors; still < 10 s in the notebook.

**Next:** the tooling items (COCO/YOLO export, segmentation masks) and further
vocabulary enrichment.

## Round 8 — roads, segmentation masks and training notebooks

User-requested follow-up.

### Backlog

| # | Item | Type | Status |
|---|---|---|---|
| R8-1 | No dead-end roads in the field/boundary mosaic | defect | **Done** |
| R8-2 | Pixel semantic-segmentation masks | tooling | **Done** |
| R8-3 | Cloud-occlusion mask | tooling | **Done** |
| R8-4 | nb04 rows: original / boxes / semantic labels / cloud mask | tooling | **Done** |
| R8-5 | Three PyTorch training notebooks (MPS, <10 min each) | tooling | **Done** |

### Round 8 progress

| Iter | Item | Status | Result |
|---|---|---|---|
| 23 | R8-1 close the road network | done | `_close_network` raises every degree-1 cell to degree >= 2 where an unused edge exists; fields/boundary roads no longer dead-end mid-field. Test added. |
| 24 | R8-2 semantic segmentation | done | `mlpsyn._mode` mask flag; `_compose` renders flat class colours in mask mode; `mlpsyn.segmentation` maps pixels to 13 classes (0 = unlabeled). `semantic_mask(name, size, seed)`. 5 tests. |
| 25 | R8-3 cloud mask | done | `cloud_alpha` + `clouds(..., return_mask=True)` expose the occlusion field. Test added. |
| 26 | R8-4 notebook section | done | nb04: 4-row figure (original, +boxes, semantic labels, cloud mask) for 5 scenes; pixel-aligned. |
| 27 | R8-5 training notebooks | done | see metrics below. |

### Round 8 inspection / metrics

- **nb04 semantic/cloud rows** — labels match the RGB geometry: cropland/bare
  soil/grassland/vegetation/water/built/road classes are clean; thin details and
  shadows fall into the *unlabeled* class. Cloud masks are soft blobs that match
  the applied occlusion.
- **nb05 object detection** (class-agnostic CenterNet-lite, 312 train): validation
  precision 0.59, recall 0.52, F1 0.55, matched-box mIoU 0.62; ~100 s total.
- **nb06 semantic segmentation** (small U-Net): pixel accuracy 0.81, mean IoU
  0.56 (best classes: vegetation 0.91, sand 0.85; rare classes snow/gravel/
  wetland are near zero); ~100 s total.
- **nb07 cloud occlusion** (small U-Net): IoU 0.94, Dice 0.97, pixel accuracy
  0.99; ~65 s total.
- All three run on Apple **MPS** and finish well under 10 minutes.

**Next:** improve rare-class segmentation (snow/gravel/wetland), raise detection
recall, and add instance masks / panoptic targets.

## Round 9 — classification, multi-class detection, segmentation legend

| # | Item | Type | Status |
|---|---|---|---|
| R9-1 | Scene-classification notebook (7 classes, ambiguous scenes) | tooling | **Done** |
| R9-2 | Multi-class detection (6 classes) + mAP@0.5 IoU | tooling | **Done** |
| R9-3 | Semantic-segmentation class colour legend | tooling | **Done** |

| Iter | Item | Status | Result |
|---|---|---|---|
| 28 | R9-1 `08_train_scene_classification` | done | Balanced dataset over 7 classes; `forest_fields`->cropland and `urban_fringe`->urban as documented ambiguous cases. Small CNN; validation accuracy 0.93; per-class accuracy + confusion matrix. Ambiguity shows up as cropland/forest/pasture confusions. |
| 29 | R9-2 reworked `05_train_object_detection` | done | 6 classes; per-class heatmap head; **mAP@0.5 = 0.35** (rock 0.46, water 0.48, boat 0.45, building 0.37, vegetation 0.24, field 0.10); prediction figure has a per-class colour legend. |
| 30 | R9-3 segmentation legend + seed | done | nb06 shows a class colour legend (0 = unlabeled); seeded training gives pixel accuracy 0.87 / mIoU 0.64. |

**Next:** improve rare-class AP/IoU (field, vegetation, snow, gravel, wetland),
and consider instance/panoptic masks.

## Round 10 — training notebooks: spaces, losses and metrics

| # | Item | Type | Status |
|---|---|---|---|
| R10-1 | Per-notebook target/prediction spaces, loss/metric math, and pre-training numeric demos | docs | **Done** |

Each of `05`–`08` gained a "Prediction and target spaces, losses and metrics"
markdown section and a self-contained code cell that computes the loss and
metric on toy prediction/target pairs before training. Verified values match the
manual formulas (e.g. detection focal loss 0.009 vs 5.54 for a wrong cell,
segmentation CE 3.185 = manual, cloud BCE 0.273 = manual, classification CE
0.687 = manual).

## Round 11 — explicit variable names in the training notebooks

| # | Item | Type | Status |
|---|---|---|---|
| R11-1 | Rename cryptic variables (`cx`, `ci`, ...) in notebooks 05–08; comment the few short names kept | docs | **Done** |

Detection, segmentation, cloud and classification notebooks now use
`center_x/center_y`, `cell_x/cell_y`, `grid_x/grid_y`, `class_idx`,
`images/labels`, `train_/val_`, `true_positives/false_positives`, etc. The
`dx/dy` offsets of the Gaussian encoding and U-Net `enc1..enc4`/`dec1..dec3`
feature maps stay short and documented with comments. All four re-executed with
no errors and unchanged metrics.

## Round 12 — anchor detection + tree instance segmentation

| # | Item | Type | Severity | Status |
|---|---|---|---|---|
| R12-1 | Detection uses one box per cell; cannot separate two objects in the same cell | model | high | **Fixed** (5 anchors/cell) |
| R12-2 | No instance-level output for overlapping trees | feature | high | **Fixed** (canopy + centre U-Net, nearest-peak masks) |
| R12-3 | *field* AP@0.5 only 0.08: largest anchor (0.44) smaller than many fields | model | high | Open |
| R12-4 | small *vegetation* often missed | model | medium | Open |
| R12-5 | occasional duplicate boxes on one building | model | low | Open |

**Detection read-out (notebook 05):** water/boat detected well (AP 0.60/0.30);
fields under-detected; vegetation recall low. mAP@0.5 = 0.26 (peak at epoch 29,
then overfit).

**Instance-segmentation read-out (notebook 05):** counts match almost exactly
(MAE 0.00 over 32 forests), instance IoU 0.98, detection 301/301; only
sub-pixel boundary jitter and rare merges when trunks nearly coincide.

## Round 13 — notebook renumbering

| # | Item | Type | Status |
|---|---|---|---|
| R13-1 | Move scene classification into position 05; shift the rest (05->06, 06->07, 07->08) | docs | **Done** |

| R13-2 | Move object detection last: `06`->`08`, `07`->`06`, `08`->`07` | docs | **Done** |

Current order: `05` scene classification, `06` semantic segmentation,
`07` cloud occlusion, `08` object detection + instance segmentation.

## Round 14 — instance segmentation via a mask head on the detector

| # | Item | Type | Severity | Status |
|---|---|---|---|---|
| R14-1 | Separate U-Net + watershed for tree instances | architecture | high | **Replaced** by a multi-head detector with a per-anchor mask head + prototype upscaling |
| R14-2 | mask boundaries blobby vs GT crowns | model | medium | Open |
| R14-3 | count off by one on crowded scenes (missed detection / split) | model | medium | Open |
| R14-4 | mask branch can collapse during training (unstable) | training | medium | Open |

**Read-out (notebook 08):** objectness/box heads as before (mAP@0.5 = 0.26);
the added mask head reaches **count MAE 1.28**, **instance IoU 0.67**,
**detection 234/281 = 0.83** — lower than the earlier watershed decoder on this
near-circular forest data, but a more general Mask R-CNN / YOLACT-style design.

## Round 15 — detection / instance improvements

| # | Item | Type | Severity | Status |
|---|---|---|---|---|
| R15-1 | detection overfits after ~30 epochs (mAP 0.26) | training | high | **Fixed** (flip augmentation + best checkpoint, mAP 0.41) |
| R15-2 | instance IoU 0.67, blobby masks | model | medium | **Improved** (IoU 0.82, detection 0.99) |
| R15-3 | mask head collapsed with flips | bug | high | **Fixed** (flip the instance-id tensor too) |
| R15-4 | vegetation / rock AP still low (0.19 / 0.14) | model | medium | Open |

**Read-out (notebook 08):** detection mAP@0.5 = 0.41 (water 0.85, boat 0.60,
building 0.39, field 0.28, vegetation 0.19, rock 0.14); instance segmentation
count MAE 1.12, instance IoU 0.82, detection 411/417 = 0.99. Runtime ~5 min
(360+360 training samples, 140/120 epochs, flips).

## Round 16 — prototype vs per-anchor instance masks

| model | params | train s | ms/img | instance IoU | detection | count MAE |
|---|---|---|---|---|---|---|
| prototype masks (YOLACT) | 144.8k | 109.6 | 2.3 | **0.816** | **0.99** | 1.12 |
| per-anchor masks | 153.6k | **69.9** | **1.9** | 0.687 | 0.92 | **0.98** |

| # | Item | Type | Severity | Status |
|---|---|---|---|---|
| R16-1 | add prototype-free per-anchor mask head + comparison | feature | high | **Done** |
| R16-2 | per-anchor masks coarser on overlapping crowns | model | medium | Open |
| R16-3 | prototype head ~1.6x slower to train | training | low | Accepted |

**Read-out:** both heads share the detector, data and flips. The prototype head
wins on mask quality; the per-anchor head is a faster/cheaper tradeoff with a
slightly lower count error.

## Round 17 — urban/coastal detection refocus

| # | Item | Type | Severity | Status |
|---|---|---|---|---|
| R17-1 | refocus on building/rock/tree/car/boat/parking, mostly urban + coastal | data | high | **Done** |
| R17-2 | cars ~3 px at 64 px (AP ~0) | resolution | high | **Improved** (96 px, AP 0.09) |
| R17-3 | dense tree clumps / small rocks under-detected | model | medium | Open |
| R17-4 | occasional duplicate boxes on large buildings | model | low | Open |

**Read-out (notebook 08):** detection now runs at `96 x 96` on mostly urban +
urban-fringe + coastal scenes: mAP@0.5 **0.59** (building 0.96, parking 0.93,
boat 0.67, tree 0.53, rock 0.37, car 0.09). Progression: 0.41 (old classes,
64 px) -> 0.45 (jitter + more data) -> 0.59 (96 px). Cars remain the bottleneck;
128 px is the next step.

## Round 18 — coastal city, field boundaries, docs

| # | Item | Type | Severity | Status |
|---|---|---|---|---|
| R18-1 | no coastal-city scenes (breakwaters/piers/marinas) | data | high | **Fixed** (`coastal_city` scene) |
| R18-2 | `forest_fields` field mosaic inset by 6 % (grass margin) | data | medium | **Fixed** (mosaic fills the frame) |
| R18-3 | `forest_fields` woodland half was bare soil | data | medium | **Fixed** (canopy fill + more patches) |
| R18-4 | anchors from a hand-rolled single-init k-means | code | low | **Fixed** (sklearn `KMeans`, `n_init=10`) |
| R18-5 | cars still the weakest detection class (AP 0.12) | model | high | Open |

**Read-out (notebook 03):** `coastal_city` shows a breakwater, piers and a
marina; `forest_fields` fields now reach the frame edges and the woodland reads
as canopy. **Read-out (notebook 08):** detection on the richer urban + coastal +
coastal-city mix reaches mAP@0.5 0.56 (building 0.84, parking 0.78, boat 0.70,
tree 0.45, rock 0.44, car 0.12).

## Round 19 — three inspection-driven iterations

| # | Item | Type | Severity | Status |
|---|---|---|---|---|
| R19-1 | coastal-city land side sparse; buildings on roads; marina basin blends with the sea | realism | medium | **Fixed** (block grid + road clearance, water-filled quay basin) |
| R19-2 | cars ~3 px: undetectable and unrealistic | data/model | high | **Fixed** (cars/trucks drawn ~4-6 px; car AP 0.12 -> 0.18) |
| R19-3 | detection mAP 0.555 | model | medium | **Improved** (0.593) |
| R19-4 | diversity analysis still said "eight scene types" | docs | low | **Fixed** (eleven, incl. coastal_city) |
| R19-5 | trees/rocks still under-detected (AP 0.46/0.50) | model | medium | Open |

**Read-out (notebook 08):** detection mAP@0.5 0.593 (building 0.89, parking 0.84,
boat 0.68, rock 0.50, tree 0.46, car 0.18). Instance segmentation unchanged
(prototype IoU 0.81, per-anchor 0.69). Runtime ~9 min (downstream of the richer
coastal-city scenes).

**Next steps:** (1) class-frequency-weighted loss for rare classes (car, rock);
(2) test-time flip augmentation for detection; (3) trim detection runtime
(data 360 -> 300, epochs 150 -> 120); (4) 128 px remains the lever for cars.

## Round 20 — dataset-generation iterations (visual inspection)

| # | Item | Type | Severity | Status |
|---|---|---|---|---|
| R20-1 | mountain relief looked like binary white/black mottle | realism | medium | **Fixed** (continuous grayscale hillshade, faded in on slopes) |
| R20-2 | water had straight horizontal banding | realism | medium | **Fixed** (gentle wavy ripples) |
| R20-3 | urban blocks were large bare-concrete slabs | realism | medium | **Improved** (a second smaller building per block, ~60 %) |
| R20-4 | object shadows were pale gray slabs | realism | medium | **Fixed** (tree/bush shadow alpha 0.30 -> 0.42; building shadow 0.14/0.24 -> 0.18/0.30) |
| R20-5 | mountain still reads washed/gray at small size | realism | low | Open |
| R20-6 | pasture / arid remain sparse | realism | low | Open |

**Read-out:** relief now reads as ridges/valleys; water ripples are natural; urban
scenes are denser; shadows are darker and consistent (down-right, matching the
upper-left hillshade light).

## Round 21 — training-notebook inspection + educational review

**Qualitative read-out**

- **05 classification:** 3x5 grid now samples round-robin across all seven
  classes (was dominated by arid/coastal). Some genuine confusions visible
  (pasture -> cropland, arid -> coastal). Validation accuracy **0.80**.
- **06 segmentation:** 3x6 RGB / ground-truth / prediction grid; predictions
  track field, forest, urban and coastal masks well. Pixel accuracy **0.86**.
- **07 cloud:** 3x6 cloudy / mask / prediction grid; masks match closely.
  IoU **0.91**, Dice **0.96**.
- **08 detection:** 3x5 grid with green GT / red predictions; buildings, parking,
  boats, trees, rocks and cars all detected. mAP@0.5 **0.60** (car 0.20).
  Instance heads: prototype IoU **0.82** (det 0.98), per-anchor **0.69** (0.94).

**What is missing to make these good computer-vision teaching resources**

1. A short **objectives/learning-outcomes** box at the top of each training
   notebook (what concept: anchors, mAP, IoU, Dice, prototypes, ...).
2. **Exercises** (e.g. "swap the anchor parameterisation", "why does mAP drop
   when you raise the NMS threshold?") and expected answers.
3. An explicit **train/val/test** split (currently train/val only) and a short
   note on leakage / evaluation protocol.
4. **Metrics definitions** already appear as math, but a compact metrics table
   with the current numbers would help learners connect theory to results.
5. A **failure-gallery** figure per model (false positives / negatives) — the
   qualitative grids currently show successes and failures mixed.
6. **05 accuracy 0.80** dropped with the richer scenes: worth a note on class
   ambiguity (`forest_fields`, `urban_fringe`) and class-balanced augmentation.
7. **08 runtime ~9 min**: a "fast mode" flag (fewer samples/epochs) would keep
   the teaching loop snappy.
8. **Reproducibility**: document the pip-installed `torch`/`scikit-learn`
   extras (`install-torch`, `install-sklearn`) in the README.

## Round 22 — educational improvements

| # | Item | Type | Status |
|---|---|---|---|
| R22-1 | Training notebooks had no stated learning objectives | docs | **Added** (objectives + exercises in 05-08) |
| R22-2 | Detection had no failure analysis | docs/model | **Added** (TP/FP/FN failure gallery in 08) |
| R22-3 | No quick smoke-run mode for the slow notebook | tooling | **Added** (`FAST` flag in 08) |
| R22-4 | pip extras (torch, scikit-learn) undocumented | docs | **Added** (README setup note) |
| R22-5 | Only train/val split; no test split | methodology | Open |

**Failure-gallery read-out (notebook 08):** the main errors are **false positives
in dense urban blocks** (cars/trees over-detected or confused) and **missed
buildings in coastal cities** (clustered waterfront structures). This is now a
teaching asset: learners can see where mAP is lost.

## Round 23 — noise, dense forests, 5 inspection-driven steps

| # | Step | Type | Result |
|---|---|---|---|
| 1 | light sensor noise + ~20% clouds in **06** segmentation | data | acc 0.81, mIoU 0.58; model segments through haze/clouds |
| 2 | same noise in **08** detection | data | softened to "a small amount" after the first (strong) version cost 0.17 mAP; final mAP 0.54 |
| 3 | diverse **tree instance** data (sparse 6 -> dense 28, soil/grass/foliage, autumn trees) | data | much harder, realistic dense canopy |
| 4 | relabel **visible** instances (drop fully-occluded crowns) + ease dense regime | metric fix | prototype count MAE 5.9 -> 4.4 |
| 5 | per-anchor **finer masks** (16 -> 24) + stricter NMS (0.5 -> 0.4) | model | per-anchor count MAE 8.7 -> **1.7** |

**Final 08 read-out:** detection mAP@0.5 **0.54** (car 0.15) under light noise +
20 % clouds. Instance: prototype IoU **0.73** / count MAE 4.40 / det 0.93;
per-anchor IoU **0.64** / count MAE **1.69** / det 0.85. The two heads now show a
clear accuracy-vs-counting tradeoff on dense forests.

## Round 24 — "interpreting these numbers" analysis

| # | Item | Type | Status |
|---|---|---|---|
| R24-1 | Results lacked the message that lower accuracy on noisy/low-res imagery is expected | docs | **Added** |

Each training notebook (05-08) now ends with an **"Interpreting these numbers"**
section explaining that high scores on clean renders can mislead, that noise /
clouds / low resolution bound what any model can recover, that some errors are
irreducible (a ~4 px car, a fully occluded crown), and that evaluation is about
choosing an operating point rather than a single number.

## Round 25 — backlog review and prioritization

**Closed / superseded (stale entries removed from the active backlog)**

- `R12-3` (field AP 0.08), `R12-4` (vegetation missed), `R12-5` (duplicate boxes):
  the detection class set changed in R17 (no more field / vegetation / water).
- `R15-4`, `R19-5`: per-class AP numbers predate the R20-23 data changes.
- `R14-2`, `R16-2`: mask quality partly addressed in R23 (finer per-anchor masks,
  visible-instance relabel); only **dense-crown** quality remains open.
- `R14-4`: mask-branch collapse no longer reproduces but has no guard.

**Prioritized active backlog**

| Pri | ID | Item | Type | Effort | Impact |
|---|---|---|---|---|---|
| P0 | E1 | Held-out **test split** + evaluation-protocol note (05-08) | methodology | M | high |
| P0 | E2 | **Failure galleries** for 05/06/07 (only 08 has one) | docs/model | M | high |
| P1 | E3 | Cross-notebook **results table** + "how to improve" guidance | docs | L | med-high |
| P1 | D1 | **Mountain** scene reads washed/gray at small size (R20-5) | realism | L | med |
| P1 | D2 | **Pasture / arid** scenes remain sparse (R20-6) | realism | L | med |
| P1 | M1 | **Resolution ablation** for detection (64/96/128 px) as a lesson | model | H | med |
| P2 | M2 | Dense-forest **mask quality** (prototype IoU on overlapping crowns) | model | M | med |
| P2 | D3 | Cross-scene **scale/density consistency** of objects | realism | M | med |
| P3 | L1 | Occasional duplicate boxes on large buildings (R17-4) | model | L | low |
| P3 | L2 | Dense-forest **count stability** (prototype count MAE 4.4 vs per-anchor 1.7) | model | M | low |

**Rationale.** The notebooks are a teaching resource, so methodology gaps that
would mislead learners outrank headline accuracy: a test split and its protocol
come first, then failure galleries that make each model's errors visible. Dataset
realism follows. Model accuracy is deliberately lower priority: as the results
analysis now stresses, noise and low resolution set a ceiling, so the most useful
modelling next step is a **resolution ablation** that *demonstrates* that ceiling
rather than chasing the last point of mAP.

## Round 26 — prioritized plan execution (E1/E2/E3, D1/D2)

| # | Item | Type | Status |
|---|---|---|---|
| E1 | Held-out **test split** + protocol in 05-08 | methodology | **Done** |
| E2 | **Failure galleries** in 05 (least-confident), 06 (worst pixel error), 07 (worst cloud IoU) | docs/model | **Done** |
| E3 | Cross-notebook **results table + guidance** (README) | docs | **Done** |
| D1 | **Mountain** hillshade calmed, rock facets toned down | realism | **Done** |
| D2 | **Pasture / arid** densified (more clumps, rocks, scrub, wadi) | realism | **Done** |
| M1 | Resolution ablation (64/96/128 px) | model | Partly (documented; 128 px still open) |
| M2 | Dense-crown mask quality; per-anchor count stability | model | Open |

**Val/test read-out (test = held-out, reported once):**

| notebook | metric | validation | test |
|---|---|---|---|
| 05 classification | accuracy | 1.00 | 1.00 |
| 06 segmentation | pixel acc / mIoU | 0.78 / 0.53 | 0.79 / 0.54 |
| 07 cloud | IoU / Dice | 0.92 / 0.96 | 0.91 / 0.96 |
| 08 detection | mAP@0.5 | 0.49 | 0.50 |
| 08 instances prototype | IoU / count MAE | 0.73 / 5.97 | 0.72 / 6.47 |
| 08 instances per-anchor | IoU / count MAE | 0.64 / 1.48 | 0.63 / 1.78 |

Val and test are close everywhere, so the models are not tuned to validation.
The 06 **failure gallery** concentrates error on the ~20 % clouded scenes
(clouds hide the ground), and the 05 **least-confident** gallery lands on the
ambiguous `forest_fields`/`forest` boundary — both are exactly the intended
teaching moments.

## Round 27 — cropland boundaries, notebook-08 simplification, contextual notes

| # | Item | Type | Severity | Status |
|---|---|---|---|---|
| R27-1 | `urban_fringe` field mosaic inset 6 % → cropland never reached the frame (edge cropland fraction 0) | data | high | **Fixed** (mosaic fills the frame; edge fraction 0.73 left, 0.33 top/bottom) |
| R27-2 | `fields` land use biased to grass/woodland (edge cropland ~0.35; woodland filled with *soil*) | data | high | **Fixed** (cropland default, rare grass/woodland, woodland filled with foliage; edge ~0.7-0.8) |
| R27-3 | Notebook 08 carried two instance heads (prototype + per-anchor) | scope | medium | **Simplified** (prototype removed; per-anchor only) |
| R27-4 | "Interpreting these numbers" mentioned cars in notebooks that do not predict cars | docs | medium | **Fixed** (05/06/07 notes rewritten with task-relevant examples) |

**Root cause (R27-1/R27-2):** the boundary scenes inset their `_field_mosaic`
box by a 6 % margin and the `fields` scene assigned land use from the shared
elevation/slope map (low→grass, steep→woodland, woodland filled with the *soil*
material). Both left large non-crop regions along the image border. Fixes: the
mosaic boxes now span the full frame and `fields` defaults to cropland.

**Notebook 08 now:** a single per-anchor instance head (no prototypes).
Validation count MAE 1.8 / IoU 0.64 / detection 0.86; test MAE 1.5 / IoU 0.63 /
detection 0.84; detection mAP@0.5 val 0.50 / test 0.54. Runtime dropped (no
prototype branch).

## Round 28 — traffic, harder forests, MAPE

| # | Item | Type | Status |
|---|---|---|---|
| R28-1 | More cars/trucks on roads, kerbside parking and **traffic jams** (urban + coastal city) | data | **Done** (car AP 0.15 -> 0.22) |
| R28-2 | Forest instance task: **zoomed out**, smaller crowns, more trees (up to ~32), wider density range, light blur/noise | data | **Done** (harder: IoU 0.64 -> 0.58) |
| R28-3 | **MAPE** added alongside MAE for tree counts | metrics | **Done** (val 20 %, test 19 %) |
| R28-4 | Detection **NaN divergence** from degenerate (zero-size) edge-clipped cars | bug | **Fixed** (bounds-check vehicles + skip degenerate boxes) |
| R28-5 | Parking-lot cars are **sub-pixel** (pads too small) so annotating them hurt car AP | data | **Reverted**; scale-consistency still open |

**Read-out (notebook 08):** detection mAP@0.5 val **0.46** / test **0.50** (car
**0.22**); instance val count MAE **3.4** / **MAPE 20 %** / IoU **0.58** /
detection 0.71, test MAE 3.1 / MAPE 19 % / IoU 0.56 / detection 0.69.

**Open (scale consistency):** parking pads and their cars are drawn far smaller
than roadside cars, so lot cars are not resolvable — making the pads larger (or
labelling them only) is the next data-generation item.

## Round 29 — dataset-generation profiling and optimization

Tooling: a standalone timing harness generated N 96 px scenes and measured
wall-clock per scene. A single **mixed** run (360 scenes, all scene types) was
used as the headline number.

| # | Optimization | Where | Effect |
|---|---|---|---|
| R29-1 | Material-texture **cache warmth** (pre-request all needed materials once per scene) | `scene._compose` | avoids re-synthesizing textures per fill |
| R29-2 | **Batched road dashes** into one `LineCollection` | `object.draw_road`, `textures.ShapeBudget.add_line_collection` | removes one artist per dash |
| R29-3 | Skip per-artist **limit bookkeeping** (explicit axes limits) | `scene._compose` | removes autoscale overhead per artist |

**Per-scene time (median, 96 px):**

| scene | before | after |
|---|---|---|
| urban | 242 ms | **84 ms** |
| forest | 210 ms | **154 ms** |
| fields | 189 ms | **122 ms** |
| coastal_city | 211 ms | **128 ms** |
| 360 mixed scenes | 76 s | **25 s** (~3x) |

**Quality check:** `/tmp/opt_quality.png` (8 scenes) rendered and inspected at
reduced size — all scenes unchanged (urban grid, forest river, fields mosaic,
coastal_city harbour, pasture, arid rocks, mountain, forest_fields). No visual
regression. 162 tests pass; ruff clean.

**Note on evaluation tooling:** visual inspection now uses small downscaled
previews (640 px JPEG, ~40 KB) instead of the full PNGs, and notebooks are never
re-read wholesale, to stay well under the cumulative image-download budget.

## Round 30 — scale consistency, legend, boulders and avenues

| # | Item | Type | Severity | Status |
|---|---|---|---|---|
| R30-1 | Parking-lot cars were 4-6x smaller than roadside cars (`0.006-0.012` vs `0.04-0.056`) | data/realism | high | **Fixed** (cars now `size*0.24-0.30`, pads enlarged: urban `step*0.85-1.0`, coastal `0.12-0.18`) |
| R30-2 | Detection figures had no on-image legend | docs | low | **Added** (green = ground truth/prediction; gallery adds yellow = missed) |
| R30-3 | Forest instance data had no non-tree blobs | realism | medium | **Added** scattered **boulders** under crowns, not counted as instances |
| R30-4 | Instance data lacked ordered urban vegetation | realism/diversity | medium | **Added** **avenue** layout (regular rows of street trees lining a road with dashed centre line) |

**Visual check.** A parking-lot close-up (`/tmp/park_check.png`) confirms lot cars
now match the standalone road car; eight urban scenes show resolvable cars on the
pads. The instance gallery shows regular avenue rows with matching predicted
masks, and grey boulders left unmasked.

**Trade-off (R30-1).** Enlarging the pads shifted the k-means anchors toward
larger boxes, costing small objects: detection mAP@0.5 val 0.46 -> **0.42**, test
0.50 -> **0.44** (rock 0.27 -> 0.17, car 0.22 -> 0.18). Annotating the now
car-sized lot cars as individual `car` boxes was tested and lowered car AP further
(0.18 -> 0.16) because the packed boxes overlap; they are drawn but **not
labelled**. We keep the visual fix: the lesson is that more realistic clutter
lowers mAP. Instance IoU is unchanged (0.572 vs 0.576), so boulders/avenues do
not degrade counting (val MAPE 23 %).

**Read-out (notebook 08):** detection mAP@0.5 val **0.42** / test **0.44**
(building 0.72, boat 0.67, parking 0.46, tree 0.32, rock 0.17, car 0.18);
instance val MAE **3.1** / **MAPE 23 %** / IoU **0.57** / detection 0.68,
test MAE 3.4 / MAPE 23 % / IoU 0.56 / detection 0.66.

## Round 31 — fresh code + notebook + visual review (plan)

Reviewed the full scene / object / material vocabulary with fresh renders
(`/tmp/scene_overview.png`, `/tmp/objs_overview.png`, `/tmp/mats_overview.png`,
`/tmp/zoom.png`) and re-read the notebook structure and metrics.

**What holds up.** Scene composition is varied and plausible; object scale is now
consistent (parking fixed); shadows all fall down-right under an upper-left light;
all views are plan; the training notebooks are well structured (objectives, spaces,
losses, test split, failure galleries).

**Defects found (severity):**

| ID | Finding | Severity |
|---|---|---|
| A1 | **Conifer reads as a spiky dahlia/star** with a dark central hole, not a fir crown; appears in forest/mountain/wilderness/pasture | high |
| A2 | **Grounds are flat**: `soil`, `grass`, `crop` have almost no texture; `sand` uses one identical sinusoidal ripple everywhere | high |
| A3 | **Mountain is washed grey** with big flat polygonal facets; `rock` material is large-facet "confetti" | medium-high |
| A4 | **Coastline is a perfectly straight horizontal line**; river banks carry harsh white lines; `water` texture repeats | medium |
| A5 | Detection **small-object AP regressed** after the parking-pad change (rock 0.27->0.17, car 0.22->0.18) as k-means anchors shifted large | medium |

**Prioritized plan (one item per iteration).**

1. **Conifer crown** — replace the alternating-radius star with a soft radial fir
   crown (concentric blobs + needle streaks, small top).
2. **Ground materials** — add grain/mottle and per-seed variation to `soil`,
   `grass`, `crop`; vary `sand` ripple wavelength/orientation and add grain.
3. **Mountain / rock** — calmer hillshade, finer rock facets with subtle grain so
   the mountain stops reading as flat grey polygons.
4. **Water / coastline** — meander the shore, soften river banks (drop the white
   lines), vary water tone/ripples, add a soft foam edge.
5. **Detection anchors** — add an explicit small anchor (or more anchors) so rock
   and car recover; document the resolution ceiling.

### Round 31 — results (all 5 items done)

| ID | Item | Status |
|---|---|---|
| A1 | Conifer crown: soft radial fir crown (concentric blobs + batched needle streaks + small top) | **Fixed** |
| A2 | Flat grounds: soil gains clods/cracks, grass gains blade ticks + 2nd mottle, crop gains finer bands + grain, sand gets cross-ripples + grain | **Fixed** |
| A3 | Rock material: smaller/denser slabs, lighter facets, gravel grain; mountain relief alpha 0.7 -> 0.45 with more grass/soil patches | **Fixed** |
| A4 | Coastline: 3 shoreline harmonics, 2-band wet-sand gradient, segmented foam; rivers lose the hard centre line for faint dashed highlights | **Fixed** |
| A5 | Detection anchors: **log-space** k-means (raw for the uniform forest crowns) | **Fixed** |

**Metric read-out after the round (all notebooks re-run):**

| notebook | metric | validation | test |
|---|---|---|---|
| 05 classification | accuracy | 0.97 | 1.00 |
| 06 segmentation | pixel acc / mIoU | 0.82 / 0.58 | 0.83 / 0.58 |
| 07 cloud | IoU / Dice | 0.93 / 0.96 | 0.93 / 0.96 |
| 08 detection | mAP@0.5 | **0.47** | **0.45** |
| 08 instances | IoU / count MAE (MAPE) | 0.58 / 3.0 (22 %) | 0.53 / 3.8 (21 %) |

The log-space anchors are the headline model change: detection mAP@0.5 recovered
from 0.42 to **0.47** val (rock 0.17 -> 0.29, car 0.18 -> 0.24, tree 0.32 ->
0.49) after the parking-pad change had starved the small-object anchors.

**Remaining backlog (next round).** The conifer is much better but still reads as
a soft dark disc rather than a needle crown; `snow`/`wetland`/`gravel` semantic
IoU stays low (0.0 / 0.25 / 0.24); detection at 96 px still caps car/rock; and
`water`/`sand` remain fairly repetitive. Candidate next items: needle detail on
conifers, a 128 px detection run for the resolution lesson, and further ground /
water texture variation.





## Round 32 — educational figures in the model notebooks

This round adds interpretation aids (not new renders). The figures were
inspected in the notebooks:

| Notebook | New figure / output | What it teaches |
|---|---|---|
| 05 | majority-class baseline print (0.14 vs 0.97) | a score only means something above "do nothing" |
| 06 | training curves; per-class IoU vs class prevalence; most-frequent-class baseline | overfitting / saturation; why rare classes dominate mIoU while accuracy looks high |
| 07 | training curves; all-clear / all-cloud baselines; threshold sweep (IoU, Dice, precision, recall) | 0.5 is a convention; the operating point trades recall for precision |
| 08 | detection + instance training curves; no-object and mean-count baselines; per-class precision-recall curves | what mAP averages; the floor/ceiling of the metric; count error vs a mean predictor |

**Read-outs.** 05 baseline 0.143 (model 0.971). 06 baseline: high pixel accuracy,
mIoU far below the model's 0.58; the prevalence plot shows the rare classes
(`snow`, `gravel`, `wetland`) with the lowest IoU. 07 all-clear = 0.78 accuracy
with IoU/Dice 0; the sweep peaks near 0.5 but the curve is flat-topped, so the
threshold is a soft choice. 08 no-object mAP 0.000, mean-count baseline MAE 5.66
vs model 3.00; the PR curves show `boat`/`building` strong and `car`/`rock`
weak, which the single mAP hides.

## Round 33 — experiments and metric behaviour in the model notebooks

| Notebook | New figure / output | Read-out |
|---|---|---|
| 08 | augmentation gallery (original / h-flip / v-flip / jitter) | boxes mirror with the image, so flipped labels stay correct — the reason the augmentation is safe |
| 05 | accuracy over 3 seeds | 0.843 / 0.971 / 0.700, mean 0.838 ± 0.111; single-run gaps below ~0.1 are noise |
| 06 | per-class IoU: plain vs class-weighted CE | mIoU 0.465 -> 0.564; gravel 0.03 -> 0.15, wetland 0.17 -> 0.29 |
| 08 | mAP at IoU 0.5 / 0.65 / 0.75; score-threshold sweep | 0.472 / 0.247 / 0.060 (loose boxes); precision/recall/count trade off against the threshold |
| 07 | reliability diagram + temperature scaling | ECE 0.117 -> 0.006 at T = 0.30 |

The figures render cleanly, axes are labelled, and each is described in its
notebook. Notebook sizes stay under 5 MB (largest 4.68 MB).

## Round 34 — review-driven generator fixes + instance analysis

| Item | Change | Read-out |
|---|---|---|
| arid wadi | braided sandy channel, no centre line | reads as a dry riverbed, not a road (arid 3/8 inspected) |
| greenhouse | parallel glazed ranges with frame ribs | reads as glasshouses from above (`pasture 4` inspected) |
| instance by density (08) | MAE / IoU split by tree count + IoU histogram | MAE 2.29/3.54/4.33 and IoU 0.60/0.59/0.55 for sparse/medium/dense |

Stale generator read-outs refreshed: notebook 02 (conifer + greenhouse + parking)
and notebook 03 (arid wadi). Remaining backlog: mountain washes toward flat
grey, and some seeds keep a straight inner edge on the transition-scene field
mosaic.

## Round 35 — clarity of the model notebooks

| Notebook | Addition | Read-out |
|---|---|---|
| 08 | anchor-assignment figure (GT boxes, target cell, matched anchor) | the abstract `(cell, anchor)` encoding is now visible; dense images show up to 24 positives |
| 08 | box-relative mask-target figure (image / instances / one instance+box / 24x24 target) | makes the Mask R-CNN-style crop-and-resize target concrete |
| 05-08 | **Key terms** glossary cell | logits/softmax/CE, IoU/mIoU/ignore_index, Dice/BCE, anchor/NMS/mAP/ECE defined in place |
| 05-08 | exercise **what to expect** notes | learners can self-check each experiment |

Both new 08 figures render cleanly and are described in the notebook; sizes stay
under 5 MB.
