# nvsahipostprocess — v1.2 Technical Review & Changelog

This document describes all issues identified in the v1.0 postprocess plugin,
the solutions implemented in v1.2, and the resulting performance and
correctness improvements.

---

## Summary of Changes

| ID | Category | Issue | Severity | Status |
|----|----------|-------|----------|--------|
| F1 | Feature | Instance-segmentation mask merge | Critical | **Implemented** |
| F2 | Feature | Cross-class label update on merge | Medium | **Implemented** |
| A1 | Algorithm | Two-phase GreedyNMM | Medium | **Implemented** |
| A2 | Algorithm | Deterministic sort for equal scores | Low | **Implemented** |
| A3 | Algorithm | NMM (non-greedy, bidirectional) | Low | Documented |
| P1 | Performance | O(n²) → spatial hash grid indexing | Critical | **Implemented** |
| P2 | Performance | Parallel per-frame processing (OpenMP) | High | **Implemented** |
| P3 | Performance | Per-class partitioning (class-agnostic=false) | Medium | **Implemented** |
| P4 | Performance | `vector<bool>` → `vector<uint8_t>` | Medium | **Implemented** |
| P5 | Performance | Lock scope optimization | Medium | **Implemented** |
| P6 | Performance | Pre-allocated memory (`reserve`) | Low | **Implemented** |
| E1 | Enhancement | Per-frame debug statistics | Low | **Implemented** |
| E2 | Enhancement | Maximum detections cap per frame | Low | **Implemented** |
| E3 | Enhancement | Configurable merge strategy | Low | **Implemented** |

---

## F1 — Instance-Segmentation Mask Merge [Critical → Implemented]

### Problem

The v1.0 plugin did not process segmentation masks. When two detections were
merged, the bounding box expanded (union) but the surviving mask stayed
unchanged, covering only the original pre-merge region. The suppressed
detection's mask was discarded entirely. Any pipeline using instance
segmentation (e.g., YOLO-Seg) produced misaligned masks after merge.

### Solution

Added mask merge support using element-wise maximum in the merged bbox
coordinate space:

1. `SahiDetection` now carries `SahiMaskData` (pointer, width, height,
   threshold) copied from `NvDsObjectMeta.mask_params`.
2. When merging detection `j` into `i`:
   - Both masks are projected into the union bbox using nearest-neighbor
     resampling (`mask_merge.h:sahi_mask_project_max`).
   - Element-wise maximum is taken, producing a combined mask.
3. The merged mask is written back to `obj->mask_params` with updated
   dimensions.
4. A `drop-mask-on-merge` property (default `false`) allows explicitly
   clearing masks on merge when exact composition is not needed.

### Files Changed

- `mask_merge.h` (new) — mask projection and merge utilities
- `gstnvsahipostprocess.h` — `SahiDetection.mask`, `SahiDetection.merged_mask_data`
- `gstnvsahipostprocess.cpp` — mask merge in phase 2, mask writeback in metadata update

---

## F2 — Cross-Class Label Update [Medium → Implemented]

### Problem

When `class-agnostic=true` and two detections with different `class_id`
values were merged, the surviving `NvDsObjectMeta` retained its original
`class_id` and `obj_label`. The suppressed detection's class was lost even
if it had the higher confidence score.

### Solution

`SahiDetection` now tracks the `best_score`, `best_class_id`, and
`best_label` of the highest-scoring contributor across all merges. During
metadata writeback, if `class_agnostic=true` and the best contributor had
a different class, `obj_meta->class_id` and `obj_meta->obj_label` are
updated accordingly.

---

## A1 — Two-Phase GreedyNMM [Medium → Implemented]

### Problem

The v1.0 plugin performed suppression and merge in a single pass. Overlap
was computed against `dets[i]` which was mutated in-place by previous merges.
This caused cascade merging: after merging A with B (expanding A), the
expanded A could absorb C that the original A did not overlap with.

```
A overlaps B:          yes  → A absorbs B, A expands
union(A,B) overlaps C: yes  → A absorbs C  (v1.0)
original A overlaps C: no   → C survives   (v1.2 two-phase)
```

### Solution

Refactored `greedy_nmm` into two phases:

1. **Phase 1**: iterate pairs using **original** (immutable) coordinates via
   the spatial grid. Record a `keep_to_merge_list` mapping. Mark matched
   candidates as suppressed.
2. **Phase 2**: iterate each survivor's merge list. Re-check overlap against
   the **current** (expanding) bbox. Merge only if still above threshold.

Controlled by the `two-phase-nmm` property (default `true`). Set to `false`
to revert to the more aggressive v1.0 single-phase behavior.

---

## A2 — Deterministic Sort Tie-Breaking [Low → Implemented]

### Problem

`std::sort` with score-only comparison produced implementation-defined
ordering for detections with identical confidence. Different runs, compilers,
or platforms could yield different merge results.

### Solution

Added a secondary sort key using lexicographic comparison of bbox coordinates
(`left`, `top`, `right`, `bottom`). Results are now deterministic and
reproducible across platforms.

---

## A3 — NMM (Non-Greedy) Algorithm [Low → Documented]

The bidirectional NMM algorithm allows transitive merge chains and is more
expensive. It is rarely needed for real-time pipelines. The GreedyNMM
algorithm covers the vast majority of use-cases. If demand arises, NMM can
be added behind a property toggle in a future version.

---

## P1 — Spatial Hash Grid Indexing [Critical → Implemented]

### Problem

The v1.0 inner loop was O(n²) where n = detections per frame:

| Detections (n) | Pair comparisons | Time per frame |
|----------------|-----------------|----------------|
| 100 | 5,000 | ~0.1 ms |
| 500 | 125,000 | ~2.5 ms |
| 1000 | 500,000 | ~10 ms |

With 5 sources processed sequentially, n=1000 costs ~50ms per batch —
exceeding the 33ms budget at 30 FPS.

### Solution

Implemented a lightweight 2D spatial hash grid (`spatial_grid.h`):

1. Frame is divided into uniform cells sized to the largest detection dimension.
2. Each detection is inserted into all cells it overlaps.
3. For each survivor, only detections in overlapping cells are checked.
4. Candidates are deduplicated via sort + unique.

Expected improvement: **10–50×** fewer pair checks for spatially distributed
detections. For n=500 across 9 slices, the grid typically checks ~15–30
candidates per survivor instead of 500.

---

## P2 — Parallel Per-Frame Processing [High → Implemented]

### Problem

`transform_ip` processed all frames in the batch sequentially. With 5
sources, the computation for frame 5 waited for frames 1–4 to finish.
Frames are independent but shared the same working vectors (struct members),
preventing concurrency.

### Solution

- All working state (detections, suppressed, order, grid) is now local to
  `process_frame`, enabling safe concurrent execution.
- `transform_ip` collects frame pointers into a vector and dispatches them
  via `#pragma omp parallel for schedule(dynamic)`.
- The batch_meta lock is acquired only during the brief metadata modification
  phase per frame.
- Compile with `-fopenmp` / link with `-lgomp` (added to Makefile).

Expected improvement: near-linear scaling with batch-size on multi-core
systems.

---

## P3 — Per-Class Partitioning [Medium → Implemented]

### Problem

When `class-agnostic=false` (default), the inner loop checked class mismatch
on every pair and skipped ~90% of comparisons. For n=500 across 10 classes:
125,000 pair checks with only 12,500 same-class.

### Solution

Detections are partitioned by `class_id` using an `unordered_map` before
the NMM loop. Each class group runs NMM independently:

```
n=500, 10 classes, ~50 each:
  Before: 125,000 pair checks
  After:  10 × 1,250 = 12,500 pair checks (10× reduction)
```

---

## P4 — `vector<bool>` → `vector<uint8_t>` [Medium → Implemented]

### Problem

`std::vector<bool>` is a C++ special case that stores bits instead of bytes.
Every access in the hot inner loop required bit-extraction through a proxy
object, preventing SIMD auto-vectorization.

### Solution

Replaced with `std::vector<uint8_t>` — direct byte access, SIMD-friendly.
Expected improvement: 15–30% speedup on the inner loop.

---

## P5 — Lock Scope Optimization [Medium → Implemented]

### Problem

The batch_meta lock was held during the entire modification loop (potentially
hundreds of remove + update operations).

### Solution

- Suppressed objects are collected into a temporary vector during the
  unlocked computation phase.
- The lock is acquired only for the removal pass.
- Merged object metadata updates (bbox, confidence, mask) are performed
  before the lock, since each thread modifies only its own frame's objects.

---

## P6 — Pre-Allocated Memory [Low → Implemented]

### Solution

`dets.reserve(512)` in `process_frame` eliminates runtime reallocation for
typical workloads (up to 512 detections per frame without realloc).

---

## E1 — Per-Frame Debug Statistics [Low → Implemented]

Added `GST_LOG` output with per-frame counters:
- Total detections, suppressed, merged, surviving.

Enable with: `GST_DEBUG=nvsahipostprocess:6`

---

## E2 — Maximum Detections Cap [Low → Implemented]

Added `max-detections` property (default `-1` = unlimited). After the merge
loop, if surviving detections exceed the cap, the lowest-scoring survivors
are removed. Useful for high-density scenes that would overwhelm downstream
tracking or rendering.

---

## E3 — Configurable Merge Strategy [Low → Implemented]

Added `merge-strategy` property with three options:

| Value | Name | Behavior |
|-------|------|----------|
| 0 | Union | bbox = min/max corners (default, standard behavior) |
| 1 | Weighted | bbox = confidence-weighted average of coordinates |
| 2 | Largest | keep the larger bbox unchanged |

All strategies use `score = max(scores)` for confidence.

---

## Performance Comparison (Estimated)

For a batch with 5 sources × 450 detections/frame:

| Phase | v1.0 (sequential, O(n²)) | v1.2 (spatial grid + parallel) |
|-------|--------------------------|-------------------------------|
| Collect detections | 5 × ~0.02ms = 0.1ms | 0.1ms |
| Sort | 5 × ~0.03ms = 0.15ms | 0.15ms |
| GreedyNMM | 5 × ~3ms = **15ms** | 5 × 0.1ms / 5 threads = **0.1ms** |
| Metadata update | 5 × ~0.5ms = 2.5ms | 2.5ms |
| **Total per batch** | **~18ms** | **~3ms** |
| **Budget at 30 FPS** | 33ms | 33ms |

At 1000 detections/frame (extreme density):

| Phase | v1.0 | v1.2 |
|-------|------|------|
| GreedyNMM | 5 × ~12ms = **60ms** (over budget) | 5 × 0.4ms / 5 threads = **0.4ms** |

---

## Known Limitations

1. **NMM (non-greedy) algorithm** is not implemented. GreedyNMM covers
   real-time use-cases adequately.
2. **Threshold operator**: the `>=` operator is used consistently. Some
   reference implementations use `>` for the re-check phase, which can
   cause an edge case where overlap exactly at threshold suppresses but
   does not merge. The `>=` behavior is intentional and more correct.
3. **Mask merge resolution** is capped at 512×512 to prevent excessive
   memory allocation for very large merged bboxes.

---

## File Structure

```
gst-nvsahipostprocess/
├── Makefile                      — build with OpenMP support
├── gstnvsahipostprocess.h        — plugin struct, SahiDetection, enums
├── gstnvsahipostprocess.cpp      — GStreamer plugin, GreedyNMM, metadata
├── spatial_grid.h                — 2D spatial hash for neighbor queries
└── mask_merge.h                  — mask projection and element-wise merge
```
