# nvsahipostprocess — Technical Review

This document consolidates all identified issues, gaps, and optimization
opportunities for the `gst-nvsahipostprocess` plugin. Each item describes
the problem, its impact, and a proposed solution.

> **Reference**: SAHI Python at `sahi/postprocess/` in this workspace.
> **Plugin source**: `deepstream_source/gst-plugins/gst-nvsahipostprocess/`
> **Detailed analysis**: [`docs/IMPROVEMENTS.md`](IMPROVEMENTS.md)

---

## Overview

| ID | Category | Issue | Severity |
|----|----------|-------|----------|
| F1 | Feature gap | Instance-segmentation mask merge not implemented | Critical |
| F2 | Feature gap | `obj_label` / `class_id` not updated on cross-class merge | Medium |
| A1 | Algorithm | Single-phase GreedyNMM diverges from SAHI two-phase behavior | Medium |
| A2 | Algorithm | Non-deterministic sort for equal scores | Low |
| A3 | Algorithm | NMM (non-greedy) algorithm unavailable | Low |
| P1 | Performance | O(n²) complexity without spatial indexing | Critical |
| P2 | Performance | Sequential per-frame processing (no parallelism) | High |
| P3 | Performance | Wasted cross-class comparisons when `class-agnostic=false` | Medium |
| P4 | Performance | `std::vector<bool>` bit-packing overhead in hot loop | Medium |
| P5 | Performance | Lock held during entire metadata modification loop | Medium |
| P6 | Performance | No pre-allocated memory (`reserve`) | Low |
| E1 | Enhancement | Per-frame debug statistics | Low |
| E2 | Enhancement | Maximum detections cap per frame | Low |
| E3 | Enhancement | Configurable merge strategy | Low |

---

## F1 — Instance-Segmentation Mask Merge [Critical]

### Problem

The plugin does not read, merge, or update segmentation masks. When two
detections are merged, the bounding box expands (union) but the surviving
mask stays unchanged — covering only the original pre-merge region. The
suppressed detection's mask is discarded entirely.

The word "mask" does not appear anywhere in the plugin source code.

### Impact

Any pipeline using instance segmentation (e.g. YOLO-Seg via
`nvdsinfer_yolo`) produces misaligned masks after merge. The bbox grows
but the mask does not follow.

### SAHI Python Reference

SAHI converts both masks to Shapely polygons, computes their geometric
union via `poly1.union(poly2)`, and stores the combined polygon:

```python
# sahi/postprocess/utils.py — get_merged_mask()
poly1 = get_shapely_multipolygon(mask1.segmentation).buffer(0)
poly2 = get_shapely_multipolygon(mask2.segmentation).buffer(0)
union_poly = poly1.union(poly2)
```

### Proposed Solution

Since DeepStream masks are dense float arrays (not COCO polygons), the
union can be computed without Shapely:

1. Add mask data pointer and dimensions to `SahiDetection`.
2. When merging detection `j` into `i`:
   - If both carry mask data: resize both masks to the merged bbox
     coordinate space, take element-wise maximum, store back.
   - If only one has a mask: keep it (resize to merged bbox).
   - If neither has a mask: skip (current behavior).
3. In the metadata-update loop, write merged mask back to `NvDsObjectMeta`.

### Alternative (Simpler)

Add a boolean property `drop-mask-on-merge` (default `true`) that clears
the mask when a merge occurs. This makes the behavior explicit instead of
silently misaligned.

---

## F2 — `obj_label` Not Updated on Cross-Class Merge [Medium]

### Problem

When `class-agnostic=true` and two detections with different `class_id`
values merge, the plugin updates bbox and confidence but never touches
`obj_meta->class_id` or `obj_meta->obj_label`. The surviving object keeps
its original class regardless of which detection had the higher score.

### Impact

In the typical case (higher-score detection survives) the class is already
correct. But in edge cases (equal scores, or when score is updated from the
suppressed detection) the label can be wrong.

### SAHI Python Reference

```python
# sahi/postprocess/utils.py — get_merged_category()
def get_merged_category(pred1, pred2):
    if pred1.score.value > pred2.score.value:
        return pred1.category
    else:
        return pred2.category
```

### Proposed Solution

In the metadata-update loop, when a detection is merged and
`class_agnostic=true`, track the `class_id` and `obj_label` of the
highest-scoring contributor and update the surviving `NvDsObjectMeta`
accordingly.

---

## A1 — Single-Phase vs. Two-Phase GreedyNMM [Medium]

### Problem

The C++ plugin performs suppression and merge in a single pass. Overlap is
computed against the bbox of `dets[i]` which is mutated in-place by
previous merges. The SAHI Python version operates in two phases:

1. **Phase 1**: determine candidates using original (immutable) bboxes.
2. **Phase 2**: re-check overlap against the expanding bbox, merge only if
   still above threshold.

The C++ version is more aggressive — after merging A with B (expanding A),
the expanded A may absorb C that the original A did not overlap with.

```
A overlaps B:          yes       →  A absorbs B, A expands
union(A,B) overlaps C: yes       →  A absorbs C  (C++ only)
original A overlaps C: no        →  C survives   (Python)
```

### Impact

Edge cases with chains of partially-overlapping detections may merge more
aggressively than the SAHI reference.

### Proposed Solution

Refactor `greedy_nmm` into two phases:

1. First pass: iterate pairs using **original** coordinates. Record
   `keep_to_merge_list` (map from survivor index to list of merge indices).
   Mark matched candidates as suppressed.
2. Second pass: iterate each survivor's merge list. Re-check overlap
   against the **current** (expanding) bbox. Merge only if still above
   threshold.

### Alternative

Document as intentional behavior and add a property `two-phase-nmm`
(default `false`) to opt-in to SAHI-exact behavior.

---

## A2 — Non-Deterministic Sort for Equal Scores [Low]

### Problem

`std::sort` with only score comparison produces implementation-defined
order for detections with identical confidence. Different runs,
compilers, or platforms may yield different merge results.

SAHI Python uses deterministic tie-breaking via lexicographic comparison
of box coordinates.

### Proposed Solution

Add a secondary sort key:

```cpp
std::sort(order.begin(), order.end(),
    [&dets](guint a, guint b) {
        if (dets[a].score != dets[b].score)
            return dets[a].score > dets[b].score;
        auto ca = std::tie(dets[a].left, dets[a].top,
                           dets[a].right, dets[a].bottom);
        auto cb = std::tie(dets[b].left, dets[b].top,
                           dets[b].right, dets[b].bottom);
        return ca < cb;
    });
```

---

## A3 — NMM (Non-Greedy) Algorithm Unavailable [Low]

### Problem

SAHI Python provides four postprocess strategies: NMS, NMM (bidirectional
merge), GreedyNMM, and LSNMS. The C++ plugin only implements GreedyNMM
(with NMS mode via `enable-merge=false`).

### Impact

Low. GreedyNMM is the recommended algorithm for real-time use. NMM creates
transitive merge chains and is more expensive. LSNMS is experimental.

### Proposed Solution

Document as unsupported. If demand arises, the NMM algorithm can be added
behind a property toggle.

---

## P1 — O(n²) Algorithm Complexity [Critical]

### Problem

The `greedy_nmm` inner loop performs n²/2 pair comparisons where
n = detections per frame. For dense scenes (n=500+), this dominates
processing time.

| Detections (n) | Pair comparisons | Time estimate (per frame) |
|----------------|-----------------|---------------------------|
| 100 | 5,000 | ~0.1 ms |
| 300 | 45,000 | ~0.9 ms |
| 500 | 125,000 | ~2.5 ms |
| 1000 | 500,000 | ~10 ms |

With 5 sources processed sequentially, n=500 costs ~12.5ms per batch,
n=1000 costs ~50ms — exceeding the 33ms budget at 30 FPS.

SAHI Python uses Shapely's STRtree (R-tree) to achieve ~O(n log n) by
querying only spatially nearby candidates.

### Proposed Solution

Implement a lightweight 2D spatial hash grid:

1. Divide the frame into cells of `max_bbox_dimension` size.
2. Insert each detection into all overlapping cells.
3. For each survivor, query only detections in overlapping cells.

This avoids external dependencies (no Shapely/GEOS) while achieving
similar O(n log n) average-case performance. Expected improvement:
**10–50x** for spatially distributed detections.

**Alternative**: Sort detections by x-coordinate and use a sweep-line to
skip pairs that cannot overlap horizontally.

---

## P2 — Sequential Per-Frame Processing [High]

### Problem

`transform_ip` processes all frames in the batch sequentially. With
5 sources, the NMM computation for frame 5 waits for frames 1–4 to
finish. Frames are independent (different `frame_meta`, different
`obj_meta_list`) but share the same `detections`, `suppressed`, and
`sorted_indices` vectors (struct members), preventing concurrency.

### Impact

Linear cost growth with number of sources. 5 sources = 5x the latency
compared to single-source.

### Proposed Solution

**Option A — OpenMP** (simplest):

```cpp
std::vector<NvDsFrameMeta*> frames;
for (auto *l = batch_meta->frame_meta_list; l; l = l->next)
    frames.push_back((NvDsFrameMeta*)l->data);

#pragma omp parallel for schedule(dynamic)
for (size_t f = 0; f < frames.size(); f++) {
    // thread-local vectors
    std::vector<SahiDetection> local_dets;
    std::vector<uint8_t> local_supp;
    std::vector<guint> local_order;
    process_frame_local(..., frames[f]);
}
// then acquire batch_meta lock once for all metadata modifications
```

**Option B — Thread pool**: Pre-allocate per-thread vectors, dispatch
frames to workers, barrier before metadata modification.

**Option C — Local vectors**: Move vectors from struct members to local
variables in `process_frame`. Simpler, still serial, but enables future
parallelism without shared-state issues.

---

## P3 — Wasted Cross-Class Comparisons [Medium]

### Problem

When `class-agnostic=false` (default), the inner loop checks class
mismatch on every pair and skips ~90% of comparisons:

```cpp
if (!agnostic && dets[i].class_id != dets[j].class_id)
    continue;
```

For n=500 across 10 classes (~50 each):
- Current: 125,000 pair checks, ~112,500 wasted
- Per-class: 10 × 1,250 = 12,500 pair checks (10x reduction)

### SAHI Python Reference

```python
# sahi/postprocess/combine.py — batched_greedy_nmm()
for category_id in torch.unique(category_ids):
    curr_indices = torch.where(category_ids == category_id)[0]
    curr_keep_to_merge_list = greedy_nmm(tensor[curr_indices], ...)
```

### Proposed Solution

Partition detections by `class_id` before the NMM loop:

1. After sorting by score, group indices using
   `std::unordered_map<int, std::vector<guint>>`.
2. Run `greedy_nmm` separately on each group.
3. Combine suppressed/merged results.

---

## P4 — `std::vector<bool>` Overhead [Medium]

### Problem

`std::vector<bool>` is a C++ special case that stores bits, not bytes.
Every access to `suppressed[j]` in the hot inner loop requires
bit-extraction through a proxy object. This prevents SIMD
auto-vectorization and causes cache-line false sharing on writes.

### Proposed Solution

Replace with `std::vector<uint8_t>`:

```cpp
// gstnvsahipostprocess.h
std::vector<uint8_t> suppressed;  // was: std::vector<bool>
```

Expected improvement: 15–30% speedup on the inner loop.

---

## P5 — Lock Scope During Metadata Modification [Medium]

### Problem

The `batch_meta` lock is held during the entire modification loop
(potentially hundreds of `nvds_remove_obj_meta_from_frame` calls + update
operations). This blocks any concurrent access to batch metadata from
other threads (e.g. downstream probes).

### Proposed Solution

**Option A**: Use per-frame locks instead of the global `batch_meta` lock,
if DeepStream supports frame-level locking. This allows concurrent
modification of different frames.

**Option B**: Collect all suppressed `obj_meta` pointers into a temporary
list during the unlocked phase, then perform removals in a single locked
pass (minimizing hold time).

**Note**: The current design already separates computation (unlocked) from
modification (locked), which is good. The remaining optimization is to
reduce the number of operations performed under the lock.

---

## P6 — No Pre-Allocated Memory [Low]

### Problem

The `detections`, `suppressed`, and `sorted_indices` vectors grow
dynamically via `push_back` / `resize`. If a new frame has more detections
than any previous frame, reallocation + copy occurs mid-processing.

### Proposed Solution

Add `reserve()` calls in `gst_nvsahipostprocess_init`:

```cpp
self->detections.reserve(1024);
self->suppressed.reserve(1024);
self->sorted_indices.reserve(1024);
```

Optionally expose as a property `max-detections-hint` for user tuning.

---

## E1 — Per-Frame Debug Statistics [Low]

### Problem

No way to observe merge behavior at runtime. Tuning `match-threshold` and
`match-metric` requires trial-and-error without feedback on how many
detections are suppressed or merged.

### Proposed Solution

Add `GST_DEBUG` output at `LOG` level with per-frame counters:
- Detections before/after merge
- Number suppressed, number merged
- Per-class breakdown

Optionally attach a lightweight custom `GstMeta` with counters readable
by downstream Python probes.

---

## E2 — Maximum Detections Cap [Low]

### Problem

In high-density scenes, hundreds of detections may survive merging,
overwhelming downstream tracking or rendering.

### Proposed Solution

Add a property `max-detections` (default `-1` = unlimited). After the
merge loop, if surviving detections exceed the cap, remove the
lowest-scoring survivors.

---

## E3 — Configurable Merge Strategy [Low]

### Problem

Merge always uses bbox union + max score. Some use-cases may benefit from
alternative strategies.

### Proposed Solution

Add a `merge-strategy` property:
- `union` (current/default): bbox = union of both boxes, score = max.
- `weighted`: bbox = confidence-weighted average, score = max.
- `largest`: keep the larger bbox unchanged, score = max.

---

## Priority Summary

| Priority | Items | Impact |
|----------|-------|--------|
| **Critical** | F1 (mask merge), P1 (O(n²) complexity) | Required for segmentation pipelines; required for high-volume workloads |
| **High** | P2 (parallel frames), F2 (class label update) | Multi-source scalability; correctness |
| **Medium** | A1 (two-phase NMM), P3 (per-class partition), P4 (`vector<bool>`), P5 (lock scope) | SAHI parity; performance for default config |
| **Low** | A2 (tie-breaking), A3 (NMM algo), P6 (reserve), E1–E3 (enhancements) | Correctness polish; developer experience |
