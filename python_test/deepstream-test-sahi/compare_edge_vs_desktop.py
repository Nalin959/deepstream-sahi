#!/usr/bin/env python3

################################################################################
# Quick CSV Comparison — Desktop (FP16) vs Edge (INT8/FP16) Model
#
# Compares per-frame detection counts from two pipeline runs to validate
# that the edge-optimized model maintains accuracy parity.
#
# Usage:
#   python3 compare_edge_vs_desktop.py \
#       results/aerial_vehicles_visdrone-full-640_sahi_*.csv \
#       results/aerial_vehicles_visdrone-drone-fp16_sahi_*.csv
#
# Or with explicit labels:
#   python3 compare_edge_vs_desktop.py \
#       -a results/desktop_run.csv \
#       -b results/edge_run.csv \
#       --label-a "Desktop FP16 (batch=4)" \
#       --label-b "Edge FP16 (batch=8)"
################################################################################

import argparse
import csv
import os
import sys
import glob


def load_csv(path):
    """Load a detection CSV and return per-frame data."""
    frames = {}
    metadata = {}

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        class_columns = [c for c in reader.fieldnames
                         if c not in {"video", "model", "sahi", "resolution",
                                      "tracker", "frame", "total_objects"}]

        for row in reader:
            frame_num = int(row["frame"])
            total = int(row["total_objects"])
            class_counts = {c: int(row[c]) for c in class_columns}
            frames[frame_num] = {
                "total": total,
                "classes": class_counts,
            }
            if not metadata:
                metadata = {
                    "video": row.get("video", ""),
                    "model": row.get("model", ""),
                    "sahi": row.get("sahi", ""),
                    "resolution": row.get("resolution", ""),
                    "tracker": row.get("tracker", ""),
                }

    return frames, metadata, class_columns


def compare(frames_a, frames_b, class_columns, label_a, label_b):
    """Compare two frame-by-frame detection outputs."""
    # Align frames
    common_frames = sorted(set(frames_a.keys()) & set(frames_b.keys()))
    only_a = set(frames_a.keys()) - set(frames_b.keys())
    only_b = set(frames_b.keys()) - set(frames_a.keys())

    if not common_frames:
        print("ERROR: No common frames found between the two CSVs.")
        return None

    # Per-frame totals
    totals_a = [frames_a[f]["total"] for f in common_frames]
    totals_b = [frames_b[f]["total"] for f in common_frames]

    mean_a = sum(totals_a) / len(totals_a)
    mean_b = sum(totals_b) / len(totals_b)

    diffs = [b - a for a, b in zip(totals_a, totals_b)]
    mean_diff = sum(diffs) / len(diffs)
    abs_diffs = [abs(d) for d in diffs]
    mean_abs_diff = sum(abs_diffs) / len(abs_diffs)
    max_diff = max(abs_diffs)

    pct_change = ((mean_b - mean_a) / mean_a * 100) if mean_a > 0 else 0

    # Per-class comparison
    class_stats = {}
    for cls in class_columns:
        cls_a = [frames_a[f]["classes"].get(cls, 0) for f in common_frames]
        cls_b = [frames_b[f]["classes"].get(cls, 0) for f in common_frames]
        mean_cls_a = sum(cls_a) / len(cls_a)
        mean_cls_b = sum(cls_b) / len(cls_b)
        class_stats[cls] = {
            "mean_a": mean_cls_a,
            "mean_b": mean_cls_b,
            "diff": mean_cls_b - mean_cls_a,
            "pct": ((mean_cls_b - mean_cls_a) / mean_cls_a * 100
                    if mean_cls_a > 0 else 0),
        }

    # Frame agreement (within tolerance)
    exact_match = sum(1 for d in diffs if d == 0)
    within_5 = sum(1 for d in abs_diffs if d <= 5)
    within_10 = sum(1 for d in abs_diffs if d <= 10)

    return {
        "common_frames": len(common_frames),
        "only_a": len(only_a),
        "only_b": len(only_b),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": mean_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_diff": max_diff,
        "pct_change": pct_change,
        "exact_match": exact_match,
        "within_5": within_5,
        "within_10": within_10,
        "class_stats": class_stats,
    }


def print_report(stats, label_a, label_b, meta_a, meta_b):
    """Print a formatted comparison report."""
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║          Detection Model Comparison Report                      ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  A: {label_a:<57s} ║")
    print(f"║  B: {label_b:<57s} ║")
    if meta_a.get("video"):
        print(f"║  Video: {meta_a['video']:<53s} ║")
    if meta_a.get("resolution"):
        print(f"║  Resolution: A={meta_a['resolution']}, B={meta_b['resolution']}")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    n = stats["common_frames"]
    print(f"  Frames compared: {n}")
    if stats["only_a"]:
        print(f"  Frames only in A: {stats['only_a']}")
    if stats["only_b"]:
        print(f"  Frames only in B: {stats['only_b']}")
    print()

    # Overall totals
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │                Overall Detection Totals             │")
    print("  ├─────────────────────────────────────────────────────┤")
    print(f"  │  Mean detections/frame (A): {stats['mean_a']:>10.1f}            │")
    print(f"  │  Mean detections/frame (B): {stats['mean_b']:>10.1f}            │")
    print(f"  │  Mean difference (B - A):   {stats['mean_diff']:>+10.1f}            │")
    print(f"  │  Percentage change:         {stats['pct_change']:>+10.1f}%           │")
    print(f"  │  Mean absolute difference:  {stats['mean_abs_diff']:>10.1f}            │")
    print(f"  │  Max absolute difference:   {stats['max_diff']:>10.0f}              │")
    print("  └─────────────────────────────────────────────────────┘")
    print()

    # Frame agreement
    print("  Frame Agreement:")
    print(f"    Exact match:     {stats['exact_match']:>5d} / {n}  "
          f"({stats['exact_match']/n*100:.1f}%)")
    print(f"    Within ±5 dets:  {stats['within_5']:>5d} / {n}  "
          f"({stats['within_5']/n*100:.1f}%)")
    print(f"    Within ±10 dets: {stats['within_10']:>5d} / {n}  "
          f"({stats['within_10']/n*100:.1f}%)")
    print()

    # Per-class breakdown
    class_stats = stats["class_stats"]
    active_classes = {k: v for k, v in class_stats.items()
                      if v["mean_a"] > 0.1 or v["mean_b"] > 0.1}

    if active_classes:
        print("  Per-Class Comparison (mean detections/frame):")
        print(f"    {'Class':<20s} {'A':>8s} {'B':>8s} {'Diff':>8s} {'Change':>8s}")
        print(f"    {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
        for cls, s in sorted(active_classes.items(),
                             key=lambda x: -abs(x[1]["mean_a"])):
            print(f"    {cls:<20s} {s['mean_a']:>8.1f} {s['mean_b']:>8.1f} "
                  f"{s['diff']:>+8.1f} {s['pct']:>+7.1f}%")
        print()

    # Verdict
    abs_pct = abs(stats["pct_change"])
    if abs_pct < 1.0:
        verdict = "✅ EXCELLENT — Negligible difference (<1%)"
    elif abs_pct < 3.0:
        verdict = "✅ GOOD — Minor difference (<3%)"
    elif abs_pct < 5.0:
        verdict = "⚠️  ACCEPTABLE — Moderate difference (<5%)"
    elif abs_pct < 10.0:
        verdict = "⚠️  REVIEW — Notable difference (<10%)"
    else:
        verdict = "🔴 SIGNIFICANT — Large difference (>10%)"

    print(f"  Verdict: {verdict}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare detection CSVs between desktop and edge models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Compare latest CSVs for aerial_vehicles
  python3 compare_edge_vs_desktop.py \\
      results/aerial_vehicles_visdrone-full-640_sahi_*.csv \\
      results/aerial_vehicles_visdrone-drone-fp16_sahi_*.csv

  # Explicit paths with labels
  python3 compare_edge_vs_desktop.py \\
      -a results/desktop.csv -b results/edge.csv \\
      --label-a "Desktop FP16" --label-b "Edge INT8"
""",
    )
    parser.add_argument("-a", "--csv-a", type=str, default=None,
                        help="Path to CSV A (baseline/desktop)")
    parser.add_argument("-b", "--csv-b", type=str, default=None,
                        help="Path to CSV B (edge/new model)")
    parser.add_argument("positional", nargs="*",
                        help="Positional args: CSV_A CSV_B (alternative to -a/-b)")
    parser.add_argument("--label-a", default=None,
                        help="Label for model A (default: auto from CSV)")
    parser.add_argument("--label-b", default=None,
                        help="Label for model B (default: auto from CSV)")
    args = parser.parse_args()

    # Resolve CSV paths
    csv_a = args.csv_a
    csv_b = args.csv_b

    if args.positional:
        # Expand globs
        expanded = []
        for p in args.positional:
            matches = sorted(glob.glob(p))
            expanded.extend(matches if matches else [p])

        if len(expanded) == 2:
            csv_a = csv_a or expanded[0]
            csv_b = csv_b or expanded[1]
        elif len(expanded) > 2:
            # Take the last two (most recent by timestamp in filename)
            csv_a = csv_a or expanded[-2]
            csv_b = csv_b or expanded[-1]

    if not csv_a or not csv_b:
        parser.print_help()
        print("\nError: Two CSV files required (A=baseline, B=new model)")

        # Show available CSVs
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        if os.path.isdir(results_dir):
            csvs = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
            if csvs:
                print(f"\nAvailable CSVs in {results_dir}:")
                for c in csvs:
                    print(f"  {os.path.basename(c)}")
        sys.exit(1)

    if not os.path.isfile(csv_a):
        print(f"Error: File not found: {csv_a}")
        sys.exit(1)
    if not os.path.isfile(csv_b):
        print(f"Error: File not found: {csv_b}")
        sys.exit(1)

    # Load CSVs
    print(f"Loading A: {os.path.basename(csv_a)}")
    frames_a, meta_a, cols_a = load_csv(csv_a)
    print(f"  → {len(frames_a)} frames, model={meta_a.get('model', '?')}")

    print(f"Loading B: {os.path.basename(csv_b)}")
    frames_b, meta_b, cols_b = load_csv(csv_b)
    print(f"  → {len(frames_b)} frames, model={meta_b.get('model', '?')}")

    # Use common class columns
    class_columns = [c for c in cols_a if c in cols_b]

    # Auto-generate labels
    label_a = args.label_a or f"{meta_a.get('model', 'A')} ({meta_a.get('sahi', '')})"
    label_b = args.label_b or f"{meta_b.get('model', 'B')} ({meta_b.get('sahi', '')})"

    # Compare
    stats = compare(frames_a, frames_b, class_columns, label_a, label_b)
    if stats:
        print_report(stats, label_a, label_b, meta_a, meta_b)


if __name__ == "__main__":
    main()
