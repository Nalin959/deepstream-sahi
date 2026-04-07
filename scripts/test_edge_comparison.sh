#!/usr/bin/env bash

################################################################################
# Test Script — Compare Desktop vs Edge Model Accuracy
#
# Runs the SAHI pipeline with the edge-optimized config and compares
# detection CSV output against the existing desktop baseline.
#
# Must be run INSIDE the DeepStream container:
#   docker start -ai deepstream-sahi
#   source /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/pyds/bin/activate
#   cd /apps/deepstream-sahi
#   scripts/test_edge_comparison.sh
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEST_DIR="${REPO_ROOT}/python_test/deepstream-test-sahi"
VIDEO_DIR="${REPO_ROOT}/python_test/videos"

# ─── Configuration ───────────────────────────────────────────────────────────

BASELINE_MODEL="visdrone-full-640"
EDGE_MODEL="visdrone-drone-fp16"  # FP16 edge config (no INT8 calibration needed)
VIDEO="aerial_vehicles.mp4"
RESOLUTION="1080p"  # Edge: 1080p instead of 1440p

# ─── Validate environment ───────────────────────────────────────────────────

if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "ERROR: DeepStream Python virtualenv not activated."
    echo "Run:  source /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/pyds/bin/activate"
    exit 1
fi

if [ ! -f "${VIDEO_DIR}/${VIDEO}" ]; then
    echo "ERROR: Test video not found: ${VIDEO_DIR}/${VIDEO}"
    echo "Download from: https://drive.google.com/drive/folders/1CRLnuH9AtTwmxRz7z-Mtu6ErKx__VMK4"
    exit 1
fi

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          Edge vs Desktop Accuracy Comparison Test               ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Baseline: ${BASELINE_MODEL} (FP16, 1440p, batch=4)"
echo "║  Edge:     ${EDGE_MODEL} (FP16, ${RESOLUTION}, batch=8)"
echo "║  Video:    ${VIDEO}"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

cd "${TEST_DIR}"

# ─── Step 1: Run baseline (if CSV doesn't exist with enough data) ────────────

echo "=== Step 1: Baseline model (${BASELINE_MODEL}) ==="

BASELINE_CSV=$(ls -t results/${VIDEO%.mp4}_${BASELINE_MODEL}_sahi_*.csv 2>/dev/null | head -1)
if [ -n "${BASELINE_CSV}" ] && [ "$(wc -l < "${BASELINE_CSV}")" -gt 10 ]; then
    echo "Using existing baseline CSV: ${BASELINE_CSV}"
    echo "  ($(wc -l < "${BASELINE_CSV}") lines)"
else
    echo "Running baseline pipeline..."
    python3 deepstream_test_sahi.py \
        --model "${BASELINE_MODEL}" \
        --resolution 1440p \
        --no-display \
        --csv \
        -i "${VIDEO_DIR}/${VIDEO}"

    BASELINE_CSV=$(ls -t results/${VIDEO%.mp4}_${BASELINE_MODEL}_sahi_*.csv 2>/dev/null | head -1)
    echo "Baseline CSV: ${BASELINE_CSV}"
fi

echo ""

# ─── Step 2: Run edge model ─────────────────────────────────────────────────

echo "=== Step 2: Edge model (${EDGE_MODEL}) ==="
echo "Running edge pipeline at ${RESOLUTION}..."

python3 deepstream_test_sahi.py \
    --model "${EDGE_MODEL}" \
    --resolution "${RESOLUTION}" \
    --no-display \
    --csv \
    -i "${VIDEO_DIR}/${VIDEO}"

EDGE_CSV=$(ls -t results/${VIDEO%.mp4}_${EDGE_MODEL}_sahi_*.csv 2>/dev/null | head -1)
echo "Edge CSV: ${EDGE_CSV}"

echo ""

# ─── Step 3: Compare ────────────────────────────────────────────────────────

echo "=== Step 3: Comparison ==="

python3 compare_edge_vs_desktop.py \
    -a "${BASELINE_CSV}" \
    -b "${EDGE_CSV}" \
    --label-a "Desktop FP16 (1440p, batch=4)" \
    --label-b "Edge FP16 (${RESOLUTION}, batch=8)"

echo ""
echo "Done! CSV files saved in: ${TEST_DIR}/results/"
