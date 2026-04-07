#!/usr/bin/env bash
################################################################################
# SAHI Parameter Grid Search — Find optimal 1080p config
#
# Tests combinations of: model, slice size, overlap, full-frame
# Compares each against the 1440p desktop baseline
################################################################################
set -euo pipefail

TEST_DIR="/apps/deepstream-sahi/python_test/deepstream-test-sahi"
VIDEO="/apps/deepstream-sahi/python_test/videos/aerial_vehicles.mp4"
BASELINE="results/aerial_vehicles_visdrone-full-640_sahi_20260401_181541.csv"
RESULTS_LOG="results/grid_search_results.txt"

cd "${TEST_DIR}"
source /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/pyds/bin/activate

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     SAHI Parameter Grid Search — 1080p Optimization        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Baseline: visdrone-full-640 @ 1440p (261.8 dets/frame)    ║"
echo "║  Target:   Maximize detections at 1080p                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Results header
echo "# SAHI Grid Search Results — $(date)" > "${RESULTS_LOG}"
echo "# Baseline: visdrone-full-640 @ 1440p, overlap=0.2, mean=261.8 dets/frame" >> "${RESULTS_LOG}"
echo "" >> "${RESULTS_LOG}"
printf "%-5s %-22s %-6s %-8s %-5s %-8s %-8s %-8s %-8s %-8s %-6s\n" \
    "Run" "Model" "Slice" "Overlap" "Full" "MeanDet" "Diff" "Pct" "PedDet" "FPS" "Status" >> "${RESULTS_LOG}"
printf "%-5s %-22s %-6s %-8s %-5s %-8s %-8s %-8s %-8s %-8s %-6s\n" \
    "---" "-----" "-----" "-------" "----" "-------" "----" "---" "------" "---" "------" >> "${RESULTS_LOG}"

RUN=0

run_test() {
    local MODEL="$1"
    local SLICE="$2"
    local OVERLAP="$3"
    local FULLFLAG="$4"
    local DESC="$5"

    RUN=$((RUN + 1))

    echo ""
    echo "━━━ Run ${RUN}: ${DESC} ━━━"
    echo "  Model=${MODEL} Slice=${SLICE} Overlap=${OVERLAP} FullFrame=${FULLFLAG}"

    # Build command
    local CMD="python3 deepstream_test_sahi.py --model ${MODEL} --resolution 1080p --no-display --csv"
    CMD="${CMD} --slice-width ${SLICE} --slice-height ${SLICE}"
    CMD="${CMD} --overlap-w ${OVERLAP} --overlap-h ${OVERLAP}"
    if [ "${FULLFLAG}" = "no" ]; then
        CMD="${CMD} --no-full-frame"
    fi
    CMD="${CMD} -i ${VIDEO}"

    echo "  CMD: ${CMD}"

    # Run pipeline, capture output
    local OUTPUT
    OUTPUT=$(${CMD} 2>&1) || true

    # Extract FPS
    local FPS
    FPS=$(echo "${OUTPUT}" | grep -oP "PERF:.*?'stream0': \K[0-9.]+" | tail -1 || echo "N/A")

    # Find the latest CSV
    local PATTERN="aerial_vehicles_${MODEL}_sahi_*.csv"
    local LATEST_CSV
    LATEST_CSV=$(ls -t results/${PATTERN} 2>/dev/null | head -1)

    if [ -z "${LATEST_CSV}" ] || [ ! -s "${LATEST_CSV}" ]; then
        echo "  ⚠️  No CSV produced, skipping"
        printf "%-5s %-22s %-6s %-8s %-5s %-8s %-8s %-8s %-8s %-8s %-6s\n" \
            "${RUN}" "${MODEL}" "${SLICE}" "${OVERLAP}" "${FULLFLAG}" "FAIL" "-" "-" "-" "${FPS}" "FAIL" >> "${RESULTS_LOG}"
        return
    fi

    # Extract stats
    local LINE_COUNT
    LINE_COUNT=$(wc -l < "${LATEST_CSV}")
    if [ "${LINE_COUNT}" -lt 10 ]; then
        echo "  ⚠️  CSV too short (${LINE_COUNT} lines), skipping"
        printf "%-5s %-22s %-6s %-8s %-5s %-8s %-8s %-8s %-8s %-8s %-6s\n" \
            "${RUN}" "${MODEL}" "${SLICE}" "${OVERLAP}" "${FULLFLAG}" "SHORT" "-" "-" "-" "${FPS}" "FAIL" >> "${RESULTS_LOG}"
        return
    fi

    # Calculate mean total_objects and mean pedestrian from CSV
    local MEAN_DET MEAN_PED
    MEAN_DET=$(tail -n +2 "${LATEST_CSV}" | awk -F',' '{sum+=$7; n++} END {printf "%.1f", sum/n}')
    MEAN_PED=$(tail -n +2 "${LATEST_CSV}" | awk -F',' '{sum+=$8; n++} END {printf "%.1f", sum/n}')

    local DIFF PCT
    DIFF=$(awk "BEGIN {printf \"%.1f\", ${MEAN_DET} - 261.8}")
    PCT=$(awk "BEGIN {printf \"%.1f\", (${MEAN_DET} - 261.8) / 261.8 * 100}")

    echo "  ✅ Mean dets/frame: ${MEAN_DET} (${PCT}% vs baseline), Pedestrians: ${MEAN_PED}, FPS: ${FPS}"

    printf "%-5s %-22s %-6s %-8s %-5s %-8s %-8s %-8s %-8s %-8s %-6s\n" \
        "${RUN}" "${MODEL}" "${SLICE}" "${OVERLAP}" "${FULLFLAG}" "${MEAN_DET}" "${DIFF}" "${PCT}%" "${MEAN_PED}" "${FPS}" "OK" >> "${RESULTS_LOG}"
}

# ─── Test Matrix ─────────────────────────────────────────────────────────────
# Model                  Slice  Overlap  FullFrame  Description

# Group 1: visdrone-full-640, vary overlap
run_test "visdrone-full-640" 640 0.2 yes "full-640, slice=640, overlap=0.2 (default)"
run_test "visdrone-full-640" 640 0.3 yes "full-640, slice=640, overlap=0.3"
run_test "visdrone-full-640" 640 0.4 yes "full-640, slice=640, overlap=0.4"

# Group 2: visdrone-full-640, smaller slices (more zoom per tile)
run_test "visdrone-full-640" 448 0.2 yes "full-640, slice=448, overlap=0.2"
run_test "visdrone-full-640" 448 0.3 yes "full-640, slice=448, overlap=0.3"
run_test "visdrone-full-640" 320 0.2 yes "full-640, slice=320, overlap=0.2"

# Group 3: visdrone-sliced-448 (model trained on sliced data)
run_test "visdrone-sliced-448" 448 0.2 yes "sliced-448, slice=448, overlap=0.2 (default)"
run_test "visdrone-sliced-448" 448 0.3 yes "sliced-448, slice=448, overlap=0.3"
run_test "visdrone-sliced-448" 448 0.4 yes "sliced-448, slice=448, overlap=0.4"
run_test "visdrone-sliced-448" 320 0.2 yes "sliced-448, slice=320, overlap=0.2"

# Group 4: No full-frame (slices only)
run_test "visdrone-full-640" 640 0.3 no "full-640, slice=640, overlap=0.3, no fullframe"
run_test "visdrone-sliced-448" 448 0.3 no "sliced-448, slice=448, overlap=0.3, no fullframe"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Grid search complete! Results saved to: ${RESULTS_LOG}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
cat "${RESULTS_LOG}"
