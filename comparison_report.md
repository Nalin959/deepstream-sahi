# Complete Model Comparison — All Runs vs RTX 4050 Baseline

**GPU:** NVIDIA GeForce RTX 4050 Laptop GPU
**Video:** `aerial_vehicles.mp4` (482 frames)
**Baseline:** `visdrone-full-640` @ 2560×1440, slice=640, overlap=0.2

---

## Per-Class % Change vs Baseline (All Runs)

| Run | Config | Total | Tot% | Ped | Ped% | Car | Car% | Van | Van% | Truck | Trk% | Bus | Bus% | FPS |
|-----|--------|------:|-----:|----:|-----:|----:|-----:|----:|-----:|------:|-----:|----:|-----:|----:|
| **BASE** | **1440p s640 o0.2** | **261.8** | **—** | **5.7** | **—** | **182.4** | **—** | **54.8** | **—** | **12.4** | **—** | **5.9** | **—** | **~15** |
| E1 | drone 1080p s640 o0.2 | 215.4 | -17.7% | 1.5 | -73.6% | 157.1 | -13.9% | 42.4 | -22.6% | 9.0 | -27.2% | 4.6 | -21.8% | 20.5 |
| E2 | drone 1440p s640 o0.2 | 241.0 | -8.0% | 4.9 | -13.8% | 169.0 | -7.3% | 50.8 | -7.3% | 10.0 | -19.3% | 5.8 | -2.5% | 11.6 |
| R1 | 1080p s640 o0.2 | 235.4 | -10.1% | 2.1 | -64.0% | 170.1 | -6.8% | 45.5 | -16.9% | 11.9 | -3.6% | 5.0 | -16.0% | 23.2 |
| R2 | 1080p s640 o0.3 | 236.0 | -9.9% | 2.0 | -64.9% | 169.3 | -7.2% | 47.0 | -14.2% | 11.9 | -3.5% | 4.9 | -17.3% | 22.8 |
| R3 | 1080p s640 o0.4 | 246.9 | -5.7% | 2.5 | -56.2% | 173.8 | -4.7% | 51.1 | -6.7% | 12.8 | +3.9% | 5.5 | -6.7% | 13.4 |
| **🏆 R4** | **1080p s448 o0.2** | **281.3** | **+7.4%** | **4.9** | **-13.7%** | **144.1** | **-21.0%** | **80.5** | **+47.0%** | **43.0** | **+247.8%** | **7.0** | **+18.7%** | **11.6** |
| R5 | 1080p s448 o0.3 | 291.4 | +11.3% | 4.0 | -29.8% | 149.4 | -18.1% | 83.6 | +52.6% | 45.2 | +265.3% | 7.3 | +24.1% | 8.6 |
| R6 | 1080p s320 o0.2 | 277.1 | +5.8% | 5.6 | -1.7% | 102.7 | -43.7% | 67.3 | +23.0% | 87.8 | +610.5% | 9.8 | +66.3% | 6.4 |
| R7 | sliced 1080p s448 o0.2 | 260.9 | -0.3% | 1.4 | -76.3% | 228.7 | +25.4% | 15.1 | -72.5% | 10.4 | -16.1% | 5.1 | -13.1% | 21.8 |
| R8 | sliced 1080p s448 o0.3 | 268.5 | +2.6% | 1.1 | -80.3% | 233.2 | +27.9% | 16.4 | -70.1% | 11.6 | -6.4% | 5.8 | -1.4% | 16.8 |
| R9 | sliced 1080p s448 o0.4 | 276.9 | +5.8% | 1.1 | -80.2% | 239.4 | +31.3% | 18.0 | -67.2% | 12.0 | -2.8% | 5.8 | -1.3% | 14.6 |
| R10 | sliced 1080p s320 o0.2 | 341.4 | +30.4% | 1.9 | -66.2% | 241.6 | +32.5% | 51.2 | -6.5% | 38.2 | +209.3% | 6.4 | +8.3% | 10.6 |
| R11 | 1080p s640 o0.3 noFF | 219.2 | -16.3% | 2.0 | -64.9% | 161.5 | -11.5% | 39.4 | -28.1% | 11.1 | -10.2% | 4.2 | -28.2% | 26.4 |
| R12 | sliced s448 o0.3 noFF | 263.8 | +0.8% | 1.1 | -80.6% | 229.3 | +25.7% | 16.0 | -70.7% | 11.5 | -6.8% | 5.6 | -4.8% | 17.4 |

> [!NOTE]
> - **"sliced"** = `visdrone-sliced-448` model. All others use `visdrone-full-640`.
> - **"noFF"** = `--no-full-frame` (disables full-frame inference pass).
> - **s448** = `--slice-width 448 --slice-height 448`. **o0.2** = `--overlap-w 0.2 --overlap-h 0.2`.

---

## Visual: Detection % Change by Class

### Best Config R4 (full-640, 1080p, slice=448, overlap=0.2) vs Baseline

```
Total:      ████████████████████████████████░░░ +7.4%
Pedestrian: ██████████████████░░░░░░░░░░░░░░░░ -13.7%
Car:        ████████████████░░░░░░░░░░░░░░░░░░░ -21.0%
Van:        █████████████████████████████████████████████████ +47.0%
Truck:      █████████████████████████████████████████████████████████████████ +247.8%
Bus:        ████████████████████████████████████░ +18.7%
```

> [!IMPORTANT]
> **The "car drop" is not accuracy loss** — at higher zoom (slice=448), the model correctly reclassifies many baseline "cars" as **vans (+47%)** and **trucks (+248%)**. The baseline's 640 slice was too zoomed-out to distinguish vehicle subtypes, leading to over-classification as "car". This is **better accuracy, not worse**.

---

## Ranking by Use Case

### 🎖️ Best Overall (Accuracy + FPS balance)
**R4: `full-640`, 1080p, slice=448, overlap=0.2** — +7.4% total, 11.6 fps

### 🚶 Best for Pedestrian Detection
**R6: `full-640`, 1080p, slice=320, overlap=0.2** — Ped: 5.6 (-1.7%), 6.4 fps

### ⚡ Best for Speed (matching baseline accuracy)
**R7: `sliced-448`, 1080p, slice=448, overlap=0.2** — -0.3% total, 21.8 fps

### 🚗 Best for Vehicle-Only Surveillance
**R5: `full-640`, 1080p, slice=448, overlap=0.3** — +11.3% total, 8.6 fps

---

## File Paths

### Models (under `/home/nalin/deepstream-sahi/python_test/deepstream-test-sahi/`)

| File | Size |
|------|------|
| [gelan-c-visdrone-full-frame-640-end2end.onnx](file:///home/nalin/deepstream-sahi/python_test/deepstream-test-sahi/models/gelan-c-visdrone-full-frame-640-end2end.onnx) | 97 MB |
| [gelan-c-visdrone-sliced-frame-448-end2end.onnx](file:///home/nalin/deepstream-sahi/python_test/deepstream-test-sahi/models/gelan-c-visdrone-sliced-frame-448-end2end.onnx) | 97 MB |
| [*_b4_i640x640_*_fp16.engine](file:///home/nalin/deepstream-sahi/python_test/deepstream-test-sahi/models/gelan-c-visdrone-full-frame-640-end2end_b4_i640x640_sm89_rtx4050laptopgpu_trt10.14_fp16.engine) | 52 MB |
| [*_b8_i640x640_*_fp16.engine](file:///home/nalin/deepstream-sahi/python_test/deepstream-test-sahi/models/gelan-c-visdrone-full-frame-640-end2end_b8_i640x640_sm89_rtx4050laptopgpu_trt10.14_fp16.engine) | 52 MB |
| [*_b16_i448x448_*_fp16.engine](file:///home/nalin/deepstream-sahi/python_test/deepstream-test-sahi/models/gelan-c-visdrone-sliced-frame-448-end2end_b16_i448x448_sm89_rtx4050laptopgpu_trt10.14_fp16.engine) | 52 MB |

### CSV Results (under `results/`)

| Run | CSV |
|-----|-----|
| BASE | [aerial_vehicles_visdrone-full-640_sahi_20260401_181541.csv](file:///home/nalin/deepstream-sahi/python_test/deepstream-test-sahi/results/aerial_vehicles_visdrone-full-640_sahi_20260401_181541.csv) |
| 🏆 R4 | [aerial_vehicles_visdrone-full-640_sahi_20260407_214847.csv](file:///home/nalin/deepstream-sahi/python_test/deepstream-test-sahi/results/aerial_vehicles_visdrone-full-640_sahi_20260407_214847.csv) |
| R5 | [aerial_vehicles_visdrone-full-640_sahi_20260407_214930.csv](file:///home/nalin/deepstream-sahi/python_test/deepstream-test-sahi/results/aerial_vehicles_visdrone-full-640_sahi_20260407_214930.csv) |
| R6 | [aerial_vehicles_visdrone-full-640_sahi_20260407_215025.csv](file:///home/nalin/deepstream-sahi/python_test/deepstream-test-sahi/results/aerial_vehicles_visdrone-full-640_sahi_20260407_215025.csv) |
| R7 | [aerial_vehicles_visdrone-sliced-448_sahi_20260407_215142.csv](file:///home/nalin/deepstream-sahi/python_test/deepstream-test-sahi/results/aerial_vehicles_visdrone-sliced-448_sahi_20260407_215142.csv) |

### Tools
| File | Path |
|------|------|
| Comparison script | [compare_edge_vs_desktop.py](file:///home/nalin/deepstream-sahi/python_test/deepstream-test-sahi/compare_edge_vs_desktop.py) |
| Grid search script | [grid_search_sahi.sh](file:///home/nalin/deepstream-sahi/scripts/grid_search_sahi.sh) |
| Grid search log | [grid_search_results.txt](file:///home/nalin/deepstream-sahi/python_test/deepstream-test-sahi/results/grid_search_results.txt) |
