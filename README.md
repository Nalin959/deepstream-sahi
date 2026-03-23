# DeepStream SAHI

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![DeepStream](https://img.shields.io/badge/NVIDIA-DeepStream%208.0%20|%209.0-76B900?logo=nvidia)](https://developer.nvidia.com/deepstream-sdk)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.x-orange)](https://developer.nvidia.com/tensorrt)

Native GStreamer plugins that integrate **SAHI** (Slicing Aided Hyper Inference) into NVIDIA DeepStream for real-time small-object detection in high-resolution video streams.

> **Inspired by [SAHI](https://github.com/obss/sahi)** — the original framework-agnostic sliced inference library by OBSS. This project implements the SAHI slicing and GreedyNMM merging algorithms as native C++ GStreamer plugins for zero-copy, GPU-accelerated inference within DeepStream pipelines.

## Highlights

- **nvsahipreprocess** — Dynamically computes SAHI slices per frame, crops and scales each slice on GPU via `NvBufSurfTransform`, and prepares tensors for `nvinfer`.
- **nvsahipostprocess** — Merges duplicate detections at slice boundaries using the GreedyNMM algorithm (IoU/IoS metrics). Operates entirely on `NvDsObjectMeta` — no tensor access, no CUDA.
- **Drop-in pipeline integration** — Fits between `nvstreammux` and `nvinfer` (preprocess) and between `nvinfer` and `nvtracker` (postprocess).

```
nvstreammux → nvsahipreprocess → nvinfer → nvsahipostprocess → nvtracker → nvdsosd
```

## What Makes It Different

DeepStream SAHI does not run as a separate Python inference workflow. Slicing,
inference, and post-processing run as native GStreamer plugins inside the
DeepStream pipeline, keeping the full path close to GPU memory.

- **TensorRT-native inference** through `nvinfer` for lower latency and higher throughput.
- **Dynamic multi-stream support** with `nvmultiurisrcbin`, including URI-based sources and runtime add/remove workflows.
- **Full DeepStream compatibility** with tracking, analytics, messaging, and on-screen display components.

This makes SAHI a production-friendly DeepStream building block rather than an
external add-on around the pipeline.

---

## Results

All tests on **2560×1440 aerial surveillance video** | **NVIDIA RTX 5080** | **FP16** | **Batch 16**

### SAHI vs No SAHI — Detection Improvement

#### visdrone-full-640 (640×640, full-frame trained)

| Video | Scene | No SAHI | SAHI | Improvement |
|-------|-------|---------|------|-------------|
| aerial_crowding_01 | Dense pedestrian crowd | 13.8 obj/frame | **84.2** obj/frame | **+510%** |
| aerial_crowding_02 | Very dense crowd | 206.2 obj/frame | **664.7** obj/frame | **+222%** |
| aerial_vehicles | Dense vehicle traffic | 92.3 obj/frame | **252.5** obj/frame | **+174%** |

#### visdrone-sliced-448 (448×448, SAHI-aware trained)

| Video | Scene | No SAHI | SAHI | Improvement |
|-------|-------|---------|------|-------------|
| aerial_crowding_01 | Dense pedestrian crowd | 2.3 obj/frame | **85.3** obj/frame | **+3,619%** |
| aerial_crowding_02 | Very dense crowd | 35.9 obj/frame | **614.9** obj/frame | **+1,613%** |
| aerial_vehicles | Dense vehicle traffic | 28.6 obj/frame | **226.7** obj/frame | **+694%** |

> The sliced-trained model (448×448) benefits dramatically more from SAHI because its native resolution is too small for 2560×1440 input. Without SAHI, small objects are invisible. With SAHI slicing, the model operates at its intended scale.

### Full-Frame vs Sliced Training — Both with SAHI

| Video | full-640 + SAHI | sliced-448 + SAHI | Difference |
|-------|----------------|------------------|------------|
| aerial_crowding_01 | 84.2 | 85.3 | +1.3% |
| aerial_crowding_02 | 664.7 | 614.9 | -7.5% |
| aerial_vehicles | 252.5 | 226.7 | -10.2% |

> With SAHI enabled, both training strategies converge to comparable detection counts. The full-frame model has a slight edge on large objects that span multiple slices.

### Dense Pedestrian Crowd — SAHI vs No SAHI (full-640)

<p align="center">
  <img src="test_results/comparison_aerial_crowding_01_visdrone-full-640-sahi_vs_aerial_crowding_01_visdrone-full-640-no-sahi/01_total_objects_over_frames.png" width="80%" alt="Total objects per frame — aerial_crowding_01 SAHI vs No SAHI"/>
</p>
<p align="center">
  <img src="test_results/comparison_aerial_crowding_01_visdrone-full-640-sahi_vs_aerial_crowding_01_visdrone-full-640-no-sahi/02_class_comparison_bar.png" width="80%" alt="Per-class comparison — aerial_crowding_01 SAHI vs No SAHI"/>
</p>

### Very Dense Crowd — SAHI vs No SAHI (full-640)

<p align="center">
  <img src="test_results/comparison_aerial_crowding_02_visdrone-full-640-sahi_vs_aerial_crowding_02_visdrone-full-640-no-sahi/01_total_objects_over_frames.png" width="80%" alt="Total objects per frame — aerial_crowding_02 SAHI vs No SAHI"/>
</p>
<p align="center">
  <img src="test_results/comparison_aerial_crowding_02_visdrone-full-640-sahi_vs_aerial_crowding_02_visdrone-full-640-no-sahi/02_class_comparison_bar.png" width="80%" alt="Per-class comparison — aerial_crowding_02 SAHI vs No SAHI"/>
</p>

### Dense Vehicle Traffic — SAHI vs No SAHI (full-640)

<p align="center">
  <img src="test_results/comparison_aerial_vehicles_visdrone-full-640-sahi_vs_aerial_vehicles_visdrone-full-640-no-sahi/01_total_objects_over_frames.png" width="80%" alt="Total objects per frame — aerial_vehicles SAHI vs No SAHI"/>
</p>
<p align="center">
  <img src="test_results/comparison_aerial_vehicles_visdrone-full-640-sahi_vs_aerial_vehicles_visdrone-full-640-no-sahi/02_class_comparison_bar.png" width="80%" alt="Per-class comparison — aerial_vehicles SAHI vs No SAHI"/>
</p>

See [Test Results](docs/TEST_RESULTS.md) for the complete evaluation with all model/video combinations.

### Video Demos

| Dense Pedestrian Crowd | Very Dense Crowd | Dense Vehicle Traffic |
|:---:|:---:|:---:|
| [![Dense Pedestrian Crowd](https://img.youtube.com/vi/_W_wBDvpzzY/hqdefault.jpg)](https://www.youtube.com/watch?v=_W_wBDvpzzY&list=PLJMGcwo73q30LtZaCw1VQ7UGPvGsmVlco) | [![Very Dense Crowd](https://img.youtube.com/vi/RFX8hIWscgw/hqdefault.jpg)](https://www.youtube.com/watch?v=RFX8hIWscgw&list=PLJMGcwo73q33YfssoIGBIMu51EPJBobxf) | [![Dense Vehicle Traffic](https://img.youtube.com/vi/3CxEp90Jy60/hqdefault.jpg)](https://www.youtube.com/watch?v=3CxEp90Jy60&list=PLJMGcwo73q33HWQfjHUD_exEVOUSnlbsA) |

---

## Quick Start

> **Note:** This repository uses [Git LFS](https://git-lfs.com/) to store ONNX model files (~97 MB each). Make sure Git LFS is installed before cloning so the models are downloaded automatically.

```bash
# Install Git LFS (once per machine)
git lfs install

# Clone — ONNX models are pulled automatically via Git LFS
git clone https://github.com/levipereira/deepstream-sahi.git
cd deepstream-sahi

# Launch DeepStream container (see docs/INSTALL.md for full options)
# DeepStream 9.x:
docker run -it --name deepstream-sahi --net=host --gpus all \
    -v `pwd`:/apps/deepstream-sahi -w /apps/deepstream-sahi \
    nvcr.io/nvidia/deepstream:9.0-triton-multiarch
# DeepStream 8.x:
# docker run -it ... nvcr.io/nvidia/deepstream:8.0-triton-multiarch

# Inside the container — single command installs everything:
# The installer auto-detects the DeepStream version and adapts accordingly.
/apps/deepstream-sahi/install.sh

# Download test videos into python_test/videos/ (see link below)
# Activate Python environment and run
source /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/pyds/bin/activate
cd python_test/deepstream-test-sahi
python3 deepstream_test_sahi.py --model visdrone-full-640 --no-display --csv ../videos/aerial_crowding_01.mp4
```

> **Test Videos:** Download the aerial surveillance test videos from [Google Drive](https://drive.google.com/drive/folders/1CRLnuH9AtTwmxRz7z-Mtu6ErKx__VMK4) and place them in `python_test/videos/`. You can also use your own videos.

## Documentation

| Document | Description |
|----------|-------------|
| [Installation Guide](docs/INSTALL.md) | Container setup, dependencies, plugin build |
| [Usage Guide](docs/USAGE.md) | Running pipelines, CLI arguments, comparing results |
| [Plugin Reference](docs/PLUGINS.md) | Complete documentation for nvsahipreprocess and nvsahipostprocess |
| [Training Guide](docs/TRAINING.md) | SAHI-aware model training: why and how to train on sliced data |
| [Test Results](docs/TEST_RESULTS.md) | Evaluation data: SAHI vs standard inference |
| [Technical Review](docs/REVIEW.md) | Known issues, SAHI parity gaps, and performance analysis |

## Repository Structure

```
deepstream-sahi/
├── deepstream_source/
│   ├── gst-plugins/
│   │   ├── gst-nvsahipreprocess/       # SAHI dynamic-slice pre-process plugin
│   │   └── gst-nvsahipostprocess/      # SAHI GreedyNMM post-process plugin
│   └── libs/
│       ├── nvdsinfer_8.0/              # Modified inference lib for DS 8.x
│       ├── nvdsinfer_9.0/              # Modified inference lib for DS 9.x
│       └── nvdsinfer_yolo/             # YOLO custom bounding-box parser
├── python_test/
│   ├── common/                         # Shared GStreamer utilities
│   ├── deepstream-test-sahi/           # Test pipelines, configs, models
│   │   └── models/                     # ONNX models (Git LFS) + labels
│   └── videos/                         # Test videos (download separately)
├── train_yolov9_visdrone/              # Training outputs (full-frame + sliced)
├── test_results/                       # Evaluation CSVs and comparison reports
├── docs/                               # Full documentation
├── install.sh                          # One-step build and install
└── README.md
```

## Prerequisites

| Requirement | DeepStream 8.0 | DeepStream 9.0 |
|-------------|----------------|----------------|
| NVIDIA DeepStream SDK | 8.0 | 9.0 |
| CUDA Toolkit | 12.8 | 13.1 |
| TensorRT | 10.9.0 | 10.14.1 |
| GStreamer | 1.24.2 (ships with DeepStream) | 1.24.2 (ships with DeepStream) |
| Python Bindings | pyds 1.2.2 (pre-built) | built from source (`--build-bindings`) |


> The `install.sh` script auto-detects the DeepStream version and adapts the build steps accordingly.

## Modified DeepStream Libraries

### nvdsinfer — Smart Engine File Caching

The repository ships version-specific builds of the `nvdsinfer` library under `deepstream_source/libs/nvdsinfer_8.0/` and `deepstream_source/libs/nvdsinfer_9.0/`. The installer auto-detects the DeepStream version and copies the matching source into the build tree.

Both versions add intelligent TensorRT engine file naming and auto-discovery. Instead of rebuilding the `.engine` file on every pipeline start, the library generates a standardized name encoding batch size, input dimensions, GPU model, compute capability, TensorRT version, and precision:

```
{model}_b{batch}_i{W}x{H}_{compute_cap}_{gpu}_{trt_ver}_{precision}.engine
```

 
### nvdsinfer_yolo — YOLO Custom Bounding-Box Parser

Custom parsing functions for YOLO models exported with **EfficientNMS_TRT** and **EfficientNMSX_TRT + ROIAlign_TRT** TensorRT plugins.

## SAHI is a Full-Pipeline Strategy

The performance gains above come not just from the inference plugins, but from **adapting the entire training pipeline** to work with SAHI:

- The **sliced-448 model** was trained on pre-sliced images (fine-tuned from the full-640 model in just 40 epochs), achieving **mAP@0.5 of 0.859** on sliced validation data — compared to 0.485 for the full-frame model on full images.
- A smaller network input (448 vs 640) means **faster inference per slice**, while SAHI guarantees objects always appear at the right scale.
- With more training epochs and potentially even smaller inputs (320×320), there is significant room to improve both accuracy and throughput.

The key insight: **reducing the model input size does not hurt accuracy when combined with SAHI** — it improves throughput while SAHI handles the scale problem.

The GELAN-C architecture used here was chosen to isolate and validate the plugins, not because it is optimal for this use case. The real opportunity is lighter, purpose-built models. For instance, [Zhu & Xie (2026)](https://www.nature.com/articles/s41598-026-35301-2) achieve **+4.6% mAP50 on VisDrone while reducing parameters by ~8.5%** with an enhanced YOLOv11n. Combined with SAHI slicing, architectures like that could deliver better accuracy with significantly less GPU per inference — making this approach especially compelling for edge deployments.

See the [Training Guide](docs/TRAINING.md) for details.

## Known Limitations & Improvement Areas

A comprehensive technical review of `nvsahipostprocess` identified the
following areas where the plugin diverges from the SAHI Python reference
or has room for optimization. See [`docs/REVIEW.md`](docs/REVIEW.md) for
the full analysis with proposed solutions.

**Feature gaps**:
- **Instance-segmentation mask merge** — When detections are merged, the bounding box expands but segmentation masks are not updated or combined. Pipelines using YOLO-Seg will have misaligned masks after merge.
- **Cross-class label update** — When `class-agnostic=true`, the surviving detection's `obj_label` is never updated after merge, which can be incorrect in edge cases with equal scores.

**Algorithm divergences from SAHI Python**:
- The C++ plugin uses a **single-phase** GreedyNMM that merges inline (more aggressive), while SAHI Python uses a two-phase approach (candidate selection on original boxes, then re-check before merge).
- **Non-deterministic ordering** for detections with identical confidence scores (missing tie-breaking).

**Performance at scale** (multi-source, dense scenes):
- **O(n²) pair comparisons** — No spatial indexing; becomes a bottleneck above ~300 detections/frame. SAHI Python uses STRtree (R-tree) for ~O(n log n).
- **Sequential frame processing** — Frames in a batch are processed one by one; independent frames could be parallelized.
- **No per-class partitioning** — When `class-agnostic=false` (default), ~90% of pair comparisons are wasted on cross-class checks.

## Disclaimer

These plugins were developed as a proof of concept and have **not been thoroughly tested for all scenarios**. In particular:

- Only single-source pipelines have been validated. **Multi-source** configurations and other combinations with additional DeepStream components have not been tested.
- The code may contain bugs or limitations that surface in production environments or non-standard pipeline configurations.

This is an open-source project provided as-is. I do not have the availability to actively maintain it. **Contributions are welcome** — feel free to fork, fix, improve, and submit pull requests. Please respect the applicable licenses for each component.

## License

- **gst-nvsahipostprocess** — [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) (Copyright 2026 Levi Pereira)
- **gst-nvsahipreprocess** — [NVIDIA Proprietary](https://developer.nvidia.com/deepstream-eula) (derivative work of NVIDIA DeepStream SDK)
- **nvdsinfer**, **nvdsinfer_yolo** — [NVIDIA Proprietary](https://developer.nvidia.com/deepstream-eula)
- **python_test/** scripts — [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) (Copyright 2026 Levi Pereira)
- **python_test/common/** — [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) (Copyright NVIDIA Corporation)
