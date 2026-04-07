# Drone Deployment Guide — Jetson Orin + SAHI + GPS-Denied Navigation

This guide covers deploying the DeepStream SAHI pipeline on an NVIDIA Jetson Orin module for real-time surveillance on military-grade drones operating in GPS-denied environments.

## Target Configuration

| Component | Specification |
|-----------|--------------|
| **Compute** | NVIDIA Jetson Orin NX 16GB or AGX Orin |
| **Model** | GELAN-C (YOLOv9-C), INT8 quantized |
| **Camera** | NextVision EO/IR (1920×1080, H.264/H.265) |
| **Pipeline** | DeepStream SAHI (sliced inference + GreedyNMM) |
| **Precision** | INT8 + FP16 mixed (TensorRT) |
| **Navigation** | Visual-Inertial Odometry (GPS-denied) |
| **Datalink** | UDP metadata + periodic JPEG keyframes |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        JETSON ORIN                                  │
│                                                                     │
│  ┌──────────────┐                                                   │
│  │  NextVision   │─── H.264/IP ──┐                                  │
│  │  Camera       │               │                                  │
│  └──────────────┘               ▼                                  │
│                    ┌─────────────────────────────────┐              │
│                    │     DeepStream SAHI Pipeline     │              │
│                    │                                 │              │
│                    │  nvstreammux → nvsahipreprocess  │              │
│                    │  → nvinfer (INT8) → queue        │              │
│                    │  → nvsahipostprocess → nvtracker │              │
│                    │  → nvdsosd → fakesink            │              │
│                    └─────────┬───────────┬───────────┘              │
│                              │           │                          │
│                    ┌─────────▼──┐  ┌─────▼──────────┐              │
│                    │ Detection  │  │   Keyframe     │              │
│                    │ Transmitter│  │   Encoder      │              │
│                    │ (UDP:5000) │  │   (UDP:5001)   │              │
│                    └─────┬──────┘  └──────┬─────────┘              │
│                          │                │                        │
│  ┌──────────────┐        │                │                        │
│  │ VIO / SLAM   │────→ Geo-Registration   │                        │
│  │ (Isaac VSLAM)│        │                │                        │
│  └──────────────┘        │                │                        │
│                          ▼                ▼                        │
│                    ┌──────────────────────────┐                    │
│                    │   Tactical Radio Link    │                    │
│                    │   (~24 KB/s + ~10 KB/s)  │                    │
│                    └──────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Ground Control       │
                    │  Station (GCS)        │
                    │  - DetectionReceiver  │
                    │  - KeyframeReceiver   │
                    │  - Map overlay        │
                    └──────────────────────┘
```

## 1. Hardware Setup

### Jetson Orin NX 16GB

| Parameter | Recommended Setting |
|-----------|-------------------|
| Power mode | 25W (`sudo nvpmodel -m 2`) |
| Clock mode | Max performance (`sudo jetson_clocks`) |
| Fan profile | Max (`sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'`) |

For extended missions, consider the 15W mode to conserve battery:
```bash
# Mission power modes
sudo nvpmodel -m 0   # 15W — transit/cruise (lower FPS, longer endurance)
sudo nvpmodel -m 2   # 25W — active surveillance (full FPS)
sudo nvpmodel -m 6   # 40W Super Mode — maximum performance (if available)
```

### Camera Connection

NextVision cameras typically output H.264/H.265 via:
- **IP stream** (RTSP): Use `uridecodebin` with RTSP URI
- **HDMI capture**: Use `v4l2src` or `nvarguscamerasrc` with HDMI capture card

```python
# RTSP input (NextVision IP stream)
uri = "rtsp://192.168.1.10:554/stream"

# File input (testing)
uri = "file:///path/to/video.mp4"
```

## 2. INT8 Calibration Workflow

INT8 quantization requires a calibration dataset representative of the deployment environment.

### Step 1 — Collect Calibration Footage

Capture 5-10 minutes of flight footage from the actual deployment environment covering:
- Different altitudes (50m, 100m, 200m)
- Various lighting conditions (dawn, day, dusk)
- Target object types (people, vehicles, etc.)
- Different backgrounds (urban, rural, desert)

### Step 2 — Extract Calibration Frames

```bash
cd /apps/deepstream-sahi

python3 scripts/generate_int8_calibration.py \
    -i flight_footage_01.mp4 flight_footage_02.mp4 \
    -o python_test/deepstream-test-sahi/models/calibration/ \
    -n 500 \
    --size 640
```

This extracts 500 uniformly-distributed frames, resized to 640×640.

### Step 3 — Build INT8 Engine on Target Jetson

> **Critical:** The engine MUST be built on the target Jetson hardware. TensorRT performs device-specific kernel auto-tuning that is not transferable.

```bash
# Build INT8+FP16 mixed precision engine
scripts/build_int8_engine.sh

# Or FP16-only (no calibration needed)
scripts/build_int8_engine.sh --fp16-only

# Build with custom batch size
scripts/build_int8_engine.sh --batch 4
```

The engine file will be saved to `python_test/deepstream-test-sahi/models/` with a name encoding the GPU, TensorRT version, and precision.

### Step 4 — Update Config

Edit `config/pgie/visdrone-drone-int8.txt` and set `model-engine-file` to point to the generated engine:

```ini
model-engine-file=../../models/gelan-c-visdrone-full-frame-640-end2end_b8_i640x640_sm87_jetsononrinnx_trt1014_int8.engine
```

## 3. Running the Pipeline

### Basic Usage (Edge Mode)

```bash
cd /apps/deepstream-sahi/python_test/deepstream-test-sahi

source /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/pyds/bin/activate

# Run with INT8 model, 1080p input
python3 deepstream_test_sahi.py \
    --model visdrone-drone-int8 \
    --resolution 1080p \
    --no-display \
    -i rtsp://camera-ip:554/stream
```

### With Object Tracking

```bash
python3 deepstream_test_sahi.py \
    --model visdrone-drone-int8 \
    --resolution 1080p \
    --tracker \
    --no-display \
    -i /dev/video0
```

### Custom SAHI Parameters for Edge

```bash
# Reduced overlap for lower compute
python3 deepstream_test_sahi.py \
    --model visdrone-drone-int8 \
    --resolution 1080p \
    --overlap-w 0.15 --overlap-h 0.15 \
    --no-display \
    -i video.mp4
```

## 4. Data Transmission

### Detection Metadata

The detection transmitter sends compact binary metadata over UDP.

**Integration with the pipeline:**

```python
from detection_transmitter import DetectionTransmitter

# Initialize transmitter
tx = DetectionTransmitter(
    dest_host="192.168.1.100",  # GCS IP
    dest_port=5000,
    verbose=True,
)

# Attach to OSD sink pad
osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, tx.probe_callback, 0)
```

**GCS receiver:**

```bash
# On the ground control station
python3 detection_transmitter.py --receive --port 5000
```

**Bandwidth usage:**

| Detections/frame | 15 fps | 20 fps | 25 fps | 30 fps |
|-----------------|--------|--------|--------|--------|
| 20 | 6.6 KB/s | 8.8 KB/s | 11.0 KB/s | 13.2 KB/s |
| 50 | 15.2 KB/s | 20.3 KB/s | 25.3 KB/s | 30.4 KB/s |
| 100 | 29.6 KB/s | 39.5 KB/s | 49.4 KB/s | 59.2 KB/s |

### Keyframe Transmission

Periodic JPEG snapshots with detection overlays for operator SA.

```python
from keyframe_encoder import KeyframeEncoder

kf = KeyframeEncoder(
    dest_host="192.168.1.100",
    dest_port=5001,
    interval_sec=3.0,      # one keyframe every 3 seconds
    jpeg_quality=50,        # quality vs bandwidth tradeoff
    max_width=960,          # downscale to 960px width
    class_names=model_cfg["class_names"],
)

osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, kf.probe_callback, 0)
```

**GCS receiver:**

```bash
python3 keyframe_encoder.py --receive --port 5001 --output-dir received_keyframes/
```

## 5. GPS-Denied Geo-Registration

### Setup

```python
from geo_registration import GeoRegistrator

# Initialize with your NextVision camera calibration
geo = GeoRegistrator(
    fx=800.0, fy=800.0,       # focal length — from camera calibration
    cx=960.0, cy=540.0,       # principal point — typically image center
    image_width=1920,
    image_height=1080,
    gimbal_pitch_deg=-90.0,   # nadir (straight down)
)

# Connect to detection transmitter
tx.geo_registrator = geo
```

### Update Pose from VIO

Each frame, update the geo-registrator with the latest VIO pose:

```python
# From Isaac ROS Visual SLAM or VINS-Fusion (quaternion format)
geo.update_pose_from_quaternion(
    qw=pose.orientation.w,
    qx=pose.orientation.x,
    qy=pose.orientation.y,
    qz=pose.orientation.z,
    tx=pose.position.x,
    ty=pose.position.y,
    tz=pose.position.z,
    altitude_m=barometer.altitude_agl,
)
```

### Self-Test

```bash
python3 geo_registration.py --test
```

## 6. Resource Budget

### Memory (Jetson Orin NX 16GB)

| Component | GPU Memory | CPU Memory |
|-----------|-----------|------------|
| DeepStream + SAHI | ~2.5 GB | ~1.0 GB |
| GELAN-C INT8 engine | ~0.8 GB | — |
| VIO (Isaac VSLAM) | ~0.5 GB | ~1.0 GB |
| OS + drivers | ~1.0 GB | ~2.0 GB |
| **Total** | **~4.8 GB** | **~4.0 GB** |
| **Available** | **~11.2 GB** | **~12.0 GB** |

### Expected Performance (1080p input)

| Configuration | Slices/frame | Est. FPS (FP16) | Est. FPS (INT8) |
|--------------|-------------|----------------|----------------|
| 640×640, overlap=0.20 | 7 | ~22 fps | ~32 fps |
| 640×640, overlap=0.15 | 6 | ~25 fps | ~36 fps |
| 640×640, overlap=0.10 | 5 | ~28 fps | ~40 fps |

## 7. Monitoring & Thermal Management

### Real-Time Monitoring

```bash
# Monitor GPU/CPU/memory/temperature
tegrastats

# Monitor pipeline performance
GST_DEBUG=nvsahipostprocess:4 python3 deepstream_test_sahi.py ...
```

### Temperature Thresholds

| Temperature | Status | Action |
|------------|--------|--------|
| < 70°C | Normal | Full performance |
| 70-85°C | Warm | Verify cooling |
| 85-95°C | Hot | GPU throttling begins |
| > 95°C | Critical | Auto-shutdown |

### Thermal Mitigation

1. **Heatsink**: mandatory on Orin NX for drone deployment
2. **Active cooling**: propeller wash provides airflow in flight
3. **Power mode switching**: reduce to 15W during transit
4. **Thermal padding**: between Orin module and drone frame

## 8. Pre-Flight Checklist

- [ ] INT8 engine built on this specific Jetson unit
- [ ] Camera RTSP/HDMI feed verified
- [ ] `tegrastats` shows <8 GB memory usage
- [ ] Detection transmitter UDP test passed
- [ ] Keyframe receiver validates images at GCS
- [ ] VIO calibrated and producing valid poses
- [ ] Power mode set (25W for mission)
- [ ] Fan profile set to max
- [ ] GPS-denied: VIO EKF converged before takeoff

## Files Reference

| File | Purpose |
|------|---------|
| `config/pgie/visdrone-drone-int8.txt` | INT8 nvinfer config |
| `config/preprocess/preprocess_640_edge.txt` | Edge SAHI preprocess (batch=8, VIC) |
| `scripts/generate_int8_calibration.py` | Extract calibration frames |
| `scripts/build_int8_engine.sh` | Build TensorRT INT8 engine |
| `detection_transmitter.py` | Binary metadata over UDP |
| `keyframe_encoder.py` | JPEG keyframes over UDP |
| `geo_registration.py` | VIO-based ground projection |
