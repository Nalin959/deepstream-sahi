#!/usr/bin/env python3

################################################################################
# INT8 Calibration Frame Extractor
#
# Extracts representative frames from video files for TensorRT INT8
# quantization calibration. The output frames are saved as individual
# images that can be used with trtexec --calib or a custom IInt8Calibrator.
#
# Usage:
#   python3 generate_int8_calibration.py \
#       -i video1.mp4 video2.mp4 \
#       -o calibration_frames/ \
#       -n 300 \
#       --size 640
#
# For drone deployment:
#   python3 generate_int8_calibration.py \
#       -i operational_footage/*.mp4 \
#       -o models/calibration/ \
#       -n 500 \
#       --size 640
################################################################################

import argparse
import os
import sys
import random

try:
    import cv2
    import numpy as np
except ImportError:
    sys.stderr.write(
        "Error: opencv-python and numpy are required.\n"
        "Install with: pip install opencv-python numpy\n"
    )
    sys.exit(1)


def extract_frames(video_path, num_frames, resize_to=None):
    """Extract uniformly-spaced frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.stderr.write(f"Warning: Cannot open {video_path}, skipping.\n")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        sys.stderr.write(f"Warning: Cannot read frame count from {video_path}, skipping.\n")
        cap.release()
        return []

    # Select uniformly-spaced frame indices
    if num_frames >= total_frames:
        indices = list(range(total_frames))
    else:
        indices = sorted(random.sample(range(total_frames), num_frames))

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        if resize_to is not None:
            frame = cv2.resize(frame, (resize_to, resize_to),
                               interpolation=cv2.INTER_LINEAR)
        frames.append(frame)

    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser(
        description="Extract calibration frames for TensorRT INT8 quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", nargs="+", required=True,
        help="Input video files (supports multiple files and globs)"
    )
    parser.add_argument(
        "-o", "--output-dir", default="models/calibration",
        help="Output directory for calibration frames (default: models/calibration)"
    )
    parser.add_argument(
        "-n", "--num-frames", type=int, default=300,
        help="Total number of calibration frames to extract (default: 300)"
    )
    parser.add_argument(
        "--size", type=int, default=640,
        help="Resize frames to this square size (default: 640, matching model input)"
    )
    parser.add_argument(
        "--format", choices=["png", "ppm", "jpg"], default="ppm",
        help="Output image format (default: ppm, fastest for TensorRT calibrator)"
    )
    parser.add_argument(
        "--no-resize", action="store_true",
        help="Do not resize frames (save at original resolution)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible frame selection (default: 42)"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Distribute requested frames across input videos
    num_videos = len(args.input)
    frames_per_video = max(1, args.num_frames // num_videos)
    remainder = args.num_frames - (frames_per_video * num_videos)

    resize_to = None if args.no_resize else args.size

    all_frames = []
    for i, video_path in enumerate(args.input):
        n = frames_per_video + (1 if i < remainder else 0)
        print(f"Extracting {n} frames from: {video_path}")
        frames = extract_frames(video_path, n, resize_to=resize_to)
        all_frames.extend(frames)
        print(f"  → Got {len(frames)} frames")

    if not all_frames:
        sys.stderr.write("Error: No frames extracted from any input.\n")
        sys.exit(1)

    # Shuffle for diversity in calibration batches
    random.shuffle(all_frames)

    # Save frames
    print(f"\nSaving {len(all_frames)} calibration frames to: {args.output_dir}")
    calib_list_path = os.path.join(args.output_dir, "calibration_list.txt")

    with open(calib_list_path, "w") as f:
        for idx, frame in enumerate(all_frames):
            filename = f"calib_{idx:04d}.{args.format}"
            filepath = os.path.join(args.output_dir, filename)
            cv2.imwrite(filepath, frame)
            f.write(f"{filepath}\n")

    print(f"Calibration list: {calib_list_path}")
    print(f"Total frames: {len(all_frames)}")
    print(f"Frame size: {'original' if resize_to is None else f'{resize_to}x{resize_to}'}")
    print(f"Format: {args.format}")
    print(f"\nNext steps:")
    print(f"  1. Copy frames to Jetson target")
    print(f"  2. Run: scripts/build_int8_engine.sh")
    print(f"     or:  trtexec --onnx=model.onnx --int8 --calib={calib_list_path}")


if __name__ == "__main__":
    main()
