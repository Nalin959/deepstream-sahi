#!/usr/bin/env python3

################################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Keyframe Encoder for Drone Deployment
#
# Periodically captures video frames from the DeepStream pipeline, overlays
# detection bounding boxes, JPEG-compresses them, and transmits to the ground
# control station for operator situational awareness.
#
# Typical bandwidth:
#   1080p JPEG (quality=50) ≈ 25-40 KB/frame
#   At 3-second interval ≈ 10 KB/s average
#
# Usage (integrated with DeepStream pipeline):
#   from keyframe_encoder import KeyframeEncoder
#   kf = KeyframeEncoder(dest_host="192.168.1.100", dest_port=5001)
#   osd_pad.add_probe(Gst.PadProbeType.BUFFER, kf.probe_callback, 0)
################################################################################

import struct
import socket
import time
import sys

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None


# ─── Keyframe Wire Format ───────────────────────────────────────────────────
#
# Since keyframes exceed UDP MTU, we split into chunks:
#
# Chunk header (16 bytes):
#   u32  frame_number
#   u64  timestamp_us
#   u16  chunk_index        — 0-based chunk index
#   u16  total_chunks       — total chunks in this keyframe
#
# Chunk payload: up to 1384 bytes of JPEG data
#   (1400 - 16 = 1384 bytes per chunk)

CHUNK_HEADER_FMT = "<QI HH"
CHUNK_HEADER_SIZE = struct.calcsize(CHUNK_HEADER_FMT)
CHUNK_PAYLOAD_SIZE = 1400 - CHUNK_HEADER_SIZE

# Class colors for drawing (BGR for OpenCV)
CLASS_COLORS_BGR = [
    (102, 255, 0),    # 0  green
    (255, 204, 0),    # 1  cyan
    (0, 217, 255),    # 2  yellow
    (255, 153, 76),   # 3  blue
    (0, 128, 255),    # 4  orange
    (76, 51, 255),    # 5  red
    (255, 102, 204),  # 6  purple
    (191, 140, 255),  # 7  pink
    (102, 255, 102),  # 8  lime
    (178, 178, 178),  # 9  gray
    (76, 191, 217),   # 10 gold
]


class KeyframeEncoder:
    """Captures and transmits periodic keyframes with detection overlays.

    Designed to be attached as a GStreamer pad probe alongside the
    DetectionTransmitter. Operates on a timer — only captures a frame
    when the configured interval has elapsed.

    Args:
        dest_host:       Destination IP address (GCS).
        dest_port:       Destination UDP port (separate from detection metadata).
        interval_sec:    Seconds between keyframes (default: 3.0).
        jpeg_quality:    JPEG compression quality 1-100 (default: 50).
        max_width:       Downscale frame to this width before encoding (default: 960).
        draw_detections: Overlay detection boxes on the keyframe.
        enabled:         If False, probe passes through without capturing.
        class_names:     List of class name strings for labeling.
    """

    def __init__(self, dest_host="127.0.0.1", dest_port=5001,
                 interval_sec=3.0, jpeg_quality=50, max_width=960,
                 draw_detections=True, enabled=True, class_names=None):
        self.dest_host = dest_host
        self.dest_port = dest_port
        self.interval_sec = interval_sec
        self.jpeg_quality = jpeg_quality
        self.max_width = max_width
        self.draw_detections = draw_detections
        self.enabled = enabled
        self.class_names = class_names or []

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._last_capture_time = 0.0
        self._keyframes_sent = 0

    def _should_capture(self):
        """Check if enough time has elapsed for next keyframe."""
        now = time.monotonic()
        if now - self._last_capture_time >= self.interval_sec:
            self._last_capture_time = now
            return True
        return False

    def draw_boxes(self, frame, detections):
        """Draw detection bounding boxes on the frame.

        Args:
            frame:      numpy array (H, W, 3) BGR image.
            detections: list of Detection objects from detection_transmitter.py.

        Returns:
            numpy array: Frame with overlays drawn.
        """
        if cv2 is None:
            return frame

        h, w = frame.shape[:2]

        for det in detections:
            color = CLASS_COLORS_BGR[det.class_id % len(CLASS_COLORS_BGR)]
            x1 = max(0, det.left)
            y1 = max(0, det.top)
            x2 = min(w, det.left + det.width)
            y2 = min(h, det.top + det.height)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"{det.class_id}"
            if det.class_id < len(self.class_names):
                label = self.class_names[det.class_id]
            label += f" {det.confidence:.0%}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.4, 1)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1),
                          color, -1)
            cv2.putText(frame, label, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # HUD: timestamp + detection count
        hud = (f"{time.strftime('%H:%M:%S')} | "
               f"{len(detections)} dets | KF #{self._keyframes_sent}")
        cv2.putText(frame, hud, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def encode_and_send(self, frame, frame_number, detections=None):
        """Encode frame as JPEG and transmit via UDP chunks.

        Args:
            frame:        numpy array (H, W, 3) BGR image.
            frame_number: Frame sequence number.
            detections:   Optional list of Detection objects to overlay.
        """
        if cv2 is None:
            sys.stderr.write("Warning: OpenCV not available, skipping keyframe\n")
            return

        # Downscale if needed
        h, w = frame.shape[:2]
        if w > self.max_width:
            scale = self.max_width / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h),
                               interpolation=cv2.INTER_AREA)
            # Scale detection coordinates if drawing
            if detections and self.draw_detections:
                for det in detections:
                    det.left = int(det.left * scale)
                    det.top = int(det.top * scale)
                    det.width = int(det.width * scale)
                    det.height = int(det.height * scale)

        # Draw detection overlays
        if detections and self.draw_detections:
            frame = self.draw_boxes(frame, detections)

        # JPEG encode
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        success, jpeg_data = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            sys.stderr.write("Warning: JPEG encode failed\n")
            return

        jpeg_bytes = jpeg_data.tobytes()
        timestamp_us = int(time.time() * 1_000_000)

        # Split into UDP chunks
        total_chunks = (len(jpeg_bytes) + CHUNK_PAYLOAD_SIZE - 1) // CHUNK_PAYLOAD_SIZE

        for i in range(total_chunks):
            start = i * CHUNK_PAYLOAD_SIZE
            end = min(start + CHUNK_PAYLOAD_SIZE, len(jpeg_bytes))
            payload = jpeg_bytes[start:end]

            header = struct.pack(CHUNK_HEADER_FMT,
                                 timestamp_us, frame_number,
                                 i, total_chunks)
            self._sock.sendto(header + payload,
                              (self.dest_host, self.dest_port))

        self._keyframes_sent += 1

    def probe_callback(self, pad, info, u_data):
        """GStreamer pad probe for periodic keyframe capture.

        Attach to the OSD sink pad:
            pad.add_probe(Gst.PadProbeType.BUFFER, kf.probe_callback, 0)
        """
        if not self.enabled or not self._should_capture():
            # Import here to avoid dependency when not in DeepStream context
            from gi.repository import Gst
            return Gst.PadProbeReturn.OK

        import pyds
        from gi.repository import Gst

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list

        if l_frame is None:
            return Gst.PadProbeReturn.OK

        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            return Gst.PadProbeReturn.OK

        # Get the frame surface
        frame_number = frame_meta.frame_num

        # Extract numpy frame from NVMM surface
        try:
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer),
                                                frame_meta.batch_id)
            frame = np.array(n_frame, copy=True)
            # Convert RGBA to BGR for OpenCV
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        except Exception as e:
            sys.stderr.write(f"Warning: Cannot get frame surface: {e}\n")
            return Gst.PadProbeReturn.OK

        # Extract detections for overlay
        detections = []
        if self.draw_detections:
            from detection_transmitter import Detection
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    rect = obj_meta.rect_params
                    det = Detection(
                        class_id=obj_meta.class_id,
                        confidence=obj_meta.confidence,
                        left=rect.left, top=rect.top,
                        width=rect.width, height=rect.height,
                        tracker_id=obj_meta.object_id,
                    )
                    detections.append(det)
                except StopIteration:
                    break
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

        self.encode_and_send(frame, frame_number, detections)

        return Gst.PadProbeReturn.OK

    def close(self):
        """Close the UDP socket."""
        if self._sock:
            self._sock.close()


# ─── Keyframe Receiver (GCS side) ───────────────────────────────────────────

class KeyframeReceiver:
    """Reassembles chunked keyframes received over UDP.

    Usage:
        rx = KeyframeReceiver(listen_port=5001)
        for frame_num, jpeg_data in rx.receive():
            with open(f'frame_{frame_num}.jpg', 'wb') as f:
                f.write(jpeg_data)
    """

    def __init__(self, listen_port=5001, listen_host="0.0.0.0"):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((listen_host, listen_port))
        self._sock.settimeout(2.0)
        self._buffers = {}  # (frame_num, ts) → {chunk_idx: payload}
        self._expected = {}  # (frame_num, ts) → total_chunks

    def receive(self):
        """Generator that yields (frame_number, jpeg_bytes) for complete keyframes."""
        while True:
            try:
                data, addr = self._sock.recvfrom(65535)
                if len(data) < CHUNK_HEADER_SIZE:
                    continue

                ts, frame_num, chunk_idx, total = struct.unpack_from(
                    CHUNK_HEADER_FMT, data, 0)
                payload = data[CHUNK_HEADER_SIZE:]

                key = (frame_num, ts)
                if key not in self._buffers:
                    self._buffers[key] = {}
                    self._expected[key] = total

                self._buffers[key][chunk_idx] = payload

                # Check if complete
                if len(self._buffers[key]) == self._expected[key]:
                    # Reassemble in order
                    jpeg = b"".join(
                        self._buffers[key][i]
                        for i in range(total)
                    )
                    del self._buffers[key]
                    del self._expected[key]
                    yield frame_num, jpeg

                # Clean up stale incomplete buffers (older than 10s)
                stale = [k for k in self._buffers
                         if k[1] < ts - 10_000_000]
                for k in stale:
                    del self._buffers[k]
                    del self._expected[k]

            except socket.timeout:
                continue
            except KeyboardInterrupt:
                break

    def close(self):
        self._sock.close()


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Keyframe encoder/receiver for drone deployment")
    parser.add_argument("--receive", action="store_true",
                        help="Run as receiver (GCS mode, saves JPEG files)")
    parser.add_argument("--port", type=int, default=5001,
                        help="UDP port (default: 5001)")
    parser.add_argument("--output-dir", default="received_keyframes",
                        help="Directory to save received keyframes")
    args = parser.parse_args()

    if args.receive:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Receiving keyframes on UDP port {args.port}...")
        print(f"Saving to: {args.output_dir}")
        rx = KeyframeReceiver(listen_port=args.port)
        try:
            for frame_num, jpeg_data in rx.receive():
                path = os.path.join(args.output_dir,
                                    f"keyframe_{frame_num:06d}.jpg")
                with open(path, "wb") as f:
                    f.write(jpeg_data)
                print(f"Saved {path} ({len(jpeg_data)} bytes)")
        except KeyboardInterrupt:
            pass
        rx.close()
    else:
        parser.print_help()
        print("\nNote: To send keyframes, use as a DeepStream probe.")
        print("See module docstring for integration instructions.")
