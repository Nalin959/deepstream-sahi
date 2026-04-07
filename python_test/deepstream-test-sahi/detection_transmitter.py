#!/usr/bin/env python3

################################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Detection Metadata Transmitter for Drone Deployment
#
# Extracts object detection results from the DeepStream pipeline and
# transmits compact binary metadata over UDP for real-time situational
# awareness at the ground control station.
#
# Wire format:
#   Frame header (14 bytes):
#     u64  timestamp_us      — microseconds since epoch
#     u32  frame_number
#     u16  num_detections
#
#   Per detection (15 bytes):
#     u8   class_id
#     u16  confidence         — fixed-point (conf × 10000)
#     u16  bbox_left
#     u16  bbox_top
#     u16  bbox_width
#     u16  bbox_height
#     u32  tracker_id
#
# Bandwidth: 50 detections @ 30fps = ~22 KB/s
#
# Usage (standalone test):
#   python3 detection_transmitter.py --test
#
# Usage (integrated with DeepStream pipeline):
#   from detection_transmitter import DetectionTransmitter
#   tx = DetectionTransmitter(dest_host="192.168.1.100", dest_port=5000)
#   osd_pad.add_probe(Gst.PadProbeType.BUFFER, tx.probe_callback, 0)
################################################################################

import struct
import socket
import time
import sys
import argparse
import json


# ─── Wire Format ─────────────────────────────────────────────────────────────

# Header: timestamp_us (u64) + frame_num (u32) + num_dets (u16) = 14 bytes
HEADER_FMT = "<QIH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# Detection: class_id (u8) + conf (u16) + left,top,w,h (4×u16) + track_id (u32)
DET_FMT = "<BHHHHHI"
DET_SIZE = struct.calcsize(DET_FMT)

# Maximum UDP payload (stay well under MTU to avoid fragmentation)
MAX_UDP_PAYLOAD = 1400


class Detection:
    """Lightweight detection data container."""

    __slots__ = ("class_id", "confidence", "left", "top", "width", "height",
                 "tracker_id", "world_x", "world_y", "world_z")

    def __init__(self, class_id=0, confidence=0.0,
                 left=0, top=0, width=0, height=0,
                 tracker_id=0):
        self.class_id = class_id
        self.confidence = confidence
        self.left = int(left)
        self.top = int(top)
        self.width = int(width)
        self.height = int(height)
        self.tracker_id = tracker_id
        # Optional geo-registration fields (set by geo_registration.py)
        self.world_x = 0.0
        self.world_y = 0.0
        self.world_z = 0.0

    def __repr__(self):
        return (f"Detection(cls={self.class_id}, conf={self.confidence:.2f}, "
                f"bbox=[{self.left},{self.top},{self.width},{self.height}], "
                f"track={self.tracker_id})")


class DetectionTransmitter:
    """Transmits detection metadata over UDP.

    Designed to be attached as a GStreamer pad probe to the OSD sink pad.
    For each frame, it extracts all NvDsObjectMeta, packs them into a
    compact binary message, and sends via UDP to the ground control station.

    Args:
        dest_host:  Destination IP address (GCS).
        dest_port:  Destination UDP port.
        enabled:    If False, probe passes through without transmitting.
        max_dets:   Maximum detections to transmit per frame (bandwidth cap).
        verbose:    Print transmission stats periodically.
    """

    def __init__(self, dest_host="127.0.0.1", dest_port=5000,
                 enabled=True, max_dets=200, verbose=False):
        self.dest_host = dest_host
        self.dest_port = dest_port
        self.enabled = enabled
        self.max_dets = max_dets
        self.verbose = verbose

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._frames_sent = 0
        self._bytes_sent = 0
        self._last_stats_time = time.monotonic()

        # Optional geo-registrator (set externally)
        self.geo_registrator = None

    def pack_frame(self, frame_number, detections):
        """Pack a frame's detections into binary wire format.

        Args:
            frame_number: Frame sequence number.
            detections:   List of Detection objects.

        Returns:
            bytes: Packed binary message.
        """
        timestamp_us = int(time.time() * 1_000_000)
        dets = detections[:self.max_dets]
        num_dets = len(dets)

        # Pre-allocate buffer
        buf = bytearray(HEADER_SIZE + num_dets * DET_SIZE)

        # Pack header
        struct.pack_into(HEADER_FMT, buf, 0,
                         timestamp_us, frame_number, num_dets)

        # Pack detections
        offset = HEADER_SIZE
        for det in dets:
            conf_fp = int(det.confidence * 10000)
            struct.pack_into(DET_FMT, buf, offset,
                             det.class_id,
                             min(conf_fp, 65535),
                             min(det.left, 65535),
                             min(det.top, 65535),
                             min(det.width, 65535),
                             min(det.height, 65535),
                             det.tracker_id)
            offset += DET_SIZE

        return bytes(buf)

    @staticmethod
    def unpack_frame(data):
        """Unpack binary wire format back to frame header and detections.

        Args:
            data: Raw bytes received.

        Returns:
            tuple: (timestamp_us, frame_number, list[Detection])
        """
        timestamp_us, frame_number, num_dets = struct.unpack_from(
            HEADER_FMT, data, 0)

        detections = []
        offset = HEADER_SIZE
        for _ in range(num_dets):
            (class_id, conf_fp, left, top, width, height,
             tracker_id) = struct.unpack_from(DET_FMT, data, offset)
            det = Detection(
                class_id=class_id,
                confidence=conf_fp / 10000.0,
                left=left, top=top,
                width=width, height=height,
                tracker_id=tracker_id,
            )
            detections.append(det)
            offset += DET_SIZE

        return timestamp_us, frame_number, detections

    def send(self, frame_number, detections):
        """Pack and send detections for a single frame.

        Splits into multiple UDP packets if necessary to stay under MTU.
        """
        if not self.enabled:
            return

        data = self.pack_frame(frame_number, detections)

        # Split into chunks if exceeding safe UDP payload
        if len(data) <= MAX_UDP_PAYLOAD:
            self._sock.sendto(data, (self.dest_host, self.dest_port))
            self._bytes_sent += len(data)
        else:
            # Send header + as many detections as fit per packet
            max_dets_per_pkt = (MAX_UDP_PAYLOAD - HEADER_SIZE) // DET_SIZE
            for i in range(0, len(detections), max_dets_per_pkt):
                chunk = detections[i:i + max_dets_per_pkt]
                pkt = self.pack_frame(frame_number, chunk)
                self._sock.sendto(pkt, (self.dest_host, self.dest_port))
                self._bytes_sent += len(pkt)

        self._frames_sent += 1

        if self.verbose:
            now = time.monotonic()
            if now - self._last_stats_time >= 5.0:
                elapsed = now - self._last_stats_time
                rate = self._bytes_sent / elapsed / 1024
                print(f"[TX] {self._frames_sent} frames, "
                      f"{self._bytes_sent} bytes, "
                      f"{rate:.1f} KB/s")
                self._bytes_sent = 0
                self._last_stats_time = now

    def extract_detections_from_frame_meta(self, frame_meta):
        """Extract Detection objects from an NvDsFrameMeta.

        Args:
            frame_meta: pyds.NvDsFrameMeta from the DeepStream pipeline.

        Returns:
            list[Detection]: Extracted detections.
        """
        # Import pyds only when actually used with DeepStream
        import pyds

        detections = []
        l_obj = frame_meta.obj_meta_list

        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            rect = obj_meta.rect_params
            det = Detection(
                class_id=obj_meta.class_id,
                confidence=obj_meta.confidence,
                left=rect.left,
                top=rect.top,
                width=rect.width,
                height=rect.height,
                tracker_id=obj_meta.object_id,
            )
            detections.append(det)

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        return detections

    def probe_callback(self, pad, info, u_data):
        """GStreamer pad probe for transmitting detections.

        Attach to the OSD sink pad (or any pad after nvsahipostprocess):
            pad.add_probe(Gst.PadProbeType.BUFFER, tx.probe_callback, 0)
        """
        import pyds
        from gi.repository import Gst

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list

        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            detections = self.extract_detections_from_frame_meta(frame_meta)

            # Optional: enrich with geo-registration
            if self.geo_registrator is not None:
                self.geo_registrator.enrich_detections(detections)

            self.send(frame_meta.frame_num, detections)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def close(self):
        """Close the UDP socket."""
        if self._sock:
            self._sock.close()


# ─── Receiver (for GCS side) ────────────────────────────────────────────────

class DetectionReceiver:
    """UDP receiver for detection metadata at the ground control station.

    Usage:
        rx = DetectionReceiver(listen_port=5000)
        for timestamp, frame_num, detections in rx.receive():
            print(f"Frame {frame_num}: {len(detections)} detections")
    """

    def __init__(self, listen_port=5000, listen_host="0.0.0.0"):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((listen_host, listen_port))
        self._sock.settimeout(1.0)

    def receive(self):
        """Generator that yields (timestamp_us, frame_number, detections)."""
        while True:
            try:
                data, addr = self._sock.recvfrom(65535)
                yield DetectionTransmitter.unpack_frame(data)
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                break

    def close(self):
        self._sock.close()


# ─── Self-Test ───────────────────────────────────────────────────────────────

def _run_self_test():
    """Verify pack/unpack roundtrip and print bandwidth estimate."""
    import random

    print("═══ Detection Transmitter — Self Test ═══\n")

    # Create sample detections
    detections = []
    for i in range(50):
        det = Detection(
            class_id=random.randint(0, 10),
            confidence=random.uniform(0.3, 0.99),
            left=random.randint(0, 1920),
            top=random.randint(0, 1080),
            width=random.randint(20, 200),
            height=random.randint(20, 200),
            tracker_id=random.randint(1, 1000),
        )
        detections.append(det)

    tx = DetectionTransmitter()

    # Pack
    data = tx.pack_frame(frame_number=42, detections=detections)
    print(f"Packed {len(detections)} detections → {len(data)} bytes")
    print(f"  Header: {HEADER_SIZE} bytes")
    print(f"  Per-detection: {DET_SIZE} bytes")
    print(f"  Total: {HEADER_SIZE} + {len(detections)} × {DET_SIZE} = {len(data)} bytes")

    # Unpack
    ts, frame_num, unpacked = DetectionTransmitter.unpack_frame(data)
    assert frame_num == 42, f"Frame number mismatch: {frame_num}"
    assert len(unpacked) == 50, f"Detection count mismatch: {len(unpacked)}"

    # Verify roundtrip
    for orig, recv in zip(detections, unpacked):
        assert orig.class_id == recv.class_id
        assert abs(orig.confidence - recv.confidence) < 0.001
        assert orig.left == recv.left
        assert orig.tracker_id == recv.tracker_id

    print(f"  Roundtrip: ✅ PASS\n")

    # Bandwidth estimate
    fps_values = [15, 20, 25, 30]
    det_counts = [20, 50, 100, 200]
    print("Bandwidth Estimates:")
    print(f"  {'Dets/frame':>12s}  {'15 fps':>10s}  {'20 fps':>10s}  "
          f"{'25 fps':>10s}  {'30 fps':>10s}")
    for nd in det_counts:
        frame_bytes = HEADER_SIZE + nd * DET_SIZE
        row = f"  {nd:>12d}"
        for fps in fps_values:
            kbps = frame_bytes * fps / 1024
            row += f"  {kbps:>8.1f} KB/s"
        print(row)

    print(f"\n  Max detections in one UDP packet (MTU-safe): "
          f"{(MAX_UDP_PAYLOAD - HEADER_SIZE) // DET_SIZE}")
    print("\n═══ Self Test Complete ═══")

    tx.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detection metadata transmitter for drone deployment")
    parser.add_argument("--test", action="store_true",
                        help="Run self-test (pack/unpack roundtrip)")
    parser.add_argument("--receive", action="store_true",
                        help="Run as receiver (GCS mode)")
    parser.add_argument("--port", type=int, default=5000,
                        help="UDP port (default: 5000)")
    args = parser.parse_args()

    if args.test:
        _run_self_test()
    elif args.receive:
        print(f"Listening on UDP port {args.port}...")
        rx = DetectionReceiver(listen_port=args.port)
        try:
            for ts, frame_num, dets in rx.receive():
                ts_sec = ts / 1_000_000
                print(f"Frame {frame_num} | {len(dets)} dets | "
                      f"t={time.strftime('%H:%M:%S', time.localtime(ts_sec))}")
                for d in dets[:5]:
                    print(f"  {d}")
                if len(dets) > 5:
                    print(f"  ... and {len(dets) - 5} more")
        except KeyboardInterrupt:
            pass
        rx.close()
    else:
        parser.print_help()
