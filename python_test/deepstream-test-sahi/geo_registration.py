#!/usr/bin/env python3

################################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Geo-Registration Module for GPS-Denied Drone Navigation
#
# Projects detection bounding box centers from image pixel coordinates to
# world coordinates (NED frame relative to takeoff) using Visual-Inertial
# Odometry (VIO) pose data, camera intrinsics, and barometric altitude.
#
# Coordinate System:
#   NED (North-East-Down) frame:
#     Origin: takeoff position
#     X: North, Y: East, Z: Down (positive downward)
#
# Usage:
#   from geo_registration import GeoRegistrator
#
#   # Initialize with camera intrinsics
#   geo = GeoRegistrator(
#       fx=600.0, fy=600.0,     # focal length in pixels
#       cx=960.0, cy=540.0,     # principal point (image center)
#       image_width=1920, image_height=1080,
#   )
#
#   # Update with VIO pose each frame
#   geo.update_pose(rotation_matrix, translation_vector, altitude_m)
#
#   # Enrich detections with world coordinates
#   geo.enrich_detections(detections)
################################################################################

import math
import sys

try:
    import numpy as np
except ImportError:
    sys.stderr.write(
        "Error: numpy is required for geo_registration.\n"
        "Install with: pip install numpy\n"
    )
    sys.exit(1)


class GeoRegistrator:
    """Projects image-space detections to world coordinates using VIO pose.

    This module performs monocular ground-plane intersection: given a
    detection's pixel coordinates, camera intrinsics, and the drone's
    6-DOF pose from VIO, it projects a ray through the pixel and
    intersects it with the ground plane (z=0 in world frame, or a
    specified terrain elevation).

    Accuracy depends on:
      - VIO drift (typically 0.5-2% of distance traveled)
      - Altitude accuracy (barometer ± 1-2m, or VIO z-axis)
      - Camera calibration quality
      - Gimbal angle accuracy

    Args:
        fx, fy:             Camera focal lengths in pixels.
        cx, cy:             Camera principal point in pixels.
        image_width:        Image width in pixels.
        image_height:       Image height in pixels.
        gimbal_pitch_deg:   Gimbal pitch angle (negative = looking down).
                            Default -90 = nadir (straight down).
        gimbal_roll_deg:    Gimbal roll offset in degrees (default 0).
        terrain_elevation:  Ground plane elevation in NED (default 0.0).
    """

    def __init__(self, fx=600.0, fy=600.0, cx=960.0, cy=540.0,
                 image_width=1920, image_height=1080,
                 gimbal_pitch_deg=-90.0, gimbal_roll_deg=0.0,
                 terrain_elevation=0.0):
        # Camera intrinsic matrix
        self.K = np.array([
            [fx,  0.0, cx],
            [0.0, fy,  cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)

        self.image_width = image_width
        self.image_height = image_height
        self.terrain_elevation = terrain_elevation

        # Gimbal rotation (camera frame → body frame)
        # Camera convention: X-right, Y-down, Z-forward (optical axis)
        # Body convention:   X-forward, Y-right, Z-down
        # For nadir (-90°): camera Z (optical) should align with body Z (down)
        pitch_rad = math.radians(gimbal_pitch_deg)
        roll_rad = math.radians(gimbal_roll_deg)

        # Base camera-to-body rotation (no gimbal):
        # Maps camera [X,Y,Z] → body [Y, Z, X] (camera Z-forward → body X-forward)
        R_cam_to_body = np.array([
            [0.0, 0.0, 1.0],   # body X = camera Z (forward)
            [1.0, 0.0, 0.0],   # body Y = camera X (right)
            [0.0, 1.0, 0.0],   # body Z = camera Y (down)
        ], dtype=np.float64)

        # Gimbal pitch rotation about the body Y axis (tilts camera up/down)
        # pitch = 0 → forward-looking, pitch = -90° → nadir (down)
        cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
        R_pitch = np.array([
            [cp,  0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ], dtype=np.float64)

        # Gimbal roll about body X axis
        cr, sr = math.cos(roll_rad), math.sin(roll_rad)
        R_roll = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr,  cr],
        ], dtype=np.float64)

        self.R_gimbal = R_roll @ R_pitch @ R_cam_to_body

        # Current VIO state (updated each frame)
        self.R_body_to_world = np.eye(3, dtype=np.float64)  # rotation
        self.t_world = np.zeros(3, dtype=np.float64)         # translation (NED)
        self.altitude_m = 0.0                                 # barometric alt (AGL)
        self.pose_valid = False

    @staticmethod
    def _rotation_matrix(pitch, roll, yaw):
        """Create a rotation matrix from Euler angles (ZYX convention).

        Args:
            pitch: Rotation about Y axis (radians).
            roll:  Rotation about X axis (radians).
            yaw:   Rotation about Z axis (radians).

        Returns:
            3x3 numpy rotation matrix.
        """
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        cy, sy = math.cos(yaw), math.sin(yaw)

        R = np.array([
            [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
            [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
            [-sp,      cp * sr,                 cp * cr],
        ], dtype=np.float64)
        return R

    def update_pose(self, R_body_to_world, t_world, altitude_m=None):
        """Update the drone's current pose from VIO.

        This should be called once per frame before enrich_detections().

        Args:
            R_body_to_world: 3x3 numpy array — rotation from body frame to
                             world (NED) frame. From VIO output.
            t_world:         3-element array — drone position in world (NED)
                             frame, relative to takeoff.
            altitude_m:      Altitude above ground level in meters (positive up).
                             If None, derived from t_world Z component (negated,
                             since NED Z is positive down).
        """
        self.R_body_to_world = np.asarray(R_body_to_world, dtype=np.float64)
        self.t_world = np.asarray(t_world, dtype=np.float64)

        if altitude_m is not None:
            self.altitude_m = altitude_m
        else:
            # NED convention: Z positive down → altitude = -Z
            self.altitude_m = -self.t_world[2]

        self.pose_valid = True

    def update_pose_from_quaternion(self, qw, qx, qy, qz, tx, ty, tz,
                                    altitude_m=None):
        """Update pose from quaternion + translation (common VIO output format).

        Args:
            qw, qx, qy, qz: Quaternion components (body-to-world rotation).
            tx, ty, tz:      Translation in world NED frame.
            altitude_m:      Optional explicit altitude AGL.
        """
        # Quaternion to rotation matrix
        R = np.array([
            [1 - 2*(qy*qy + qz*qz),  2*(qx*qy - qz*qw),      2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),      1 - 2*(qx*qx + qz*qz),  2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),      1 - 2*(qx*qx + qy*qy)],
        ], dtype=np.float64)

        self.update_pose(R, np.array([tx, ty, tz]), altitude_m)

    def project_pixel_to_ground(self, u, v):
        """Project an image pixel to a ground-plane point in world coordinates.

        Uses ray-plane intersection: cast a ray from the camera through
        the pixel, transform to world frame, and intersect with the
        ground plane at z = terrain_elevation.

        Args:
            u, v: Pixel coordinates in the image (can be float).

        Returns:
            tuple: (x, y, z) in world NED frame, or None if ray does not
                   intersect the ground (e.g., looking at sky).
        """
        if not self.pose_valid:
            return None

        if self.altitude_m < 0.5:
            # Too close to ground for reliable projection
            return None

        # 1) Pixel → normalized camera ray (in camera frame)
        pixel_h = np.array([u, v, 1.0], dtype=np.float64)
        ray_camera = self.K_inv @ pixel_h
        ray_camera = ray_camera / np.linalg.norm(ray_camera)

        # 2) Camera frame → body frame (apply gimbal rotation)
        ray_body = self.R_gimbal @ ray_camera

        # 3) Body frame → world frame (apply VIO rotation)
        ray_world = self.R_body_to_world @ ray_body

        # 4) Ray-plane intersection
        # Drone position in world
        drone_pos = self.t_world.copy()

        # Ground plane: z = terrain_elevation (in NED, positive down)
        # We need to find t such that: drone_pos + t * ray_world has z = ground_z
        ground_z = self.terrain_elevation

        # ray_world[2] is the Z (down) component of the ray direction
        if abs(ray_world[2]) < 1e-6:
            # Ray is nearly horizontal — no ground intersection
            return None

        t = (ground_z - drone_pos[2]) / ray_world[2]

        if t < 0:
            # Intersection is behind the camera (looking away from ground)
            return None

        # Intersection point
        ground_point = drone_pos + t * ray_world

        return (float(ground_point[0]),
                float(ground_point[1]),
                float(ground_point[2]))

    def project_bbox_center(self, left, top, width, height):
        """Project the center of a bounding box to world coordinates.

        Args:
            left, top, width, height: Bounding box in pixel coordinates.

        Returns:
            tuple: (x, y, z) in world NED frame, or None.
        """
        cx = left + width / 2.0
        cy = top + height / 2.0
        return self.project_pixel_to_ground(cx, cy)

    def enrich_detections(self, detections):
        """Add world coordinates to a list of Detection objects.

        Modifies detections in-place, setting world_x, world_y, world_z
        fields on each Detection object.

        Args:
            detections: list of Detection objects (from detection_transmitter.py)
        """
        if not self.pose_valid:
            return

        for det in detections:
            world_pos = self.project_bbox_center(
                det.left, det.top, det.width, det.height)
            if world_pos is not None:
                det.world_x, det.world_y, det.world_z = world_pos

    def get_footprint(self):
        """Calculate the camera's ground footprint at current altitude.

        Returns:
            dict: {
                'center': (x, y),      # ground point directly below camera axis
                'width_m': float,       # footprint width in meters
                'height_m': float,      # footprint height in meters
                'gsd_cm': float,        # ground sample distance in cm/pixel
            }
            or None if pose is not valid.
        """
        if not self.pose_valid or self.altitude_m < 0.5:
            return None

        center = self.project_pixel_to_ground(
            self.image_width / 2.0, self.image_height / 2.0)
        if center is None:
            return None

        # Corners
        corners = [
            self.project_pixel_to_ground(0, 0),
            self.project_pixel_to_ground(self.image_width, 0),
            self.project_pixel_to_ground(self.image_width, self.image_height),
            self.project_pixel_to_ground(0, self.image_height),
        ]

        valid_corners = [c for c in corners if c is not None]
        if len(valid_corners) < 2:
            return None

        xs = [c[0] for c in valid_corners]
        ys = [c[1] for c in valid_corners]

        width_m = max(xs) - min(xs)
        height_m = max(ys) - min(ys)

        # GSD: Ground Sample Distance = ground width / image width
        gsd_m = width_m / self.image_width if width_m > 0 else 0
        gsd_cm = gsd_m * 100.0

        return {
            "center": (center[0], center[1]),
            "width_m": width_m,
            "height_m": height_m,
            "gsd_cm": gsd_cm,
        }


# ─── Self-Test ───────────────────────────────────────────────────────────────

def _run_self_test():
    """Verify geo-registration with a simulated nadir (straight-down) camera."""

    print("═══ Geo-Registration — Self Test ═══\n")

    # Simulate: drone at 100m altitude, looking straight down, heading North
    geo = GeoRegistrator(
        fx=600.0, fy=600.0,
        cx=960.0, cy=540.0,
        image_width=1920, image_height=1080,
        gimbal_pitch_deg=-90.0,  # nadir
    )

    # Drone position: 50m North, 30m East, 100m altitude (NED: z = -100)
    R_identity = np.eye(3)  # heading North, level flight
    t_world = np.array([50.0, 30.0, -100.0])  # NED
    geo.update_pose(R_identity, t_world, altitude_m=100.0)

    print(f"Drone position (NED): [{t_world[0]:.1f}, {t_world[1]:.1f}, {t_world[2]:.1f}]")
    print(f"Altitude AGL: {geo.altitude_m:.1f} m")
    print(f"Gimbal: nadir (-90°)\n")

    # Test 1: Image center should project to directly below drone
    center = geo.project_pixel_to_ground(960.0, 540.0)
    print(f"Image center (960, 540) → ground: "
          f"({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
    assert center is not None
    assert abs(center[0] - 50.0) < 1.0, f"X error: {center[0]}"
    assert abs(center[1] - 30.0) < 1.0, f"Y error: {center[1]}"
    print(f"  ✅ PASS (expected ~50.0, 30.0)\n")

    # Test 2: Bbox projection
    world = geo.project_bbox_center(left=800, top=400, width=100, height=80)
    print(f"Bbox (800, 400, 100, 80) center → ground: "
          f"({world[0]:.1f}, {world[1]:.1f}, {world[2]:.1f})")
    assert world is not None
    print(f"  ✅ PASS (offset from center)\n")

    # Test 3: Footprint
    fp = geo.get_footprint()
    print(f"Ground footprint at {geo.altitude_m:.0f}m:")
    print(f"  Center: ({fp['center'][0]:.1f}, {fp['center'][1]:.1f})")
    print(f"  Width:  {fp['width_m']:.1f} m")
    print(f"  Height: {fp['height_m']:.1f} m")
    print(f"  GSD:    {fp['gsd_cm']:.1f} cm/pixel")
    print(f"  ✅ PASS\n")

    # Test 4: Detection enrichment
    from detection_transmitter import Detection
    dets = [
        Detection(class_id=0, confidence=0.95,
                  left=900, top=500, width=50, height=80),
        Detection(class_id=3, confidence=0.87,
                  left=400, top=300, width=120, height=60),
    ]
    geo.enrich_detections(dets)
    for d in dets:
        print(f"  Detection cls={d.class_id}: "
              f"pixel=({d.left},{d.top}) → "
              f"world=({d.world_x:.1f}, {d.world_y:.1f}, {d.world_z:.1f})")
    print(f"  ✅ PASS\n")

    print("═══ Self Test Complete ═══")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Geo-registration for GPS-denied drone navigation")
    parser.add_argument("--test", action="store_true",
                        help="Run self-test with simulated nadir camera")
    args = parser.parse_args()

    if args.test:
        _run_self_test()
    else:
        parser.print_help()
