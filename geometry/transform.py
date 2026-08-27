"""
SE(3) rigid transform utilities.

Convention:
- Rotation matrices are 3x3, applied as R @ p (column vectors).
- RPY (roll, pitch, yaw) uses the common robotics intrinsic ZYX convention:
      R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
  i.e. roll about X, then pitch about Y, then yaw about Z, applied in that
  order to a vector in the rotated frame (standard REP-103 / ROS convention).
- Homogeneous transforms T are 4x4:
      T = [[R, t],
           [0, 1]]
  and applied to points as p_out = T @ [p; 1].
- T_CL denotes "transform that takes points expressed in the LiDAR frame and
  expresses them in the Camera frame": p_cam = T_CL @ p_lidar (in homogeneous
  form). This matches the parent/child convention in the input spec where
  parent=lidar, child=camera is stored as T_CL (child_from_parent... see
  input/extrinsic.py for the explicit parent/child bookkeeping).
"""

from __future__ import annotations

import numpy as np


def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float, degrees: bool = False) -> np.ndarray:
    """Build a 3x3 rotation matrix from roll/pitch/yaw using intrinsic ZYX
    (R = Rz(yaw) @ Ry(pitch) @ Rx(roll))."""
    if degrees:
        roll, pitch, yaw = np.radians([roll, pitch, yaw])

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]])
    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0],
                   [sy, cy, 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from a quaternion (x, y, z, w order).
    Quaternion is normalized internally; raises on (near-)zero norm."""
    q = np.array([x, y, z, w], dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Quaternion has (near-)zero norm; cannot normalize.")
    x, y, z, w = q / norm

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def is_valid_rotation_matrix(R: np.ndarray, tol: float = 1e-3) -> tuple[bool, dict]:
    """
    Check whether R is a valid rotation matrix:
      - det(R) ~= 1 (not -1, which would indicate a reflection)
      - R @ R.T ~= I (orthogonality)

    Returns (is_valid, diagnostics_dict) so callers can report *why* it
    failed rather than just a boolean.
    """
    R = np.asarray(R, dtype=float)
    diagnostics = {}

    det = np.linalg.det(R)
    diagnostics["determinant"] = float(det)
    det_ok = abs(det - 1.0) < tol

    orthogonality_error = float(np.max(np.abs(R @ R.T - np.eye(3))))
    diagnostics["orthogonality_error"] = orthogonality_error
    orthogonal_ok = orthogonality_error < tol

    diagnostics["det_ok"] = det_ok
    diagnostics["orthogonal_ok"] = orthogonal_ok

    return (det_ok and orthogonal_ok), diagnostics


def rotation_geodesic_distance(R_a: np.ndarray, R_b: np.ndarray, degrees: bool = False) -> float:
    """SO(3) geodesic angular distance between two rotation matrices --
    the angle of the relative rotation R_a^T @ R_b, via the standard
    trace formula: theta = arccos((trace(R_a^T @ R_b) - 1) / 2). Unlike a
    per-axis Euler/RPY difference, this is a single, convention-independent
    "how far apart are these two orientations" number."""
    R_a = np.asarray(R_a, dtype=float)
    R_b = np.asarray(R_b, dtype=float)
    relative = R_a.T @ R_b
    cos_theta = (np.trace(relative) - 1.0) / 2.0
    theta = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    return np.degrees(theta) if degrees else theta


def to_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compose a 3x3 rotation and 3-vector translation into a 4x4 homogeneous transform."""
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 homogeneous SE(3) transform analytically (faster/more
    stable than np.linalg.inv for this structure)."""
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def compose_transforms(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """Compose two 4x4 transforms: applying the result to a point p gives
    T_a @ (T_b @ p), i.e. T_b is applied first."""
    return np.asarray(T_a, dtype=float) @ np.asarray(T_b, dtype=float)


def rotation_matrix_to_rpy(R: np.ndarray, degrees: bool = False) -> tuple[float, float, float]:
    """Inverse of rpy_to_rotation_matrix: extract (roll, pitch, yaw) from a
    3x3 rotation matrix built as R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    R = np.asarray(R, dtype=float)
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    if degrees:
        roll, pitch, yaw = np.degrees([roll, pitch, yaw])
    return float(roll), float(pitch), float(yaw)


def rotation_matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 rotation matrix -> quaternion (x, y, z, w), via Shepperd's method
    (numerically stable across all rotation angles, unlike the naive
    trace-only formula near a 180 degree rotation)."""
    R = np.asarray(R, dtype=float)
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    q = q / np.linalg.norm(q)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 (or 3x4) homogeneous transform to an (N, 3) array of points.
    Returns an (N, 3) array.
    """
    T = np.asarray(T, dtype=float)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got shape {points.shape}")
    if T.shape not in [(4, 4), (3, 4)]:
        raise ValueError(f"T must be 4x4 or 3x4, got shape {T.shape}")

    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t
