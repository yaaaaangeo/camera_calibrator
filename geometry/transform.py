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
