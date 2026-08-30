"""
camera_calibrator.camera_lidar.targetless_prior
===================================================

Loads a `direct_visual_lidar_calibration` `calib.json` result and converts
it into this project's Common Data Model TargetlessPrior -- a coarse
Camera->LiDAR extrinsic used ONLY to seed camera_lidar.guided_roi's LiDAR
search region (see camera_lidar/types.py's TargetlessPrior docstring).

No ROS, PCL, GTSAM, or Ceres imports here -- this module only parses a JSON
file and builds a 4x4 transform with geometry/transform.py's existing
helpers.

Convention: `direct_visual_lidar_calibration` stores its result vectors as
[tx, ty, tz, qx, qy, qz, qw] and its `T_lidar_camera` key is ALREADY the
Camera->LiDAR direction this project needs (p_lidar = T_lidar_camera @
p_camera) -- so, unlike most other transform ingestion in this codebase,
no invert_transform() call happens here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from geometry.transform import is_valid_rotation_matrix, quaternion_to_rotation_matrix, to_homogeneous
from camera_lidar.types import TargetlessPrior

_KEY_BY_SOURCE = {
    "final": "T_lidar_camera",
    "auto_initial": "init_T_lidar_camera_auto",
    "manual_initial": "init_T_lidar_camera",
}
_AUTO_SELECT_ORDER = ("T_lidar_camera", "init_T_lidar_camera_auto", "init_T_lidar_camera")
_VALID_SOURCES = frozenset({"auto", "final", "auto_initial", "manual_initial"})


def load_direct_visual_calib(path: "str | Path", source: str = "auto") -> TargetlessPrior:
    """Parse a direct_visual_lidar_calibration calib.json into a
    TargetlessPrior.

    source: "auto" tries T_lidar_camera, then init_T_lidar_camera_auto,
    then init_T_lidar_camera, in that priority order, and uses the FIRST
    ONE THAT IS ACTUALLY VALID -- not merely present. A higher-priority key
    that exists but fails validation (corrupt/malformed) is skipped in
    favor of the next candidate, rather than aborting the whole load; if
    NONE of the three candidates validate, raises ValueError listing why
    each one failed.

    "final" / "auto_initial" / "manual_initial" force one specific key --
    unlike "auto", these NEVER fall back to a different key: an invalid or
    missing value for the explicitly-requested key raises immediately."""
    if source not in _VALID_SOURCES:
        raise ValueError(f"Unknown prior source {source!r}; expected one of {sorted(_VALID_SOURCES)}")

    path = Path(path)
    try:
        with open(path, "r") as f:
            root = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Could not read/parse calib.json at {path}: {e}") from e

    if not isinstance(root, dict):
        raise ValueError(f"calib.json root must be a JSON object, got {type(root).__name__}")

    results = root.get("results")
    if not isinstance(results, dict):
        raise ValueError("calib.json has no 'results' object")

    if source == "auto":
        failure_reasons: dict[str, str] = {}
        for key in _AUTO_SELECT_ORDER:
            if key not in results:
                failure_reasons[key] = "missing"
                continue
            try:
                T = _parse_transform_values(results[key], key)
            except ValueError as e:
                failure_reasons[key] = str(e)
                continue
            return TargetlessPrior(T_lidar_from_camera=T, source_path=str(path), source_key=key)

        detail = "\n\n".join(f"{key}:\n{reason}" for key, reason in failure_reasons.items())
        raise ValueError(f"No valid direct_visual calibration transform found.\n\n{detail}")

    actual_key = _KEY_BY_SOURCE[source]
    if actual_key not in results:
        raise ValueError(f"calib.json['results'] is missing key {actual_key!r} (source={source!r})")

    values = results[actual_key]
    T = _parse_transform_values(values, actual_key)

    return TargetlessPrior(T_lidar_from_camera=T, source_path=str(path), source_key=actual_key)


def _parse_transform_values(values, key: str) -> np.ndarray:
    if not isinstance(values, (list, tuple)) or len(values) != 7:
        raise ValueError(f"calib.json['results'][{key!r}] must be a length-7 array, got {values!r}")

    try:
        numeric = [float(v) for v in values]
    except (TypeError, ValueError) as e:
        raise ValueError(f"calib.json['results'][{key!r}] contains a non-numeric value: {e}") from e

    if not all(math.isfinite(v) for v in numeric):
        raise ValueError(f"calib.json['results'][{key!r}] contains a non-finite value: {numeric}")

    tx, ty, tz, qx, qy, qz, qw = numeric
    quat_norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if quat_norm < 1e-12:
        raise ValueError(f"calib.json['results'][{key!r}] has a (near-)zero-norm quaternion")

    try:
        R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    except ValueError as e:
        raise ValueError(f"calib.json['results'][{key!r}]: invalid quaternion: {e}") from e

    is_valid, diagnostics = is_valid_rotation_matrix(R)
    if not is_valid:
        raise ValueError(
            f"calib.json['results'][{key!r}] quaternion did not produce a valid rotation "
            f"matrix: {diagnostics}"
        )

    t = np.array([tx, ty, tz], dtype=float)
    T = to_homogeneous(R, t)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError(f"calib.json['results'][{key!r}] produced an invalid 4x4 transform")

    return T
