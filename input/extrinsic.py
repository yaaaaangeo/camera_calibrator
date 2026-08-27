"""
input/extrinsic.py

Existing-calibration (T_CL / T_LC) loader + sanity check, per the Input
Loader Spec (v0.1) in evaluation_metric_spec.md.

Two distinct responsibilities kept separate, matching the spec:
  1. load_extrinsic(...): parse whatever format the user provides
     (rpy/quaternion/matrix, any parent/child direction, any unit) and
     normalize it into a single 4x4 T_CL matrix.
  2. verify_extrinsic(...): check the *loaded* result for mathematical
     validity (rotation validity, finiteness, plausible unit/magnitude).
     This is intentionally NOT re-deriving or judging calibration quality
     (that's the Evaluation Engine's job) -- only "is this a well-formed
     transform at all".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

import numpy as np

from geometry.transform import (
    rpy_to_rotation_matrix,
    quaternion_to_rotation_matrix,
    is_valid_rotation_matrix,
    to_homogeneous,
    invert_transform,
)


UNIT_TO_METERS = {"m": 1.0, "cm": 0.01, "mm": 0.001}

RotationInput = Union[
    tuple[float, float, float],
    tuple[float, float, float, float],
    list,
    np.ndarray,
]


@dataclass
class ExtrinsicRaw:
    parent: Literal["lidar", "camera"]
    child: Literal["lidar", "camera"]
    translation: tuple[float, float, float]
    rotation: RotationInput
    rotation_format: Literal["rpy_deg", "rpy_rad", "quaternion", "matrix3x3", "matrix4x4"]
    unit: Literal["m", "cm", "mm"] = "m"


@dataclass
class ExtrinsicModel:
    T_CL: np.ndarray  # 4x4 homogeneous, camera_from_lidar: p_cam = T_CL @ p_lidar
    parent: str
    child: str
    raw: ExtrinsicRaw


def _rotation_to_matrix(rotation: RotationInput, fmt: str) -> np.ndarray:
    if fmt == "rpy_deg":
        roll, pitch, yaw = rotation
        return rpy_to_rotation_matrix(roll, pitch, yaw, degrees=True)
    if fmt == "rpy_rad":
        roll, pitch, yaw = rotation
        return rpy_to_rotation_matrix(roll, pitch, yaw, degrees=False)
    if fmt == "quaternion":
        x, y, z, w = rotation
        return quaternion_to_rotation_matrix(x, y, z, w)
    if fmt == "matrix3x3":
        R = np.asarray(rotation, dtype=float)
        if R.shape != (3, 3):
            raise ValueError(f"rotation_format='matrix3x3' but got shape {R.shape}")
        return R
    if fmt == "matrix4x4":
        M = np.asarray(rotation, dtype=float)
        if M.shape != (4, 4):
            raise ValueError(f"rotation_format='matrix4x4' but got shape {M.shape}")
        return M[:3, :3]
    raise ValueError(f"Unknown rotation_format: {fmt!r}")


def load_extrinsic(raw: ExtrinsicRaw) -> ExtrinsicModel:
    """
    Normalize an ExtrinsicRaw (any supported rotation format, any unit, and
    EITHER parent/child direction) into a canonical 4x4 T_CL matrix such
    that p_cam = T_CL @ p_lidar.

    This is the single choke point defending against the "T_CL vs T_LC
    swapped" mistake called out in the original design notes: whichever
    direction the user's data is in, this function inverts as needed so
    every downstream consumer can assume T_CL means the same thing.
    """
    unit_scale = UNIT_TO_METERS.get(raw.unit)
    if unit_scale is None:
        raise ValueError(f"Unknown unit: {raw.unit!r}")

    if raw.rotation_format == "matrix4x4":
        M = np.asarray(raw.rotation, dtype=float)
        if M.shape != (4, 4):
            raise ValueError(f"rotation_format='matrix4x4' but got shape {M.shape}")
        R = M[:3, :3]
        # translation comes from the matrix itself in this case, still unit-scaled
        t = M[:3, 3] * unit_scale
    else:
        R = _rotation_to_matrix(raw.rotation, raw.rotation_format)
        t = np.asarray(raw.translation, dtype=float) * unit_scale

    T = to_homogeneous(R, t)

    if raw.parent == "lidar" and raw.child == "camera":
        # user's transform already goes lidar -> camera i.e. IS T_CL
        T_CL = T
    elif raw.parent == "camera" and raw.child == "lidar":
        # user's transform goes camera -> lidar i.e. is T_LC; invert to get T_CL
        T_CL = invert_transform(T)
    else:
        raise ValueError(
            f"parent/child must be one of ('lidar','camera')/('camera','lidar'), "
            f"got parent={raw.parent!r}, child={raw.child!r}"
        )

    return ExtrinsicModel(T_CL=T_CL, parent=raw.parent, child=raw.child, raw=raw)


@dataclass
class SanityCheckItem:
    name: str
    passed: bool
    detail: str


@dataclass
class SanityReport:
    items: list[SanityCheckItem] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(item.passed for item in self.items)

    def failed_items(self) -> list[SanityCheckItem]:
        return [i for i in self.items if not i.passed]


def verify_extrinsic(
    model: ExtrinsicModel,
    rotation_tol: float = 1e-3,
    max_plausible_translation_m: float = 100.0,
) -> SanityReport:
    """
    Structural / mathematical sanity check on a loaded ExtrinsicModel.
    Does NOT judge calibration *quality* -- only whether T_CL is well-formed.

    Checks:
      - rotation validity (det(R) ~= 1, orthogonality)
      - translation finiteness
      - translation magnitude plausibility (catches unit mistakes, e.g. a
        cm/mm value accidentally treated as meters would blow past any
        sane camera-lidar rig baseline)
    """
    items: list[SanityCheckItem] = []
    T = model.T_CL
    R = T[:3, :3]
    t = T[:3, 3]

    rot_valid, rot_diag = is_valid_rotation_matrix(R, tol=rotation_tol)
    items.append(SanityCheckItem(
        name="rotation_valid",
        passed=rot_valid,
        detail=f"det={rot_diag['determinant']:.6f}, "
               f"orthogonality_error={rot_diag['orthogonality_error']:.2e}",
    ))

    t_finite = bool(np.all(np.isfinite(t)))
    items.append(SanityCheckItem(
        name="translation_finite",
        passed=t_finite,
        detail=f"translation={t.tolist()}",
    ))

    if t_finite:
        norm = float(np.linalg.norm(t))
        plausible = 0.0 <= norm <= max_plausible_translation_m
        items.append(SanityCheckItem(
            name="translation_magnitude_plausible",
            passed=plausible,
            detail=f"||t||={norm:.4f} m (threshold {max_plausible_translation_m} m); "
                   f"{'' if plausible else 'check units -- possible mm/cm-as-m mistake'}",
        ))
    else:
        items.append(SanityCheckItem(
            name="translation_magnitude_plausible",
            passed=False,
            detail="skipped: translation not finite",
        ))

    homogeneous_row_ok = bool(np.allclose(T[3, :], [0, 0, 0, 1], atol=1e-9))
    items.append(SanityCheckItem(
        name="homogeneous_bottom_row_valid",
        passed=homogeneous_row_ok,
        detail=f"T[3,:]={T[3, :].tolist()}",
    ))

    return SanityReport(items=items)
