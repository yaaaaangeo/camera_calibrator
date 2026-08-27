"""
Sensor-relative noise floor calculation for GT-free Cam-LiDAR calibration
evaluation.

Implements floor(Z) as specified in evaluation_metric_spec.md (v0.3):

    floor(Z) = sqrt( floor_angular^2 + floor_range(Z)^2 + floor_edge^2 )

    floor_angular       = fx * theta_res
    floor_range(Z)      = fx * b * sigma_r / Z^2
    floor_edge          = constant (edge-detector sub-pixel floor)

All three terms are treated as independent error sources and combined via
quadrature (root-sum-of-squares).

This module is intentionally dependency-light (numpy only) and side-effect
free: every function is pure and returns explicit warning flags rather than
logging, so callers (report generation, CLI, etc.) can decide how to surface
fallback usage to the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math

import numpy as np


# ---------------------------------------------------------------------------
# Default / fallback constants
#
# Kept in one place (per "Input Loader 결정 필요 사항 #2": fallback 값을
# constants 로 분리) so they're easy to tune once reference-dataset
# validation (spec section "Reference dataset으로 anchor 검증") is done.
# ---------------------------------------------------------------------------

DEFAULT_RANGE_ACCURACY_M: float = 0.02          # sigma_r fallback, 2 cm (1-sigma)
DEFAULT_ANGULAR_RESOLUTION_DEG: float = 0.2     # theta_res fallback, industry-average-ish
DEFAULT_EDGE_LOCALIZATION_FLOOR_PX: float = 0.5 # floor_edge fallback (sub-pixel edge detector)

# Multiplier-based thresholds (spec: M2 uses 2x/5x, M3/M4 STD uses 1x/3x)
M2_GOOD_MULTIPLIER: float = 2.0
M2_WARNING_MULTIPLIER: float = 5.0
STD_GOOD_MULTIPLIER: float = 1.0
STD_WARNING_MULTIPLIER: float = 3.0


@dataclass
class LidarSensorSpecForFloor:
    """
    Minimal subset of LidarSensorSpec (see input loader spec) needed for
    floor(Z). Mirrors the field names/optionality of the real loader
    dataclass so this can later be constructed directly from LidarModel
    without translation.
    """
    horizontal_resolution_deg: Optional[float] = None
    vertical_resolution_deg: Optional[float] = None
    channels: Optional[int] = None
    vertical_fov_deg: Optional[float] = None
    range_accuracy_m: Optional[float] = None


@dataclass
class FloorInputs:
    """Fully-resolved inputs to floor(Z), after fallback resolution."""
    fx_px: float
    theta_res_rad: float
    baseline_m: float
    sigma_r_m: float
    floor_edge_px: float

    # provenance / warnings
    used_angular_fallback: bool = False
    used_range_accuracy_fallback: bool = False
    used_edge_floor_fallback: bool = False
    fallback_warnings: list[str] = field(default_factory=list)


def _resolve_angular_resolution_deg(spec: LidarSensorSpecForFloor) -> tuple[float, bool, Optional[str]]:
    """
    Resolve theta_res (degrees) as the WORST CASE (largest) resolution across
    horizontal and vertical axes, per the spec ("worst case(더 큰 값) 사용").

    Each axis is resolved independently:
      - horizontal: use horizontal_resolution_deg if given, else this axis
        contributes nothing (we don't have a horizontal channel-count fallback)
      - vertical: use vertical_resolution_deg if given; else approximate from
        channels + vertical_fov_deg if both are available

    If neither axis can be resolved at all, fall back to
    DEFAULT_ANGULAR_RESOLUTION_DEG with a strong warning.

    Note: the vertical approximation is applied whenever vertical_resolution_deg
    is missing, REGARDLESS of whether horizontal_resolution_deg was given -
    otherwise a provided horizontal spec would silently mask a much coarser
    (and unresolved) vertical resolution.
    """
    candidates: list[float] = []
    used_fallback = False
    warning: Optional[str] = None

    if spec.horizontal_resolution_deg is not None:
        candidates.append(spec.horizontal_resolution_deg)

    if spec.vertical_resolution_deg is not None:
        candidates.append(spec.vertical_resolution_deg)
    elif spec.channels is not None and spec.vertical_fov_deg is not None and spec.channels > 0:
        approx_vertical_res = spec.vertical_fov_deg / spec.channels
        candidates.append(approx_vertical_res)
        used_fallback = True
        warning = (
            f"vertical_resolution_deg not provided; approximated as "
            f"vertical_fov_deg/channels = {approx_vertical_res:.4f} deg"
        )

    if candidates:
        return max(candidates), used_fallback, warning

    warning = (
        f"No angular resolution info provided (horizontal/vertical_resolution_deg, "
        f"or channels+vertical_fov_deg); using default "
        f"{DEFAULT_ANGULAR_RESOLUTION_DEG} deg. Thresholds derived from floor(Z) "
        f"are unreliable until real sensor specs are supplied."
    )
    return DEFAULT_ANGULAR_RESOLUTION_DEG, True, warning


def _resolve_range_accuracy_m(spec: LidarSensorSpecForFloor) -> tuple[float, bool, Optional[str]]:
    if spec.range_accuracy_m is not None:
        return spec.range_accuracy_m, False, None
    warning = (
        f"range_accuracy_m (sigma_r) not provided; using default "
        f"{DEFAULT_RANGE_ACCURACY_M} m."
    )
    return DEFAULT_RANGE_ACCURACY_M, True, warning


def resolve_floor_inputs(
    fx_px: float,
    T_CL: np.ndarray,
    lidar_spec: LidarSensorSpecForFloor,
    edge_localization_floor_px: Optional[float] = None,
) -> FloorInputs:
    """
    Resolve all inputs required by floor(Z), applying fallback rules and
    recording which fallbacks were used.

    Parameters
    ----------
    fx_px:
        Camera focal length in pixels (from CameraIntrinsics.fx).
    T_CL:
        4x4 (or 3x4) homogeneous extrinsic transform, camera_from_lidar,
        as produced by the extrinsic loader. Only the translation block
        is used here (baseline b = ||t||).
    lidar_spec:
        Sensor spec fields relevant to floor(Z) (subset of LidarSensorSpec).
    edge_localization_floor_px:
        Optional override for floor_edge (Term 3). If None, uses
        DEFAULT_EDGE_LOCALIZATION_FLOOR_PX.
    """
    if fx_px <= 0:
        raise ValueError(f"fx_px must be positive, got {fx_px}")

    T_CL = np.asarray(T_CL, dtype=float)
    if T_CL.shape not in [(4, 4), (3, 4)]:
        raise ValueError(f"T_CL must be 4x4 or 3x4, got shape {T_CL.shape}")
    translation = T_CL[:3, 3]
    baseline_m = float(np.linalg.norm(translation))

    theta_res_deg, used_angular_fallback, angular_warning = _resolve_angular_resolution_deg(lidar_spec)
    sigma_r_m, used_range_fallback, range_warning = _resolve_range_accuracy_m(lidar_spec)

    if edge_localization_floor_px is not None:
        floor_edge_px = edge_localization_floor_px
        used_edge_fallback = False
    else:
        floor_edge_px = DEFAULT_EDGE_LOCALIZATION_FLOOR_PX
        used_edge_fallback = True

    warnings: list[str] = [w for w in (angular_warning, range_warning) if w is not None]
    if used_edge_fallback:
        warnings.append(
            f"edge_localization_floor_px not provided; using default "
            f"{DEFAULT_EDGE_LOCALIZATION_FLOOR_PX} px."
        )

    return FloorInputs(
        fx_px=fx_px,
        theta_res_rad=math.radians(theta_res_deg),
        baseline_m=baseline_m,
        sigma_r_m=sigma_r_m,
        floor_edge_px=floor_edge_px,
        used_angular_fallback=used_angular_fallback,
        used_range_accuracy_fallback=used_range_fallback,
        used_edge_floor_fallback=used_edge_fallback,
        fallback_warnings=warnings,
    )


def floor_angular(inputs: FloorInputs) -> float:
    """Term 1: LiDAR angular resolution contribution (px), distance-independent."""
    return inputs.fx_px * inputs.theta_res_rad


def floor_range(inputs: FloorInputs, z_m: float) -> float:
    """Term 2: LiDAR range-noise contribution (px) at distance z_m. Falls as 1/Z^2."""
    if z_m <= 0:
        raise ValueError(f"z_m must be positive, got {z_m}")
    return inputs.fx_px * inputs.baseline_m * inputs.sigma_r_m / (z_m ** 2)


def floor_edge(inputs: FloorInputs) -> float:
    """Term 3: image-side edge localization floor (px), constant."""
    return inputs.floor_edge_px


def compute_floor(inputs: FloorInputs, z_m: float) -> float:
    """
    floor(Z) = sqrt(floor_angular^2 + floor_range(Z)^2 + floor_edge^2)

    z_m should be the representative distance for the point set being
    evaluated (spec recommends median depth of the LiDAR edge points used
    by M2/M3/M4).
    """
    t1 = floor_angular(inputs)
    t2 = floor_range(inputs, z_m)
    t3 = floor_edge(inputs)
    return math.sqrt(t1 ** 2 + t2 ** 2 + t3 ** 2)


@dataclass
class FloorBreakdown:
    """Full breakdown of a floor(Z) computation, useful for diagnostics/report."""
    z_m: float
    floor_px: float
    term_angular_px: float
    term_range_px: float
    term_edge_px: float
    dominant_term: str
    inputs: FloorInputs


def compute_floor_breakdown(inputs: FloorInputs, z_m: float) -> FloorBreakdown:
    """Same as compute_floor but returns the per-term breakdown and flags
    which term dominates the total (useful for explaining why floor is
    large/small in a given rig/scene)."""
    t1 = floor_angular(inputs)
    t2 = floor_range(inputs, z_m)
    t3 = floor_edge(inputs)
    total = math.sqrt(t1 ** 2 + t2 ** 2 + t3 ** 2)

    terms = {"angular": t1, "range": t2, "edge": t3}
    dominant = max(terms, key=terms.get)

    return FloorBreakdown(
        z_m=z_m,
        floor_px=total,
        term_angular_px=t1,
        term_range_px=t2,
        term_edge_px=t3,
        dominant_term=dominant,
        inputs=inputs,
    )


def multiplier_thresholds(floor_px: float, good_mult: float, warning_mult: float) -> dict:
    """
    Convert a floor(Z) value into concrete GOOD/WARNING/BAD boundaries in px,
    given the multiplier scheme for a specific metric (M2 uses 2x/5x,
    M3/M4 STD uses 1x/3x per the spec).
    """
    return {
        "good_below_px": good_mult * floor_px,
        "warning_below_px": warning_mult * floor_px,
        # anything >= warning_below_px is BAD
    }


def classify(value_px: float, floor_px: float, good_mult: float, warning_mult: float) -> str:
    """Classify a measured px value against floor-derived thresholds."""
    bounds = multiplier_thresholds(floor_px, good_mult, warning_mult)
    if value_px < bounds["good_below_px"]:
        return "GOOD"
    if value_px < bounds["warning_below_px"]:
        return "WARNING"
    return "BAD"
