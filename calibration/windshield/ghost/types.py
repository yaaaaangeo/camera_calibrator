"""
calibration.windshield.ghost.types
==============================================================

STEP 8 - Ghost / Double Image 결과 타입.

Reflection(calibration.windshield.reflection)과 완전히 독립된 타입 집합이다
(사용자 스펙 1/22번) - `ReflectionEvaluationResult`와 절대 합치지 않는다.
Ghost는 "다른 장면의 반사"가 아니라 "같은 exterior scene의 shifted/warped
duplicate"라는 서로 다른 image formation model을 갖기 때문이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from calibration.windshield.ghost.config import (
    DEFAULT_BRIGHT_SOURCE_THRESHOLD,
    DEFAULT_EDGE_MAX_SEARCH_RADIUS_PX,
    DEFAULT_EDGE_MIN_GRADIENT,
    DEFAULT_EDGE_MIN_SECONDARY_RATIO,
    DEFAULT_GAUSSIAN_SIGMA,
    DEFAULT_LIKELIHOOD_MAX_SEARCH_RADIUS_PX,
    DEFAULT_LIKELIHOOD_MIN_GRADIENT,
    DEFAULT_MAX_SEARCH_RADIUS_PX,
    DEFAULT_MIN_BLOB_AREA_PX,
    DEFAULT_MIN_PEAK_DISTANCE_PX,
    DEFAULT_SPATIAL_COLS,
    DEFAULT_SPATIAL_ROWS,
    GHOST_METRIC_VERSION,
    GHOST_MODEL_VERSION,
)


@dataclass
class GhostEvaluationConfig:
    mode: str = "point_source"  # "point_source" | "edge_target" | "general_likelihood"
    bright_source_threshold: float = DEFAULT_BRIGHT_SOURCE_THRESHOLD
    min_blob_area_px: int = DEFAULT_MIN_BLOB_AREA_PX
    max_search_radius_px: float = DEFAULT_MAX_SEARCH_RADIUS_PX
    gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA
    min_peak_distance_px: float = DEFAULT_MIN_PEAK_DISTANCE_PX
    edge_min_gradient: float = DEFAULT_EDGE_MIN_GRADIENT
    edge_max_search_radius_px: float = DEFAULT_EDGE_MAX_SEARCH_RADIUS_PX
    edge_min_secondary_ratio: float = DEFAULT_EDGE_MIN_SECONDARY_RATIO
    edge_axis: str = "vertical"  # 대상 edge가 수직선이면 "vertical"(오프셋은 x축), 수평선이면 "horizontal"
    likelihood_min_gradient: float = DEFAULT_LIKELIHOOD_MIN_GRADIENT
    likelihood_max_search_radius_px: float = DEFAULT_LIKELIHOOD_MAX_SEARCH_RADIUS_PX
    spatial_rows: int = DEFAULT_SPATIAL_ROWS
    spatial_cols: int = DEFAULT_SPATIAL_COLS


@dataclass
class GhostPointDetection:
    main_x: float
    main_y: float
    ghost_x: Optional[float] = None
    ghost_y: Optional[float] = None
    offset_x_px: Optional[float] = None
    offset_y_px: Optional[float] = None
    distance_px: Optional[float] = None
    angular_separation_deg: Optional[float] = None
    strength_ratio: Optional[float] = None
    detected: bool = False


@dataclass
class GhostRegionMetrics:
    mean_distance_px: float = 0.0
    mean_strength_ratio: float = 0.0
    detection_rate: float = 0.0


@dataclass
class GhostSpatialCell:
    row: int
    col: int
    mean_offset_x_px: Optional[float] = None
    mean_offset_y_px: Optional[float] = None
    mean_distance_px: Optional[float] = None
    mean_strength_ratio: Optional[float] = None
    sample_count: int = 0


@dataclass
class GhostEvaluationResult:
    success: bool
    mode: str
    metric_version: int = GHOST_METRIC_VERSION
    pair_id: str = ""

    detection_count: int = 0
    candidate_count: int = 0
    detection_rate: Optional[float] = None

    mean_offset_x_px: Optional[float] = None
    mean_offset_y_px: Optional[float] = None
    median_distance_px: Optional[float] = None
    p95_distance_px: Optional[float] = None

    mean_angular_separation_deg: Optional[float] = None
    p95_angular_separation_deg: Optional[float] = None

    mean_strength_ratio: Optional[float] = None
    p95_strength_ratio: Optional[float] = None

    # General(No-Reference) mode 전용(사용자 스펙 24-26번) - heuristic
    # likelihood일 뿐 ground truth가 아니다. Point-source/edge 모드에서는
    # 항상 None/False로 남는다.
    ghost_likelihood: Optional[float] = None
    is_likelihood: bool = False

    regional_metrics: dict[str, GhostRegionMetrics] = field(default_factory=dict)
    spatial_map: list[GhostSpatialCell] = field(default_factory=list)
    detections: list[GhostPointDetection] = field(default_factory=list)

    warning_message: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class GhostDatasetResult:
    mode: str
    metric_version: int = GHOST_METRIC_VERSION
    per_frame: list[GhostEvaluationResult] = field(default_factory=list)
    mean_distance_px: Optional[float] = None
    p95_distance_px: Optional[float] = None
    mean_strength: Optional[float] = None
    p95_strength: Optional[float] = None
    worst_frame_id: Optional[str] = None
    success: bool = True
    warning_message: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class GhostField:
    """Grid basis의 spatially varying displacement/strength field(사용자
    스펙 40번). `offset_x`/`offset_y`/`strength` 모두 (rows, cols) 크기의
    dense하지 않은 성긴 grid고, runtime에서 이미지 해상도로 resize해서
    쓴다."""
    offset_x: np.ndarray
    offset_y: np.ndarray
    strength: np.ndarray
    image_width: float
    image_height: float
    model_version: int = GHOST_MODEL_VERSION


@dataclass
class GhostSuppressionResult:
    success: bool

    suppressed_image: Optional[np.ndarray] = None
    predicted_ghost_image: Optional[np.ndarray] = None
    correction_map: Optional[np.ndarray] = None

    iterations: int = 0
    mean_correction: float = 0.0
    max_correction: float = 0.0

    fell_back_to_original: bool = False
    warning_message: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class GhostSuppressionEvaluation:
    """STEP 8A evaluator를 suppression 전/후에 동일하게 적용한 결과(사용자
    스펙 46번)."""
    before: GhostEvaluationResult
    after: GhostEvaluationResult

    strength_reduction: Optional[float] = None
    detection_reduction: Optional[float] = None

    over_suppression_score: Optional[float] = None

    success: bool = True
    warning_message: Optional[str] = None
    error_message: Optional[str] = None
