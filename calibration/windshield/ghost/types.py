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
    DEFAULT_MIN_CONSENSUS_CANDIDATES,
    DEFAULT_MIN_PEAK_DISTANCE_PX,
    DEFAULT_PAIRING_CONSENSUS_RADIUS_PX,
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
    edge_axis: str = "vertical"  # "vertical" | "horizontal" | "auto" - 대상 edge가 수직선이면
                                  # "vertical"(오프셋은 x축), 수평선이면 "horizontal", 방향을 모르면 "auto"
    likelihood_min_gradient: float = DEFAULT_LIKELIHOOD_MIN_GRADIENT
    likelihood_max_search_radius_px: float = DEFAULT_LIKELIHOOD_MAX_SEARCH_RADIUS_PX
    spatial_rows: int = DEFAULT_SPATIAL_ROWS
    spatial_cols: int = DEFAULT_SPATIAL_COLS
    pairing_consensus_radius_px: float = DEFAULT_PAIRING_CONSENSUS_RADIUS_PX
    min_consensus_candidates: int = DEFAULT_MIN_CONSENSUS_CANDIDATES


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
    # Global displacement consensus pairing(STEP 8 stabilization 1-C번) -
    # 이 pair의 (dx,dy)가 dataset 전체의 dominant ghost displacement vector와
    # 얼마나 떨어져 있는지(px). consensus를 아예 쓰지 못한 fallback pairing
    # (candidate가 너무 적을 때)에서는 None으로 남는다.
    pair_residual_px: Optional[float] = None


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
    # Multi-frame robust aggregation(STEP 8 stabilization 3-D번) - median 기반
    # 통계이므로 이름은 "mean_*"이지만 실제로는 robust median이다(기존
    # 필드명/의미를 유지하기 위해 이름은 바꾸지 않았다). MAD와 outlier로
    # 제외된 개수를 함께 기록해 fit stability를 진단할 수 있게 한다.
    mad_offset_x_px: Optional[float] = None
    mad_offset_y_px: Optional[float] = None
    mad_strength: Optional[float] = None
    outlier_rejected_count: int = 0
    # 이 cell에 직접 관측값이 없어 인접 cell 보간 또는 global median으로
    # 채워진 경우 True(STEP 8 stabilization 3-F번) - 실제 관측인지 fallback
    # 채움인지 UI/진단에서 구분할 수 있게 한다.
    is_filled: bool = False


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

    # Edge-target mode 전용(STEP 8 stabilization 2-F번) - Point Source의
    # dx/dy(2D 벡터)와 Edge의 scalar offset(1D, edge normal 방향)을 절대
    # 같은 필드로 재사용하지 않는다. Point Source/General 모드에서는 항상
    # None으로 남는다.
    edge_offset_median_px: Optional[float] = None
    edge_offset_p95_px: Optional[float] = None

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
class GhostFieldDiagnostics:
    """Dataset 전체에서 `GhostField`를 fit한 과정에 대한 진단 정보(STEP 8
    stabilization 3-H번). `GhostField` 자체를 과도하게 키우지 않기 위해
    별도 타입으로 분리했다."""
    num_frames: int = 0
    num_detections: int = 0
    grid_rows: int = 0
    grid_cols: int = 0
    samples_per_cell: list[int] = field(default_factory=list)  # row-major 순서, len == rows*cols
    global_median_dx: Optional[float] = None
    global_median_dy: Optional[float] = None
    global_median_strength: Optional[float] = None
    # fit stability = 채워진(비어있지 않은) cell 비율(0~1) - 낮을수록 dataset
    # coverage가 부족해 빈 cell을 fallback으로 채운 비중이 크다는 뜻이다.
    fit_stability: Optional[float] = None


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
    diagnostics: Optional[GhostFieldDiagnostics] = None


@dataclass
class GhostReconstructionMetrics:
    """Synthetic GT(clean 원본을 알고 있는 경우)에서만 계산 가능한
    reconstruction 품질 지표(STEP 8 stabilization 7-D번) - real footage
    평가에는 쓰이지 않는다."""
    mae: float
    rmse: float
    psnr_db: Optional[float] = None


@dataclass
class GhostSuppressionResult:
    success: bool

    suppressed_image: Optional[np.ndarray] = None
    predicted_ghost_image: Optional[np.ndarray] = None
    correction_map: Optional[np.ndarray] = None

    iterations: int = 0
    mean_correction: float = 0.0
    max_correction: float = 0.0

    # Detail Retention / Over-Suppression(STEP 8 stabilization 7번) - Ghost
    # 제거만으로는 성공이 아니다: main edge가 유지되고 clean 영역에
    # 불필요한 변화가 없어야 한다.
    clean_region_change: Optional[float] = None
    edge_retention: Optional[float] = None
    over_suppression_score: Optional[float] = None

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
