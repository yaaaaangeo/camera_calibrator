"""
Photometric windshield reflection evaluation data types.

This module is intentionally independent from the windshield geometry models.
Reflection evaluation measures image-domain photometric artifacts; it must not
modify camera intrinsics, poses, or windshield geometry parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


REFLECTION_METRIC_VERSION = 1


@dataclass
class ReflectionEvaluationConfig:
    mode: str = "reference"  # "reference" or "no_reference"
    coverage_threshold: float = 0.08
    saturation_threshold: float = 250.0
    glare_luminance_threshold: float = 220.0
    glare_contrast_threshold: float = 12.0
    spatial_rows: int = 4
    spatial_cols: int = 6
    align: bool = True
    alignment_model: str = "translation"
    photometric_normalize: bool = True
    allow_unsafe_reference_bypass: bool = False
    automotive_bottom_roi_fraction: float = 0.25


@dataclass
class ReflectionImagePair:
    normal_image_path: str
    reference_image_path: Optional[str] = None
    pair_id: str = ""
    day_night: Optional[str] = None
    exposure_normal: Optional[float] = None
    exposure_reference: Optional[float] = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ReflectionRegionMetrics:
    mean_strength: float = 0.0
    p95_strength: float = 0.0
    coverage: float = 0.0


@dataclass
class ReflectionSpatialCell:
    row: int
    col: int
    mean_strength: float
    p95_strength: float
    coverage: float


@dataclass
class ReflectionEvaluationResult:
    mode: str
    metric_version: int = REFLECTION_METRIC_VERSION
    pair_id: str = ""

    # Canonical reference-mode reflection metrics. These mirror the legacy
    # *_strength fields below so downstream callers can use explicit names.
    reflection_mean: Optional[float] = None
    reflection_median: Optional[float] = None
    reflection_p95: Optional[float] = None
    reflection_p99: Optional[float] = None
    reflection_max: Optional[float] = None
    reflection_coverage: Optional[float] = None

    # No-reference mode is a heuristic likelihood, not ground truth.
    reflection_likelihood: Optional[float] = None
    no_reference_is_likelihood: bool = False

    mean_strength: float = 0.0
    median_strength: float = 0.0
    p95_strength: float = 0.0
    p99_strength: float = 0.0
    max_strength: float = 0.0
    positive_mean_strength: float = 0.0
    positive_p95_strength: float = 0.0
    coverage: float = 0.0
    coverage_threshold: float = 0.08
    saturation_threshold: float = 250.0
    glare_luminance_threshold: float = 220.0
    glare_contrast_threshold: float = 12.0
    severity_score: Optional[float] = None
    saturation_coverage: float = 0.0
    glare_coverage: Optional[float] = None
    glare_strength: Optional[float] = None
    contrast_retention: Optional[float] = None
    edge_retention: Optional[float] = None
    bottom_roi_mean_strength: Optional[float] = None
    bottom_roi_coverage: Optional[float] = None
    regional_metrics: dict[str, ReflectionRegionMetrics] = field(default_factory=dict)
    spatial_map: list[ReflectionSpatialCell] = field(default_factory=list)
    alignment_score: Optional[float] = None
    alignment_error_px: Optional[float] = None
    alignment_status: str = "not_run"
    alignment_method: str = "none"
    # Phase B-2 안정화 - Affine alignment 사용 시 사람이 읽을 수 있는
    # 진단(사용자 스펙). Translation 모드에서는 rotation_deg=0/scale=1/
    # shear_deg=0으로 고정된다.
    alignment_translation_x_px: Optional[float] = None
    alignment_translation_y_px: Optional[float] = None
    alignment_rotation_deg: Optional[float] = None
    alignment_scale: Optional[float] = None
    alignment_shear_deg: Optional[float] = None
    photometric_normalized: bool = False
    photometric_gain: Optional[float] = None
    photometric_bias: Optional[float] = None
    heatmap_rows: int = 0
    heatmap_cols: int = 0
    downsampled_reflection_map: list[list[float]] = field(default_factory=list)
    warning_message: Optional[str] = None
    error_message: Optional[str] = None
    success: bool = True


@dataclass
class ReflectionDatasetResult:
    mode: str
    metric_version: int = REFLECTION_METRIC_VERSION
    pair_results: list[ReflectionEvaluationResult] = field(default_factory=list)
    reference_mean_strength: Optional[float] = None
    reference_p95_strength: Optional[float] = None
    reference_coverage: Optional[float] = None
    mean_reflection_likelihood: Optional[float] = None
    p95_reflection_likelihood: Optional[float] = None
    mean_strength: float = 0.0
    median_strength: float = 0.0
    p95_strength: float = 0.0
    worst_pair_id: Optional[str] = None
    coverage: float = 0.0
    severity_score: Optional[float] = None
    by_day_night: dict[str, dict[str, float]] = field(default_factory=dict)
    # Phase A-7 안정화 - 개별 pair 성공/실패 개수를 명시적으로 남긴다(예:
    # "Valid 18 / Invalid 2") - `pair_results`를 순회해서 매번 세지 않아도
    # UI/로그에서 바로 쓸 수 있다. Ghost의 num_valid_frames/num_failed_frames
    # 와 동일한 패턴.
    num_valid_pairs: int = 0
    num_invalid_pairs: int = 0
    success: bool = True
    warning_message: Optional[str] = None
    error_message: Optional[str] = None
