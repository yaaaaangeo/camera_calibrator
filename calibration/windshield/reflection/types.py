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
    mean_strength: float = 0.0
    median_strength: float = 0.0
    p95_strength: float = 0.0
    worst_pair_id: Optional[str] = None
    coverage: float = 0.0
    severity_score: Optional[float] = None
    by_day_night: dict[str, dict[str, float]] = field(default_factory=dict)
    success: bool = True
    warning_message: Optional[str] = None
    error_message: Optional[str] = None
